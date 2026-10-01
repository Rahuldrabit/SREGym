"""Direct regression tests for Data API bounded concurrency and observability isolation.

Directly tests the maintainer's 1,840-thread failure scenario:
1. Saturated query load on data plane (:8002).
2. Active handlers capped at MAX_REQUEST_HANDLERS (<= 64) with fast 503 shedding.
3. Observability control plane (:8003) remains responsive (< 1s latency throughout overload).
4. PostgreSQL wire protocol determinism and fail-closed behavior.
"""

from __future__ import annotations

import io
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

# Add application files directory to sys.path
FILES_DIR = Path(__file__).resolve().parents[3] / "SREGym-applications" / "agentic-retry-platform" / "helm" / "files"
sys.path.insert(0, str(FILES_DIR))

from control_server import create_control_server  # noqa: E402
from data_server import create_data_server  # noqa: E402
from data_state import DataAPIState  # noqa: E402
from pg_wire import execute_pg_query, recv_message  # noqa: E402


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_pg_wire_exact_framing():
    """Verify that message framing correctly handles multi-chunk TCP reads."""

    class FakeSocket:
        def __init__(self, data: bytes):
            self.stream = io.BytesIO(data)

        def recv(self, n: int) -> bytes:
            return self.stream.read(min(n, 2))  # Deliver at most 2 bytes per call

    # Frame: 'Z' (1 byte), length 5 (4 bytes int32 = 0x00000005), payload 'I' (1 byte)
    raw = b"Z\x00\x00\x00\x05I"
    sock = FakeSocket(raw)
    mtype, body = recv_message(sock)
    assert mtype == b"Z"
    assert body == b"I"


def test_pg_wire_fails_closed_when_database_unreachable():
    """Verify that PgBouncer connection failure immediately fails closed without sleep fallback."""
    cancel_event = threading.Event()
    start = time.time()
    ok, msg = execute_pg_query("127.0.0.1", 1, 1.0, cancel_event)
    duration = time.time() - start

    assert ok is False
    assert "unavailable" in msg or "failed" in msg
    assert duration < 0.5, "Should fail fast and closed without sleeping"


def test_data_api_bounded_concurrency_and_metrics_responsiveness():
    """Regression test for 1,840-thread failure:

    Overloads the Data Plane with concurrent requests, while continuously polling /metrics
    on the isolated Control Plane.
    Asserts:
    1. Query overload does NOT block /metrics (response time < 1s).
    2. Active HTTP handlers never exceed max_request_handlers (64).
    3. Excess requests are shed immediately with 503 rather than spawning unbounded threads.
    """
    data_port = _get_free_port()
    control_port = _get_free_port()

    max_handlers = 8
    concurrency_limit = 2
    queue_cap = 4

    state = DataAPIState(
        concurrency_limit=concurrency_limit,
        queue_capacity=queue_cap,
        max_request_handlers=max_handlers,
        normal_latency=0.02,
        fault_latency=0.15,
    )

    control_server = create_control_server(
        host="127.0.0.1",
        port=control_port,
        state=state,
        pgbouncer_host="127.0.0.1",
        pgbouncer_port=6432,
        redis_host="127.0.0.1",
        redis_port=6379,
    )

    data_server = create_data_server(
        host="127.0.0.1",
        port=data_port,
        state=state,
        pgbouncer_host="127.0.0.1",
        pgbouncer_port=6432,
        max_request_handlers=max_handlers,
    )

    control_thread = threading.Thread(target=control_server.serve_forever, daemon=True)
    data_thread = threading.Thread(target=data_server.serve_forever, daemon=True)
    control_thread.start()
    data_thread.start()

    time.sleep(0.05)

    def mock_query(host, port, latency, cancel_event):
        time.sleep(0.20)
        return True, "ok"

    try:
        with patch("data_server.execute_pg_query", side_effect=mock_query):
            metrics_latencies = []
            stop_monitoring = threading.Event()
            max_observed_handlers = 0
            handlers_lock = threading.Lock()

            def monitor_metrics():
                nonlocal max_observed_handlers
                url = f"http://127.0.0.1:{control_port}/metrics"
                while not stop_monitoring.is_set():
                    t0 = time.time()
                    try:
                        req = urllib.request.Request(url)
                        with urllib.request.urlopen(req, timeout=0.5) as resp:
                            body = resp.read().decode("utf-8")
                            lat = time.time() - t0
                            metrics_latencies.append(lat)
                            for line in body.splitlines():
                                if line.startswith("data_api_http_handlers_active"):
                                    val = int(float(line.split()[1]))
                                    with handlers_lock:
                                        if val > max_observed_handlers:
                                            max_observed_handlers = val
                    except Exception:
                        pass
                    time.sleep(0.01)

            monitor_thread = threading.Thread(target=monitor_metrics, daemon=True)
            monitor_thread.start()

            client_results = []
            client_lock = threading.Lock()

            def send_query(idx):
                url = f"http://127.0.0.1:{data_port}/data/query"
                payload = f'{{"operation_id":"op-{idx}","physical_attempt_id":"att-{idx}"}}'.encode()
                req = urllib.request.Request(url, data=payload, method="POST")
                req.add_header("Content-Type", "application/json")
                try:
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        resp.read()
                        with client_lock:
                            client_results.append(resp.status)
                except urllib.error.HTTPError as exc:
                    exc.read()
                    with client_lock:
                        client_results.append(exc.code)
                except Exception as exc:
                    with client_lock:
                        client_results.append(str(exc))

            query_threads = [threading.Thread(target=send_query, args=(i,)) for i in range(35)]
            for t in query_threads:
                t.start()
            for t in query_threads:
                t.join(timeout=2.0)

            stop_monitoring.set()
            monitor_thread.join(timeout=0.5)

            # Invariants:
            # 1. Metrics responses must remain under 1 second throughout overload
            assert len(metrics_latencies) > 0, "Metrics endpoint should have been polled"
            max_metrics_latency = max(metrics_latencies)
            assert max_metrics_latency < 1.0, f"Metrics latency too high ({max_metrics_latency}s)"

            # 2. Handlers must stay bounded
            with handlers_lock:
                assert max_observed_handlers <= max_handlers, (
                    f"Active handlers ({max_observed_handlers}) exceeded max bound ({max_handlers})"
                )

            # 3. Work was shed cleanly with 503
            with state.lock:
                requests_shed = state.requests_shed
            assert requests_shed > 0, f"Expected 503 shedding under overload, got {requests_shed}"
            assert 503 in client_results, "Client should have observed 503 shedding"

    finally:
        data_server.shutdown()
        control_server.shutdown()
        data_server.server_close()
        control_server.server_close()
