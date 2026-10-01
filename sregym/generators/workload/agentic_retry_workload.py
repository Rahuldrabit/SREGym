"""Live open-loop HTTP workload engine and metrics collector for Agentic Retry Platform."""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from sregym.generators.workload.k8s_port_forward import KubectlPortForward

logger = logging.getLogger("all.infra.agentic_retry_workload")


@dataclass
class WorkloadSnapshot:
    submitted: int
    completed: int
    succeeded: int
    failed: int
    actual_rate: float
    success_rate: float
    p95_latency_seconds: float
    amplification_ratio: float
    backend_queue_depth: int
    backend_waiting_requests: int
    backend_active_requests: int
    branch_amplification_ratio: float = 1.0
    goodput_rate: float = 10.0
    orphaned_operations: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


class AgenticRetryWorkload:
    """Live open-loop HTTP workload generator and live metrics collector for AgenticRetryPlatform."""

    def __init__(
        self,
        namespace: str = "agentic-retry-platform",
        base_rate: float = 10.0,
        concurrency_limit: int = 25,
        normal_latency: float = 0.10,
        fault_latency: float = 1.50,
        trigger_duration_seconds: float = 10.0,
        planner_max_retries: int = 3,
        tool_max_retries: int = 2,
        transport_max_retries: int = 2,
        test_mode: bool = False,
    ):
        self.namespace = namespace
        self.base_rate = base_rate
        self.concurrency_limit = concurrency_limit
        self.normal_latency = normal_latency
        self.fault_latency = fault_latency
        self.trigger_duration_seconds = trigger_duration_seconds

        self.planner_max_retries = planner_max_retries
        self.tool_max_retries = tool_max_retries
        self.transport_max_retries = transport_max_retries
        self.test_mode = test_mode

        self.orchestrator_forward = KubectlPortForward(self.namespace, "agent-orchestrator", 8000)
        self.gateway_forward = KubectlPortForward(self.namespace, "tool-gateway", 8001)
        self.data_api_forward = KubectlPortForward(self.namespace, "data-api", 8003)

        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

        self._requests_history: deque[tuple[float, bool, float]] = deque(maxlen=2000)
        self._submitted_count = 0
        self._completed_count = 0
        self._succeeded_count = 0
        self._failed_count = 0
        self._fault_active = False
        self._fault_triggered = False
        self._mitigation_applied = False

        # Live metric baseline tracking for rate and amplification deltas
        self._last_snapshot_time = time.time()
        self._last_logical_workflows = 0.0
        self._last_backend_requests = 0.0
        self._last_tool_operations = 0.0
        self._last_goodput_requests = 0.0
        self._metric_baseline_initialized = False

    def start(self):
        """Starts port forwards and background open-loop traffic generator."""
        if self._running:
            return

        self._executor = ThreadPoolExecutor(max_workers=30)
        self._running = True
        if not self.test_mode:
            self.orchestrator_forward.start()
            self.gateway_forward.start()
            self.data_api_forward.start()

        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()
        logger.info(f"AgenticRetryWorkload started at {self.base_rate} req/s against namespace {self.namespace}")

    def stop(self):
        """Stops background traffic generator and tears down tunnels."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        self.orchestrator_forward.stop()
        self.gateway_forward.stop()
        self.data_api_forward.stop()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        logger.info("AgenticRetryWorkload stopped")

    def _send_single_request(self, req_id: str, wf_id: str):
        port = self.orchestrator_forward.local_port or 8000
        url = f"http://127.0.0.1:{port}/agent/workflow"
        payload = json.dumps({"logical_request_id": req_id, "workflow_id": wf_id}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-Logical-Request-ID", req_id)
        req.add_header("X-Workflow-ID", wf_id)

        start = time.time()
        success = False
        duration = 0.0

        try:
            with urllib.request.urlopen(req, timeout=14.0) as resp:
                success = resp.status == 200
                duration = time.time() - start
        except Exception:
            duration = time.time() - start

        with self._lock:
            self._completed_count += 1
            if success:
                self._succeeded_count += 1
            else:
                self._failed_count += 1
            self._requests_history.append((time.time(), success, duration))

    def _run_loop(self):
        interval = 1.0 / max(1.0, self.base_rate)
        req_counter = 0

        while self._running:
            loop_start = time.time()
            req_counter += 1
            run_id = uuid.uuid4().hex
            req_id = f"req-{run_id}-{req_counter}"
            wf_id = f"wf-{run_id}"

            with self._lock:
                self._submitted_count += 1

            executor = self._executor
            if executor is None or not self._running:
                break
            try:
                executor.submit(self._send_single_request, req_id, wf_id)
            except RuntimeError:
                break

            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def inject_latency_fault(self, latency_ms: float = 1500.0, duration_seconds: float = 10.0):
        """Sends HTTP POST to Data API /admin/fault on control port to inject transient latency perturbation."""
        self._fault_active = True
        self._fault_triggered = True
        if not self.test_mode and not self.data_api_forward._healthy():
            self.data_api_forward.start()
        port = self.data_api_forward.local_port or 8003
        url = f"http://127.0.0.1:{port}/admin/fault"
        payload = json.dumps({"latency_ms": latency_ms, "duration_seconds": duration_seconds}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                logger.info(f"Injected cluster latency fault: {resp.read().decode('utf-8')}")
        except Exception as e:
            logger.warning(f"Failed to post to /admin/fault on cluster: {e}")
            if not self.test_mode:
                raise RuntimeError(f"Failed to inject backend latency fault on live cluster: {e}") from e

    def remove_latency_fault(self):
        """Sends HTTP POST to Data API /admin/recover on control port."""
        self._fault_active = False
        if not self.test_mode and not self.data_api_forward._healthy():
            self.data_api_forward.start()
        port = self.data_api_forward.local_port or 8003
        url = f"http://127.0.0.1:{port}/admin/recover"
        req = urllib.request.Request(url, data=b"{}", method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                logger.info(f"Recovered cluster latency fault: {resp.read().decode('utf-8')}")
        except Exception as e:
            logger.warning(f"Failed to post to /admin/recover on cluster: {e}")
            if not self.test_mode:
                raise RuntimeError(f"Failed to recover backend latency fault on live cluster: {e}") from e

    def apply_mitigation(
        self,
        cap_planner_retries: int = 1,
        disable_nested_retries: bool = True,
        enable_backoff: bool = True,
        shed_stale_queue: bool = True,
        enable_cancellation: bool = True,
        enable_retry_budget: bool = True,
    ):
        """Applies orchestrator and gateway mitigation settings via ConfigMap patch."""
        self.planner_max_retries = cap_planner_retries
        if disable_nested_retries:
            self.tool_max_retries = 1
            self.transport_max_retries = 1
        self._fault_active = False
        self._fault_triggered = False
        self._mitigation_applied = True

        policy = {
            "workflow": {
                "timeout_ms": 2500,
                "max_attempts": cap_planner_retries,
                "max_inflight": 25,
                "cancel_children_on_timeout": enable_cancellation,
            },
            "tool": {"timeout_ms": 1000, "max_attempts": 1 if disable_nested_retries else 2},
            "transport": {"timeout_ms": 600, "max_attempts": 1 if disable_nested_retries else 2},
            "retry_budget": {"enabled": enable_retry_budget, "max_physical_attempts_per_workflow": 4},
        }
        if self.test_mode:
            return

        patch = json.dumps({"data": {"policy.json": json.dumps(policy)}})
        subprocess.run(
            [
                "kubectl",
                "patch",
                "configmap",
                "agentic-retry-policy",
                "-n",
                self.namespace,
                "--type",
                "merge",
                "-p",
                patch,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # Projected ConfigMaps update asynchronously. Restart the consumers so the
        # policy boundary is deterministic, then prove both live services read it.
        for deployment in ("agent-orchestrator", "tool-gateway"):
            subprocess.run(
                ["kubectl", "rollout", "restart", f"deployment/{deployment}", "-n", self.namespace],
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                ["kubectl", "rollout", "status", f"deployment/{deployment}", "-n", self.namespace, "--timeout=90s"],
                capture_output=True,
                text=True,
                check=True,
            )

        for forward in (self.orchestrator_forward, self.gateway_forward):
            # A service port-forward chooses one endpoint when it starts. It can
            # remain connected to a terminating pre-rollout pod, so recreate it
            # before using /config as evidence of the new policy.
            forward.stop()
            port = forward.start()
            request = urllib.request.Request(f"http://127.0.0.1:{port}/config")
            with urllib.request.urlopen(request, timeout=3.0) as response:
                observed = json.loads(response.read().decode("utf-8"))
            if observed != policy:
                raise RuntimeError(f"Live retry policy did not converge after rollout: {observed}")
        self.reset_metric_baseline()

    def reset_metric_baseline(self):
        """Immediately captures current counter values to form baseline for next delta."""
        now = time.time()
        if self.test_mode:
            self._last_snapshot_time = now
            self._metric_baseline_initialized = True
            return

        orch_port = self.orchestrator_forward.local_port or 8000
        gw_port = self.gateway_forward.local_port or 8001
        data_port = self.data_api_forward.local_port or 8003

        orch = self._scrape_prometheus_metrics(orch_port)
        gw = self._scrape_prometheus_metrics(gw_port)
        data = self._scrape_prometheus_metrics(data_port)

        self._last_logical_workflows = orch.get("logical_workflows_total", 0.0)
        self._last_goodput_requests = orch.get("goodput_requests_total", 0.0)
        self._last_tool_operations = gw.get("tool_operations_started_total", 0.0)
        self._last_backend_requests = data.get("backend_requests_total", 0.0)

        self._last_snapshot_time = now
        self._metric_baseline_initialized = True

    def _scrape_prometheus_metrics(self, port: int) -> dict[str, float]:
        """Scrapes and parses Prometheus metrics format from live endpoint."""
        url = f"http://127.0.0.1:{port}/metrics"
        req = urllib.request.Request(url)
        metrics: dict[str, float] = {}
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                content = resp.read().decode("utf-8")
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        name = parts[0].split("{")[0]
                        with contextlib.suppress(ValueError):
                            metrics[name] = float(parts[1])
            return metrics
        except Exception:
            return {}

    def snapshot(self, window_seconds: float = 5.0) -> WorkloadSnapshot:
        """Samples metrics over the given window. Reads live metrics from pods without synthetic fallbacks."""
        now = time.time()
        window_start = now - window_seconds
        elapsed_delta = max(0.5, now - self._last_snapshot_time)

        with self._lock:
            recent = [item for item in self._requests_history if item[0] >= window_start]
            total_recent = len(recent)
            succeeded_recent = sum(1 for item in recent if item[1])

            durations = sorted([item[2] for item in recent])
            p95 = durations[int(len(durations) * 0.95)] if durations else 0.25
            rate = total_recent / max(1.0, window_seconds)
            client_success_rate = (succeeded_recent / total_recent) if total_recent > 0 else 0.95

            submitted_count = self._submitted_count
            completed_count = self._completed_count
            succeeded_count = self._succeeded_count
            failed_count = self._failed_count

        orch_port = self.orchestrator_forward.local_port or 8000
        gw_port = self.gateway_forward.local_port or 8001
        data_port = self.data_api_forward.local_port or 8003

        orch_metrics = self._scrape_prometheus_metrics(orch_port)
        gw_metrics = self._scrape_prometheus_metrics(gw_port)
        data_metrics = self._scrape_prometheus_metrics(data_port)

        # In live benchmark evaluation, metrics MUST be readable from the cluster
        if not self.test_mode:
            required_metrics = {
                "agent-orchestrator": (
                    orch_metrics,
                    {
                        "logical_workflows_total",
                        "goodput_requests_total",
                        "workflow_generation_total",
                    },
                ),
                "tool-gateway": (gw_metrics, {"tool_operations_started_total"}),
                "data-api": (
                    data_metrics,
                    {
                        "backend_requests_total",
                        "backend_waiting_requests",
                        "backend_active_requests",
                        "data_api_waiting_requests",
                        "orphaned_operations_active",
                    },
                ),
            }
            missing = [
                f"{service}: {sorted(names - set(metrics))}"
                for service, (metrics, names) in required_metrics.items()
                if names - set(metrics)
            ]
            if missing:
                raise RuntimeError(
                    "Live cluster metrics incomplete; refusing to synthesize benchmark state: " + "; ".join(missing)
                )

        if orch_metrics or data_metrics or gw_metrics:
            # 1. Real metrics from live cluster pods
            logical_total = orch_metrics.get("logical_workflows_total", 0.0)
            goodput_total = orch_metrics.get("goodput_requests_total", 0.0)
            tool_ops_total = gw_metrics.get("tool_operations_started_total", 0.0)
            backend_total = data_metrics.get("backend_requests_total", 0.0)

            has_baseline = self._metric_baseline_initialized
            delta_logical = max(0.0, logical_total - self._last_logical_workflows) if has_baseline else 0.0
            delta_backend = max(0.0, backend_total - self._last_backend_requests) if has_baseline else 0.0
            delta_tool = max(0.0, tool_ops_total - self._last_tool_operations) if has_baseline else 0.0
            delta_goodput = max(0.0, goodput_total - self._last_goodput_requests) if has_baseline else 0.0

            self._last_snapshot_time = now
            self._last_logical_workflows = logical_total
            self._last_backend_requests = backend_total
            self._last_tool_operations = tool_ops_total
            self._last_goodput_requests = goodput_total
            self._metric_baseline_initialized = True

            amp_ratio = (delta_backend / delta_logical) if delta_logical > 0 else 1.0
            branch_amp = (delta_tool / delta_logical) if delta_logical > 0 else 1.0
            goodput_rate = delta_goodput / elapsed_delta

            queue_depth = int(data_metrics.get("backend_waiting_requests", 0))
            backend_waiting_requests = int(data_metrics["data_api_waiting_requests"])
            active_workers = int(data_metrics["backend_active_requests"])
            orphaned = int(data_metrics["orphaned_operations_active"])

            success_rate = delta_goodput / delta_logical if delta_logical > 0 else client_success_rate

            return WorkloadSnapshot(
                submitted=submitted_count,
                completed=completed_count,
                succeeded=succeeded_count,
                failed=failed_count,
                actual_rate=rate,
                success_rate=success_rate,
                p95_latency_seconds=p95,
                amplification_ratio=amp_ratio,
                backend_queue_depth=queue_depth,
                backend_waiting_requests=backend_waiting_requests,
                backend_active_requests=active_workers,
                branch_amplification_ratio=branch_amp,
                goodput_rate=goodput_rate,
                orphaned_operations=orphaned,
                metrics={
                    "amplification_ratio": amp_ratio,
                    "branch_amplification_ratio": branch_amp,
                    "queue_depth": queue_depth,
                    "backend_waiting_requests": backend_waiting_requests,
                    "active_workers": active_workers,
                    "goodput_rate": goodput_rate,
                    "orphaned_operations": orphaned,
                },
            )

        # 2. Test mode mock double (active ONLY during offline unit tests when test_mode=True)
        is_metastable = self._fault_active or (self._fault_triggered and not self._mitigation_applied)
        amp_ratio = 9.5 if self._fault_active else (2.4 if is_metastable else 1.05)
        branch_amp = 3.0 if self._fault_active else (2.0 if is_metastable else 1.0)
        queue_depth = 45 if self._fault_active else (25 if is_metastable else 0)
        db_waiting = queue_depth
        active_workers = 25 if is_metastable else 8
        orphaned = 24 if is_metastable else 0
        success_rate = 0.40 if self._fault_active else (0.50 if is_metastable else 0.98)
        goodput = rate * success_rate

        return WorkloadSnapshot(
            submitted=submitted_count,
            completed=completed_count,
            succeeded=succeeded_count,
            failed=failed_count,
            actual_rate=rate,
            success_rate=success_rate,
            p95_latency_seconds=p95,
            amplification_ratio=amp_ratio,
            backend_queue_depth=queue_depth,
            backend_waiting_requests=db_waiting,
            backend_active_requests=active_workers,
            branch_amplification_ratio=branch_amp,
            goodput_rate=goodput,
            orphaned_operations=orphaned,
            metrics={
                "amplification_ratio": amp_ratio,
                "branch_amplification_ratio": branch_amp,
                "queue_depth": queue_depth,
                "backend_waiting_requests": db_waiting,
                "active_workers": active_workers,
                "goodput_rate": goodput,
                "orphaned_operations": orphaned,
            },
        )
