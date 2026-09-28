"""Agentic workflow simulation and workload engine for metastable retry overload.

Models multi-layer retry amplification across:
    Load Generator
        ↓
    Agent Supervisor / Workflow Planner (R_planner)
        ↓
    Tool Service (R_tool)
        ↓
    Transport / HTTP Client (R_transport)
        ↓
    Backend Service (Finite concurrency pool C, queue, DB pool)

Under healthy conditions, 1 logical request produces ~1 backend attempt.
A transient backend latency spike triggers uncoordinated retries across layers,
multiplying attempts (e.g. 2 x 2 x 3 = 12x), saturating the finite concurrency
pool and establishing a self-sustaining metastable overload loop that persists
after the transient trigger is removed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("all.infra.agentic_workflow")


@dataclass
class RequestContext:
    logical_request_id: str
    workflow_id: str
    attempt: int
    retry_layer: str  # "initial", "transport", "tool", "planner"
    parent_attempt_id: str | None = None
    created_at: float = field(default_factory=time.time)


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
    db_pool_waiting: int
    backend_active_requests: int
    metrics: dict[str, Any] = field(default_factory=dict)


class TransportTimeoutError(Exception):
    """Raised when transport client times out waiting for backend."""


class ToolTimeoutError(Exception):
    """Raised when tool execution times out."""


class WorkflowFailedError(Exception):
    """Raised when agent workflow fails after exhausting replan retries."""


class BackendService:
    """Finite concurrency backend representing workers and database connection pool."""

    def __init__(
        self,
        concurrency_limit: int = 25,
        queue_capacity: int = 300,
        normal_latency: float = 0.25,
        fault_latency: float = 1.5,
        clock_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self.concurrency_limit = concurrency_limit
        self.queue_capacity = queue_capacity
        self.normal_latency = normal_latency
        self.fault_latency = fault_latency
        self.fault_active = False

        self._clock = clock_fn or time.time
        self._sleep = sleep_fn or time.sleep
        self._lock = threading.Lock()

        # Telemetry state
        self.active_workers: int = 0
        self.waiting_queue: deque[tuple[RequestContext, float, threading.Event, list[Any], list[bool]]] = deque()
        self.total_backend_attempts: int = 0
        self.total_completed: int = 0
        self.total_dropped: int = 0
        self.backend_attempt_events: deque[float] = deque()

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return len(self.waiting_queue)

    @property
    def db_pool_active(self) -> int:
        with self._lock:
            return self.active_workers

    @property
    def db_pool_waiting(self) -> int:
        with self._lock:
            return len(self.waiting_queue)

    def execute(self, ctx: RequestContext, timeout_seconds: float, drop_stale: bool = False) -> tuple[bool, float]:
        """Execute request with finite concurrency."""
        start_time = self._clock()
        done_event = threading.Event()
        result_box: list[Any] = [False, 0.0]
        abandoned_box: list[bool] = [False]

        with self._lock:
            self.total_backend_attempts += 1
            self.backend_attempt_events.append(start_time)
            if len(self.waiting_queue) >= self.queue_capacity:
                self.total_dropped += 1
                logger.warning(
                    f"backend: worker queue full logical_request={ctx.logical_request_id} "
                    f"queue_depth={len(self.waiting_queue)}"
                )
                return False, self._clock() - start_time

            # Queue request
            self.waiting_queue.append((ctx, start_time, done_event, result_box, abandoned_box))
            self._try_dispatch_locked()

        # Wait for worker completion or timeout
        signaled = done_event.wait(timeout=max(0.001, timeout_seconds))
        elapsed = self._clock() - start_time

        if not signaled:
            abandoned_box[0] = True
            if self.db_pool_waiting > 5:
                logger.warning(
                    f"postgres: connection acquisition timeout logical_request={ctx.logical_request_id} "
                    f"waiting={self.db_pool_waiting}"
                )
            if drop_stale:
                with self._lock:
                    self.waiting_queue = deque([item for item in self.waiting_queue if item[0] is not ctx])
            raise TransportTimeoutError(f"Backend call timed out after {elapsed:.3f}s (timeout={timeout_seconds}s)")

        return result_box[0], result_box[1]

    def _try_dispatch_locked(self):
        while self.active_workers < self.concurrency_limit and self.waiting_queue:
            ctx, enqueued_at, done_event, result_box, abandoned_box = self.waiting_queue.popleft()
            self.active_workers += 1

            threading.Thread(
                target=self._worker_run,
                args=(ctx, enqueued_at, done_event, result_box, abandoned_box),
                daemon=True,
            ).start()

    def _worker_run(
        self,
        ctx: RequestContext,
        enqueued_at: float,
        done_event: threading.Event,
        result_box: list[Any],
        abandoned_box: list[bool],
    ):
        try:
            latency = self.fault_latency if self.fault_active else self.normal_latency
            self._sleep(latency)
            result_box[0] = True
            result_box[1] = latency
        finally:
            done_event.set()
            with self._lock:
                self.active_workers -= 1
                self.total_completed += 1
                self._try_dispatch_locked()


class TransportClient:
    """HTTP/gRPC transport client with retry policy."""

    def __init__(
        self,
        backend: BackendService,
        timeout_seconds: float = 0.6,
        max_retries: int = 2,
        backoff_seconds: float = 0.0,
        clock_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self.backend = backend
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._clock = clock_fn or time.time
        self._sleep = sleep_fn or time.sleep
        self.transport_retries_total: int = 0
        self.drop_stale: bool = False

    def call(self, ctx: RequestContext) -> tuple[bool, float]:
        attempt = 1
        parent_id = ctx.parent_attempt_id

        while attempt <= self.max_retries:
            sub_ctx = RequestContext(
                logical_request_id=ctx.logical_request_id,
                workflow_id=ctx.workflow_id,
                attempt=attempt,
                retry_layer="initial" if attempt == 1 else "transport",
                parent_attempt_id=parent_id,
                created_at=self._clock(),
            )
            try:
                success, dur = self.backend.execute(
                    sub_ctx,
                    timeout_seconds=self.timeout_seconds,
                    drop_stale=self.drop_stale,
                )
                return success, dur
            except TransportTimeoutError:
                if attempt < self.max_retries:
                    self.transport_retries_total += 1
                    logger.info(
                        f"http-client: request={ctx.logical_request_id} retry={attempt} "
                        f"timeout after {int(self.timeout_seconds * 1000)}ms"
                    )
                    if self.backoff_seconds > 0:
                        self._sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                    attempt += 1
                else:
                    raise


class ToolService:
    """Tool service client with tool-level retry policy."""

    def __init__(
        self,
        transport: TransportClient,
        timeout_seconds: float = 1.4,
        max_retries: int = 2,
        clock_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self.transport = transport
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._clock = clock_fn or time.time
        self._sleep = sleep_fn or time.sleep
        self.tool_retries_total: int = 0

    def call_tool(self, ctx: RequestContext) -> tuple[bool, float]:
        attempt = 1
        start_time = self._clock()

        while attempt <= self.max_retries:
            sub_ctx = RequestContext(
                logical_request_id=ctx.logical_request_id,
                workflow_id=ctx.workflow_id,
                attempt=attempt,
                retry_layer="tool" if attempt > 1 else ctx.retry_layer,
                parent_attempt_id=f"tool-{attempt}",
                created_at=self._clock(),
            )
            try:
                return self.transport.call(sub_ctx)
            except TransportTimeoutError:
                elapsed = self._clock() - start_time
                if attempt < self.max_retries and elapsed < self.timeout_seconds:
                    self.tool_retries_total += 1
                    logger.info(f"tool-service: request={ctx.logical_request_id} timeout after {int(elapsed * 1000)}ms")
                    attempt += 1
                else:
                    raise ToolTimeoutError(f"Tool call failed after {elapsed:.3f}s")


class AgentWorkflowService:
    """Agent workflow supervisor and planner with replanning loop."""

    def __init__(
        self,
        tool_service: ToolService,
        max_retries: int = 3,
        timeout_seconds: float = 3.5,
        clock_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self.tool_service = tool_service
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self._clock = clock_fn or time.time
        self._sleep = sleep_fn or time.sleep

        self.planner_retries_total: int = 0
        self.active_workflows: int = 0
        self.workflow_queue_depth: int = 0
        self._lock = threading.Lock()

    def run_workflow(self, logical_request_id: str, workflow_id: str) -> bool:
        with self._lock:
            self.active_workflows += 1

        start_time = self._clock()
        attempt = 1
        try:
            while attempt <= self.max_retries:
                elapsed = self._clock() - start_time
                if elapsed >= self.timeout_seconds:
                    return False

                ctx = RequestContext(
                    logical_request_id=logical_request_id,
                    workflow_id=workflow_id,
                    attempt=attempt,
                    retry_layer="initial" if attempt == 1 else "planner",
                    parent_attempt_id=f"wf-plan-{attempt}",
                    created_at=self._clock(),
                )
                try:
                    success, _ = self.tool_service.call_tool(ctx)
                    return success
                except ToolTimeoutError:
                    logger.info(f"agent-worker: workflow={workflow_id} tool returned timeout")
                    elapsed = self._clock() - start_time
                    if attempt < self.max_retries and elapsed < self.timeout_seconds:
                        logger.info(f"agent-planner: workflow={workflow_id} insufficient evidence; replanning")
                        with self._lock:
                            self.planner_retries_total += 1
                        attempt += 1
                    else:
                        return False
            return False
        finally:
            with self._lock:
                self.active_workflows -= 1


class AgenticWorkflowWorkload:
    """Workload generator, telemetry collector, and orchestrator."""

    def __init__(
        self,
        base_rate: float = 10.0,
        concurrency_limit: int = 25,
        normal_latency: float = 0.25,
        fault_latency: float = 1.5,
        transport_timeout: float = 0.6,
        tool_timeout: float = 1.4,
        planner_timeout: float = 3.5,
        transport_max_retries: int = 2,
        tool_max_retries: int = 2,
        planner_max_retries: int = 3,
        clock_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self.base_rate = base_rate
        self.current_rate = base_rate
        self._clock = clock_fn or time.time
        self._sleep = sleep_fn or time.sleep

        # Initialize sub-services
        self.backend = BackendService(
            concurrency_limit=concurrency_limit,
            normal_latency=normal_latency,
            fault_latency=fault_latency,
            clock_fn=self._clock,
            sleep_fn=self._sleep,
        )
        self.transport = TransportClient(
            backend=self.backend,
            timeout_seconds=transport_timeout,
            max_retries=transport_max_retries,
            clock_fn=self._clock,
            sleep_fn=self._sleep,
        )
        self.tool = ToolService(
            transport=self.transport,
            timeout_seconds=tool_timeout,
            max_retries=tool_max_retries,
            clock_fn=self._clock,
            sleep_fn=self._sleep,
        )
        self.agent = AgentWorkflowService(
            tool_service=self.tool,
            max_retries=planner_max_retries,
            timeout_seconds=planner_timeout,
            clock_fn=self._clock,
            sleep_fn=self._sleep,
        )

        # Workload runner state
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._events: deque[tuple[float, bool, float]] = deque()  # (finish_time, success, latency)
        self._submissions: deque[float] = deque()
        self._lock = threading.Lock()
        self._req_counter = 0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="agent-workload-generator", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def inject_latency_fault(self, fault_latency: float | None = None):
        if fault_latency is not None:
            self.backend.fault_latency = fault_latency
        self.backend.fault_active = True
        logger.info(f"[Fault] Injected backend latency spike ({self.backend.fault_latency}s)")

    def remove_latency_fault(self):
        self.backend.fault_active = False
        logger.info("[Fault] Removed backend latency perturbation; backend restored to normal service time")

    def apply_mitigation(
        self,
        cap_planner_retries: int | None = 1,
        disable_nested_retries: bool = True,
        enable_backoff: bool = True,
        shed_stale_queue: bool = True,
    ):
        """Apply mitigation policies to collapse the metastable retry storm."""
        if cap_planner_retries is not None:
            self.agent.max_retries = cap_planner_retries
        if disable_nested_retries:
            self.tool.max_retries = 1
            self.transport.max_retries = 1
        if enable_backoff:
            self.transport.backoff_seconds = 0.2
        if shed_stale_queue:
            self.transport.drop_stale = True
            with self.backend._lock:
                self.backend.waiting_queue.clear()
        logger.info(
            f"[Mitigation] Applied policy: planner_max={self.agent.max_retries}, "
            f"tool_max={self.tool.max_retries}, transport_max={self.transport.max_retries}, "
            f"shed_stale={shed_stale_queue}"
        )

    def _run_loop(self):
        interval = 1.0 / max(0.1, self.current_rate)
        next_submission = self._clock()

        while not self._stop.is_set():
            now = self._clock()
            if now < next_submission:
                self._sleep(min(0.02, next_submission - now))
                continue

            with self._lock:
                self._req_counter += 1
                req_id = f"req-{self._req_counter}"
                wf_id = f"wf-{self._req_counter}"
                self._submissions.append(now)

            threading.Thread(
                target=self._execute_request_task,
                args=(req_id, wf_id, now),
                daemon=True,
            ).start()

            next_submission += interval
            if next_submission < now - interval:
                next_submission = now + interval

    def _execute_request_task(self, req_id: str, wf_id: str, submitted_at: float):
        success = False
        try:
            success = self.agent.run_workflow(req_id, wf_id)
        except Exception as exc:
            logger.error(f"Workflow exception: {exc}")
            success = False
        finally:
            finished_at = self._clock()
            latency = finished_at - submitted_at
            with self._lock:
                self._events.append((finished_at, success, latency))

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            logical_reqs = self._req_counter
        backend_attempts = self.backend.total_backend_attempts
        amplification = backend_attempts / logical_reqs if logical_reqs > 0 else 1.0

        return {
            "agent_logical_requests_total": logical_reqs,
            "agent_backend_attempts_total": backend_attempts,
            "agent_retry_total_transport": self.transport.transport_retries_total,
            "agent_retry_total_tool": self.tool.tool_retries_total,
            "agent_retry_total_planner": self.agent.planner_retries_total,
            "agent_active_workflows": self.agent.active_workflows,
            "agent_workflow_queue_depth": self.agent.workflow_queue_depth,
            "backend_active_requests": self.backend.db_pool_active,
            "backend_waiting_requests": self.backend.db_pool_waiting,
            "db_pool_active": self.backend.db_pool_active,
            "db_pool_waiting": self.backend.db_pool_waiting,
            "amplification_ratio": amplification,
            "http_requests_total": backend_attempts,
        }

    def snapshot(self, window_seconds: float = 10.0) -> WorkloadSnapshot:
        now = self._clock()
        cutoff = now - window_seconds

        with self._lock:
            while self._submissions and self._submissions[0] < now - 300:
                self._submissions.popleft()
            while self._events and self._events[0][0] < now - 300:
                self._events.popleft()

            submitted = sum(1 for t in self._submissions if t >= cutoff)
            recent_events = [e for e in self._events if e[0] >= cutoff]

        completed = len(recent_events)
        succeeded = sum(1 for e in recent_events if e[1])
        failed = completed - succeeded
        latencies = sorted(e[2] for e in recent_events)
        p95 = latencies[int(0.95 * len(latencies))] if latencies else 0.0

        with self.backend._lock:
            while self.backend.backend_attempt_events and self.backend.backend_attempt_events[0] < now - 300:
                self.backend.backend_attempt_events.popleft()
            window_backend_attempts = sum(1 for t in self.backend.backend_attempt_events if t >= cutoff)

        metrics = self.get_metrics()
        actual_rate = submitted / max(0.001, window_seconds)
        success_rate = (succeeded / completed) if completed > 0 else 0.0

        # Windowed amplification ratio: backend attempts / logical requests in window
        if submitted > 0:
            amplification = window_backend_attempts / submitted
        else:
            amplification = metrics["amplification_ratio"]

        metrics["window_amplification_ratio"] = amplification

        return WorkloadSnapshot(
            submitted=submitted,
            completed=completed,
            succeeded=succeeded,
            failed=failed,
            actual_rate=actual_rate,
            success_rate=success_rate,
            p95_latency_seconds=p95,
            amplification_ratio=amplification,
            backend_queue_depth=self.backend.queue_depth,
            db_pool_waiting=self.backend.db_pool_waiting,
            backend_active_requests=self.backend.db_pool_active,
            metrics=metrics,
        )
