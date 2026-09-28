"""Behavioral mitigation oracle for agent retry metastable overload."""

from __future__ import annotations

import logging
import time
from typing import Any

from sregym.conductor.oracles.base import Oracle
from sregym.conductor.oracles.failure import FailureClass

logger = logging.getLogger("all.conductor.oracle.agent_retry_metastable")


class AgentRetryMetastableMitigationOracle(Oracle):
    """Verify that mitigation has collapsed the metastable retry storm and restored stability."""

    importance = 1.0
    poll_interval_seconds = 2.0
    sample_seconds = 5.0
    recovery_timeout_seconds = 60.0

    FAILURE_CLASSES = {
        "traffic_did_not_recover": FailureClass.AGENT_ERROR,
        "amplification_too_high": FailureClass.AGENT_ERROR,
        "queue_depth_too_high": FailureClass.AGENT_ERROR,
        "db_pool_waiting_not_zero": FailureClass.AGENT_ERROR,
        "p95_latency_too_high": FailureClass.AGENT_ERROR,
        "metrics_unreadable": FailureClass.ENVIRONMENT_ERROR,
    }

    def __init__(
        self,
        problem,
        min_success_rate: float = 0.95,
        max_p95_latency: float = 1.0,
        max_queue_depth: int = 5,
        max_amplification: float = 1.5,
    ):
        super().__init__(problem)
        self.min_success_rate = min_success_rate
        self.max_p95_latency = max_p95_latency
        self.max_queue_depth = max_queue_depth
        self.max_amplification = max_amplification

    def _sample_health(self) -> dict[str, Any] | None:
        """Sample current health metrics from problem workload. Returns None if healthy, else failure dict."""
        try:
            snapshot = self.problem.workload.snapshot(window_seconds=self.sample_seconds)
        except Exception as exc:
            logger.error(f"[FAIL] Failed to read workload snapshot: {exc}")
            return self.fail("metrics_unreadable", error=str(exc))

        rate = snapshot.actual_rate
        success = snapshot.success_rate
        p95 = snapshot.p95_latency_seconds
        amp = snapshot.amplification_ratio
        queue_len = snapshot.backend_queue_depth
        db_waiting = snapshot.db_pool_waiting

        print(
            f"[Health Probe] success={success:.1%} p95={p95:.2f}s "
            f"amp={amp:.2f} queue={queue_len} db_waiting={db_waiting} rate={rate:.1f}req/s"
        )

        if snapshot.completed > 0 and success < self.min_success_rate:
            return self.fail(
                "traffic_did_not_recover",
                success_rate=round(success, 3),
                required=self.min_success_rate,
            )

        if p95 > self.max_p95_latency and snapshot.completed > 0:
            return self.fail(
                "p95_latency_too_high",
                p95_latency=round(p95, 3),
                maximum=self.max_p95_latency,
            )

        if queue_len > self.max_queue_depth:
            return self.fail(
                "queue_depth_too_high",
                queue_depth=queue_len,
                maximum=self.max_queue_depth,
            )

        if db_waiting > 0:
            return self.fail(
                "db_pool_waiting_not_zero",
                db_pool_waiting=db_waiting,
            )

        if amp > self.max_amplification:
            return self.fail(
                "amplification_too_high",
                amplification_ratio=round(amp, 2),
                maximum=self.max_amplification,
            )

        return None

    def evaluate(self, solution=None, trace=None, duration=None) -> dict[str, Any]:
        """Evaluate whether the metastable failure has collapsed and healthy service is restored."""
        print("== Agent Retry Metastable Mitigation Evaluation ==")
        deadline = time.monotonic() + self.recovery_timeout_seconds
        last_failure = None

        # Give mitigation a moment to take effect and verify sustained recovery
        consecutive_healthy = 0
        required_healthy_probes = 2

        while time.monotonic() < deadline:
            failure = self._sample_health()
            if failure is None:
                consecutive_healthy += 1
                if consecutive_healthy >= required_healthy_probes:
                    print(
                        f"[PASS] Metastable overload collapsed: service healthy for "
                        f"{consecutive_healthy} consecutive probes."
                    )
                    return {"success": True}
            else:
                consecutive_healthy = 0
                last_failure = failure

            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(self.poll_interval_seconds, remaining))

        print(f"[FAIL] Service remained in degraded/metastable state: {last_failure}")
        return last_failure or self.fail("traffic_did_not_recover")
