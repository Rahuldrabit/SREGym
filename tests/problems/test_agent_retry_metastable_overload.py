"""Tests for agent_retry_metastable_overload problem, architecture, and dynamics."""

import time

import pytest

from sregym.conductor.problems.agent_retry_metastable_overload import AgentRetryMetastableOverload
from sregym.generators.workload.agentic_workflow import (
    AgenticWorkflowWorkload,
    BackendService,
    RequestContext,
    ToolService,
    ToolTimeoutError,
    TransportClient,
    TransportTimeoutError,
)


def test_problem_metadata_and_attributes():
    problem = AgentRetryMetastableOverload()
    assert problem.namespace in ("agentic-retry-platform", "agentic-rag-platform")
    assert problem.run_default_workload is False
    assert problem.base_rate == 10.0
    assert problem.concurrency_limit == 25
    assert problem.fault_latency in (1.5, 1.50)
    assert any(s in problem.faulty_service for s in ("agent-orchestrator", "agent-workflow"))
    assert any(s in problem.faulty_service for s in ("backend", "data-api"))
    assert "retry-policy" in problem.root_cause


def test_request_telemetry_context_structure():
    ctx = RequestContext(
        logical_request_id="req-219",
        workflow_id="wf-882",
        attempt=1,
        retry_layer="initial",
        parent_attempt_id=None,
    )
    assert ctx.logical_request_id == "req-219"
    assert ctx.workflow_id == "wf-882"
    assert ctx.attempt == 1
    assert ctx.retry_layer == "initial"


def test_healthy_state_has_low_amplification():
    """Verify that in healthy state, attempts per request A(t) ~ 1.0 and success is high."""
    workload = AgenticWorkflowWorkload(
        base_rate=15.0,
        concurrency_limit=25,
        normal_latency=0.02,
        fault_latency=0.2,
        transport_timeout=0.08,
        tool_timeout=0.2,
        planner_timeout=0.5,
    )
    workload.start()
    try:
        time.sleep(0.4)
        snapshot = workload.snapshot(window_seconds=0.3)
        assert snapshot.completed > 0
        assert snapshot.success_rate >= 0.90
        assert snapshot.amplification_ratio <= 1.3
        assert snapshot.backend_queue_depth <= 2
    finally:
        workload.stop()


def test_multi_layer_retry_amplification_math():
    """Verify that a backend timeout triggers compounded retries: R_transport x R_tool x R_planner."""
    backend = BackendService(concurrency_limit=5, normal_latency=0.05, fault_latency=0.5)
    backend.fault_active = True  # induce timeout

    transport = TransportClient(backend=backend, timeout_seconds=0.02, max_retries=2)
    tool = ToolService(transport=transport, timeout_seconds=0.06, max_retries=2)

    ctx = RequestContext(
        logical_request_id="req-test",
        workflow_id="wf-test",
        attempt=1,
        retry_layer="initial",
    )

    with pytest.raises((ToolTimeoutError, TransportTimeoutError)):
        tool.call_tool(ctx)

    # 1 tool attempt causes 2 transport attempts; tool retries once = 2 tool attempts x 2 transport = 4 backend attempts
    assert backend.total_backend_attempts == 4
    assert transport.transport_retries_total == 2
    assert tool.tool_retries_total == 1


def test_metastable_sustaining_loop_reproduction():
    """Core SREGym property:
    A transient backend latency spike triggers multi-layer retries.
    When the fault is removed, the system remains degraded because queued and retried
    requests continue to saturate finite backend concurrency.
    """
    workload = AgenticWorkflowWorkload(
        base_rate=20.0,
        concurrency_limit=6,
        normal_latency=0.08,
        fault_latency=0.35,
        transport_timeout=0.10,
        tool_timeout=0.25,
        planner_timeout=0.55,
        transport_max_retries=2,
        tool_max_retries=2,
        planner_max_retries=3,
    )
    workload.start()
    try:
        # 1. Healthy baseline
        time.sleep(0.3)
        baseline = workload.snapshot(0.2)
        assert baseline.success_rate >= 0.85
        assert baseline.amplification_ratio <= 1.3

        # 2. Inject temporary fault
        workload.inject_latency_fault()
        time.sleep(0.5)

        # 3. Remove fault
        workload.remove_latency_fault()
        time.sleep(0.4)

        # 4. Check that system REMAINS degraded (metastable state)
        post_fault = workload.snapshot(0.3)
        assert post_fault.amplification_ratio > 1.5
        assert post_fault.success_rate < 0.60
        assert post_fault.backend_queue_depth > 5
    finally:
        workload.stop()


def test_negative_control_self_recovers():
    """Negative control:
    Run the exact same temporary perturbation with unified retry budget (R_planner=1, R_tool=1).
    After the fault is removed, the system self-recovers, proving uncoordinated stacked retries are causal.
    """
    workload = AgenticWorkflowWorkload(
        base_rate=20.0,
        concurrency_limit=6,
        normal_latency=0.08,
        fault_latency=0.35,
        transport_timeout=0.10,
        tool_timeout=0.25,
        planner_timeout=0.55,
        transport_max_retries=2,
        tool_max_retries=1,  # Capped tool retries
        planner_max_retries=1,  # Capped planner retries
    )
    workload.start()
    try:
        time.sleep(0.3)
        workload.inject_latency_fault()
        time.sleep(0.4)
        workload.remove_latency_fault()

        # Wait for queue to drain
        time.sleep(1.0)
        post_fault = workload.snapshot(0.3)

        # System self-recovers in negative control
        assert post_fault.amplification_ratio <= 1.5
        assert post_fault.success_rate >= 0.80
        assert post_fault.backend_queue_depth <= 5
    finally:
        workload.stop()


def test_mitigation_collapses_metastable_overload():
    """Verify that applying mitigation collapses the sustaining retry overload and restores stability."""
    workload = AgenticWorkflowWorkload(
        base_rate=20.0,
        concurrency_limit=6,
        normal_latency=0.08,
        fault_latency=0.35,
        transport_timeout=0.10,
        tool_timeout=0.25,
        planner_timeout=0.55,
        transport_max_retries=2,
        tool_max_retries=2,
        planner_max_retries=3,
    )
    workload.start()
    try:
        # Enter metastable overload
        workload.inject_latency_fault()
        time.sleep(0.4)
        workload.remove_latency_fault()
        time.sleep(0.3)

        # Apply mitigation
        workload.apply_mitigation(
            cap_planner_retries=1,
            disable_nested_retries=True,
            enable_backoff=True,
            shed_stale_queue=True,
        )
        time.sleep(0.5)

        # Settle and verify recovery
        snapshot = workload.snapshot(0.3)
        assert snapshot.success_rate >= 0.85
        assert snapshot.amplification_ratio <= 1.5
        assert snapshot.backend_queue_depth <= 5
    finally:
        workload.stop()
