"""Tests for AgentRetryMetastableMitigationOracle."""

from types import SimpleNamespace

from sregym.conductor.oracles.agent_retry_metastable_mitigation import AgentRetryMetastableMitigationOracle
from sregym.generators.workload.agentic_workflow import WorkloadSnapshot


def test_mitigation_oracle_passes_when_healthy():
    problem = SimpleNamespace()
    oracle = AgentRetryMetastableMitigationOracle(
        problem=problem,
        min_success_rate=0.95,
        max_p95_latency=1.0,
        max_queue_depth=5,
        max_amplification=1.5,
    )
    oracle.recovery_timeout_seconds = 2.0
    oracle.poll_interval_seconds = 0.1
    oracle.sample_seconds = 0.1

    # Healthy snapshot
    healthy_snapshot = WorkloadSnapshot(
        submitted=20,
        completed=20,
        succeeded=20,
        failed=0,
        actual_rate=10.0,
        success_rate=1.0,
        p95_latency_seconds=0.25,
        amplification_ratio=1.05,
        backend_queue_depth=0,
        db_pool_waiting=0,
        backend_active_requests=2,
    )
    problem.workload = SimpleNamespace(snapshot=lambda window_seconds: healthy_snapshot)

    result = oracle.evaluate()
    assert result.get("success") is True


def test_mitigation_oracle_fails_when_traffic_degraded():
    problem = SimpleNamespace()
    oracle = AgentRetryMetastableMitigationOracle(
        problem=problem,
        min_success_rate=0.95,
    )
    oracle.recovery_timeout_seconds = 0.5
    oracle.poll_interval_seconds = 0.1
    oracle.sample_seconds = 0.1

    # Degraded snapshot (success rate low)
    degraded_snapshot = WorkloadSnapshot(
        submitted=20,
        completed=20,
        succeeded=10,
        failed=10,
        actual_rate=10.0,
        success_rate=0.50,
        p95_latency_seconds=0.25,
        amplification_ratio=1.0,
        backend_queue_depth=0,
        db_pool_waiting=0,
        backend_active_requests=2,
    )
    problem.workload = SimpleNamespace(snapshot=lambda window_seconds: degraded_snapshot)

    result = oracle.evaluate()
    assert result.get("success") is False
    assert result.get("reason") == "traffic_did_not_recover"


def test_mitigation_oracle_fails_when_amplification_too_high():
    problem = SimpleNamespace()
    oracle = AgentRetryMetastableMitigationOracle(
        problem=problem,
        max_amplification=1.5,
    )
    oracle.recovery_timeout_seconds = 0.5
    oracle.poll_interval_seconds = 0.1
    oracle.sample_seconds = 0.1

    # Retry storm snapshot (amplification = 4.5)
    storm_snapshot = WorkloadSnapshot(
        submitted=20,
        completed=20,
        succeeded=20,
        failed=0,
        actual_rate=10.0,
        success_rate=1.0,
        p95_latency_seconds=0.3,
        amplification_ratio=4.5,
        backend_queue_depth=0,
        db_pool_waiting=0,
        backend_active_requests=10,
    )
    problem.workload = SimpleNamespace(snapshot=lambda window_seconds: storm_snapshot)

    result = oracle.evaluate()
    assert result.get("success") is False
    assert result.get("reason") == "amplification_too_high"


def test_mitigation_oracle_fails_when_db_pool_waiting():
    problem = SimpleNamespace()
    oracle = AgentRetryMetastableMitigationOracle(
        problem=problem,
    )
    oracle.recovery_timeout_seconds = 0.5
    oracle.poll_interval_seconds = 0.1
    oracle.sample_seconds = 0.1

    # Saturated concurrency queue
    queued_snapshot = WorkloadSnapshot(
        submitted=20,
        completed=20,
        succeeded=20,
        failed=0,
        actual_rate=10.0,
        success_rate=1.0,
        p95_latency_seconds=0.3,
        amplification_ratio=1.1,
        backend_queue_depth=15,
        db_pool_waiting=15,
        backend_active_requests=25,
    )
    problem.workload = SimpleNamespace(snapshot=lambda window_seconds: queued_snapshot)

    result = oracle.evaluate()
    assert result.get("success") is False
    assert result.get("reason") == "queue_depth_too_high"
