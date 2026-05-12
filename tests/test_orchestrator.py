from orchestrator import gate, should_escalate, AgentName
from schemas import Bucket


def test_safe_bucket_runs_no_agents():
    assert gate(Bucket.SAFE) == set()


def test_watch_bucket_runs_no_agents():
    assert gate(Bucket.WATCH) == set()


def test_at_risk_bucket_runs_voice_and_executor():
    assert gate(Bucket.AT_RISK) == {AgentName.VOICE, AgentName.EXECUTOR}


def test_critical_bucket_runs_voice_and_executor():
    assert gate(Bucket.CRITICAL) == {AgentName.VOICE, AgentName.EXECUTOR}


def test_only_critical_triggers_escalation():
    assert should_escalate(Bucket.CRITICAL) is True
    assert should_escalate(Bucket.AT_RISK) is False
    assert should_escalate(Bucket.WATCH) is False
    assert should_escalate(Bucket.SAFE) is False
