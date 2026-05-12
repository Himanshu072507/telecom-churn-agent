"""Rule-based gating: decides which agents run for which bucket."""
from enum import Enum

from schemas import Bucket


class AgentName(str, Enum):
    ANALYST = "ANALYST"
    VOICE = "VOICE"
    EXECUTOR = "EXECUTOR"


def gate(bucket: Bucket) -> set[AgentName]:
    """Return the set of follow-up agents to run for a given bucket."""
    if bucket in (Bucket.SAFE, Bucket.WATCH):
        return set()
    return {AgentName.VOICE, AgentName.EXECUTOR}


def should_escalate(bucket: Bucket) -> bool:
    """Critical-bucket customers get the escalation banner."""
    return bucket == Bucket.CRITICAL
