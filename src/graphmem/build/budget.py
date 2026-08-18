"""Per-memory build token budget.

``ModelConfig.semantic_average_tokens_per_memory`` has declared 220,000 since
V5 and never had a consumer, so the measured cost of the frozen V5.4 build ran
at 275,261 tokens per LongMemEval memory (p95 486,110) with nothing to stop it.
This ledger makes the ceiling real.

The policy is a ladder, not a cliff, because simply refusing calls at the limit
would silently drop the last sessions of every large memory -- the graph would
lose exactly the late-conversation facts that multi-session questions need:

1. below ``degrade_at`` of the ceiling: extract normally;
2. above it: keep extracting but with a reduced fact cap, which cuts output
   (the larger half of extraction cost) while still covering every scene;
3. above the ceiling: stop calling and fall back to deterministic scene
   summaries, so later scenes stay present in the graph and merely lose their
   distilled facts.

Every degradation is counted and surfaced in the build manifest.  A run that
silently degraded would produce an unexplained accuracy drop.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class BuildTokenLedger:
    """Tracks and caps the LLM tokens one memory may spend during a build.

    Thread-safe: extraction fans out over a worker pool and every worker
    reserves against the same ledger.
    """

    memory_id: str
    ceiling: int
    degrade_at: float = 0.75
    #: Whether a call that costs more than it reserved degrades the calls after
    #: it.  With reservations based on an expected output rather than the output
    #: ceiling, individual overruns are normal and should not cascade; the
    #: running total still governs.
    fallback_on_overrun: bool = True
    spent: int = 0
    reserved: int = 0
    degraded_calls: int = 0
    skipped_scenes: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _condition: threading.Condition = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._condition = threading.Condition(self._lock)

    @property
    def enforced(self) -> bool:
        return self.ceiling > 0

    @property
    def degrade_threshold(self) -> float:
        return self.ceiling * self.degrade_at

    def reserve(self, estimate: int, *, wait_for_capacity: bool = False) -> tuple[bool, bool]:
        """Ask to spend ``estimate`` tokens.

        Returns ``(allowed, degrade)``.  ``degrade`` asks the caller to request
        fewer facts for this call.  The estimate is held as a reservation so
        concurrent workers cannot each individually fit under the ceiling and
        collectively blow through it.  In strict hard-reservation mode a call
        may wait when only *in-flight reservations* make it appear not to fit.
        Once those calls settle, the decision is made from their actual cost.
        This prevents worker scheduling from turning unused budget into a
        deterministic fallback while preserving the hard ceiling.
        """
        if not self.enforced:
            return True, False
        with self._condition:
            while (
                wait_for_capacity
                and self.reserved > 0
                and self.spent + estimate <= self.ceiling
                and self.spent + self.reserved + estimate > self.ceiling
            ):
                self._condition.wait()
            committed = self.spent + self.reserved
            if committed + estimate > self.ceiling:
                # Refusing the call is the only hard stop.  With
                # ``fallback_on_overrun`` off the ceiling still bounds the memory
                # -- what changes is that a single call costing more than it
                # reserved no longer forces the rest into fallback, which matters
                # once reservations are an expected value rather than the output
                # ceiling and overruns are routine.
                if self.fallback_on_overrun or committed >= self.ceiling:
                    self.skipped_scenes += 1
                    return False, False
            self.reserved += estimate
            degrade = committed >= self.degrade_threshold
            if degrade:
                self.degraded_calls += 1
            return True, degrade

    def settle(self, estimate: int, actual: int) -> None:
        """Release a reservation and record what the call actually cost."""
        if not self.enforced:
            with self._condition:
                self.spent += actual
                self.calls += 1
                self._condition.notify_all()
            return
        with self._condition:
            self.reserved = max(0, self.reserved - estimate)
            self.spent += actual
            self.calls += 1
            self._condition.notify_all()

    def cancel(self, estimate: int) -> None:
        """Release a failed call's reservation without recording token usage.

        Waiting workers must be woken even when the model request raises;
        otherwise a transient service restart can leave the memory build
        permanently blocked behind a reservation that will never settle.
        """
        if not self.enforced:
            return
        with self._condition:
            self.reserved = max(0, self.reserved - estimate)
            self._condition.notify_all()

    def snapshot(self) -> Mapping[str, object]:
        with self._condition:
            return {
                "memory_id": self.memory_id,
                "ceiling": self.ceiling,
                "enforced": self.enforced,
                "spent": self.spent,
                "calls": self.calls,
                "degraded_calls": self.degraded_calls,
                "skipped_scenes": self.skipped_scenes,
                "within_budget": (not self.enforced) or self.spent <= self.ceiling,
                "utilization": (self.spent / self.ceiling) if self.enforced else None,
            }
