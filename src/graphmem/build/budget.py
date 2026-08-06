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
    spent: int = 0
    reserved: int = 0
    degraded_calls: int = 0
    skipped_scenes: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def enforced(self) -> bool:
        return self.ceiling > 0

    @property
    def degrade_threshold(self) -> float:
        return self.ceiling * self.degrade_at

    def reserve(self, estimate: int) -> tuple[bool, bool]:
        """Ask to spend ``estimate`` tokens.

        Returns ``(allowed, degrade)``.  ``degrade`` asks the caller to request
        fewer facts for this call.  The estimate is held as a reservation so
        concurrent workers cannot each individually fit under the ceiling and
        collectively blow through it.
        """
        if not self.enforced:
            return True, False
        with self._lock:
            committed = self.spent + self.reserved
            if committed + estimate > self.ceiling:
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
            with self._lock:
                self.spent += actual
                self.calls += 1
            return
        with self._lock:
            self.reserved = max(0, self.reserved - estimate)
            self.spent += actual
            self.calls += 1

    def snapshot(self) -> Mapping[str, object]:
        with self._lock:
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
