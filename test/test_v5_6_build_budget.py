from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from graphmem.build.budget import BuildTokenLedger
from graphmem.build.semantic import strict_scene_schema
from graphmem.config import GraphMemV5Config, ModelConfig, config_hash


# --- ledger -------------------------------------------------------------------

def test_an_unenforced_ledger_never_refuses() -> None:
    """ceiling=0 must reproduce the pre-V5.6 unbounded build exactly."""
    ledger = BuildTokenLedger("m", 0)

    for _ in range(100):
        allowed, degrade = ledger.reserve(1_000_000)
        assert allowed and not degrade
    assert ledger.snapshot()["within_budget"]


def test_spending_stops_at_the_ceiling() -> None:
    ledger = BuildTokenLedger("m", 1000)

    assert ledger.reserve(600)[0]
    ledger.settle(600, 600)
    assert ledger.reserve(600)[0] is False
    assert ledger.snapshot()["skipped_scenes"] == 1


def test_the_fact_cap_degrades_before_the_ceiling_is_hit() -> None:
    """A cliff at the ceiling would drop the tail of every long conversation."""
    ledger = BuildTokenLedger("m", 1000, degrade_at=0.5)

    allowed, degrade = ledger.reserve(400)
    assert allowed and not degrade  # 40% of the ceiling
    ledger.settle(400, 400)

    allowed, degrade = ledger.reserve(150)
    assert allowed and not degrade  # still 40% committed at reserve time
    ledger.settle(150, 150)

    allowed, degrade = ledger.reserve(150)
    assert allowed and degrade  # 55% committed, past the 50% threshold


def test_reservations_stop_concurrent_workers_overshooting_together() -> None:
    """Each worker fits alone; without reservations they collectively overrun."""
    ledger = BuildTokenLedger("m", 1000)
    granted = []

    def worker(_index: int) -> None:
        allowed, _ = ledger.reserve(300)
        if allowed:
            granted.append(300)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))

    assert sum(granted) <= 1000


def test_settling_releases_the_reservation_so_a_cheap_call_frees_room() -> None:
    ledger = BuildTokenLedger("m", 1000)
    ledger.reserve(800)
    ledger.settle(800, 100)  # the call was far cheaper than its upper bound

    assert ledger.reserve(800)[0]


def test_the_snapshot_reports_utilization_and_degradation() -> None:
    ledger = BuildTokenLedger("m", 1000)
    ledger.reserve(500); ledger.settle(500, 500)

    snapshot = ledger.snapshot()

    assert snapshot["spent"] == 500 and snapshot["utilization"] == 0.5
    assert snapshot["within_budget"] and snapshot["calls"] == 1


# --- schema -------------------------------------------------------------------

def test_the_quote_field_is_required_by_default() -> None:
    schema = strict_scene_schema(2, 4)
    fact = schema["properties"]["s"]["items"]["properties"]["f"]["items"]

    assert "q" in fact["required"] and "q" in fact["properties"]


def test_a_quote_free_schema_drops_the_field_entirely() -> None:
    """The quote is 26% of extraction output and the projection re-derives it."""
    schema = strict_scene_schema(2, 4, quote_evidence=False)
    fact = schema["properties"]["s"]["items"]["properties"]["f"]["items"]

    assert "q" not in fact["required"] and "q" not in fact["properties"]
    # The turn citation must survive: it is what grounds the fact.
    assert "r" in fact["required"]


def test_the_fact_cap_still_bounds_a_quote_free_schema() -> None:
    schema = strict_scene_schema(3, 5, quote_evidence=False)

    assert schema["properties"]["s"]["items"]["properties"]["f"]["maxItems"] == 5


# --- config -------------------------------------------------------------------

def test_budget_enforcement_is_off_by_default() -> None:
    """Turning it on must be a deliberate act, or frozen runs would change."""
    assert ModelConfig().semantic_max_tokens_per_memory == 0
    assert ModelConfig().semantic_quote_evidence is True


def test_a_negative_ceiling_is_rejected() -> None:
    with pytest.raises(ValueError, match="semantic_max_tokens_per_memory"):
        GraphMemV5Config(models=ModelConfig(semantic_max_tokens_per_memory=-1))


def test_a_degrade_threshold_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="semantic_budget_degrade_at"):
        GraphMemV5Config(models=ModelConfig(semantic_budget_degrade_at=1.5))


def test_enabling_the_budget_changes_the_config_hash() -> None:
    """A budgeted build is a different artifact and must not reuse a frozen one."""
    base = GraphMemV5Config()
    budgeted = GraphMemV5Config(models=ModelConfig(semantic_max_tokens_per_memory=220_000))

    assert config_hash(base) != config_hash(budgeted)


def test_a_call_costing_more_than_its_estimate_cannot_breach_the_ceiling() -> None:
    """Measured regression: a memory finished at 221,305 against a 220,000 cap.

    Reservations are estimates, and an estimate that comes in low is spent
    before anyone can take it back.  The margin has to absorb that.
    """
    from graphmem.build.semantic import ESTIMATE_SAFETY

    ledger = BuildTokenLedger("m", 1000, degrade_at=1.0)
    estimate = 500
    spent_per_call = int(estimate / ESTIMATE_SAFETY * 1.05)  # 5% worse than predicted

    while ledger.reserve(estimate)[0]:
        ledger.settle(estimate, spent_per_call)

    assert ledger.snapshot()["within_budget"], ledger.snapshot()


def test_the_estimate_carries_a_safety_margin() -> None:
    from graphmem.build.semantic import ESTIMATE_SAFETY

    assert ESTIMATE_SAFETY > 1.0
