"""Offline-only evaluation assets.

Nothing in this package may be imported by ``graphmem.build``, retrieval, or
runtime modules. Gold annotations are evaluation inputs, never online features.
"""

from .gold_turns import GoldTurnAnnotation, GoldTurnSet, load_gold_turns
from .devset import DevQuestion, GoldTurnRef, calibration40, ingest_questions, load_dev_questions
from .metrics import aggregate_metrics, navigation_metrics

__all__ = [
    "DevQuestion", "GoldTurnAnnotation", "GoldTurnRef", "GoldTurnSet",
    "aggregate_metrics", "calibration40", "ingest_questions", "load_dev_questions",
    "load_gold_turns", "navigation_metrics",
]
