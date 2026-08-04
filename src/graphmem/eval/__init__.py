"""Offline-only evaluation assets.

Nothing in this package may be imported by ``graphmem.build``, retrieval, or
runtime modules. Gold annotations are evaluation inputs, never online features.
"""

from .gold_turns import GoldTurnAnnotation, GoldTurnSet, load_gold_turns

__all__ = ["GoldTurnAnnotation", "GoldTurnSet", "load_gold_turns"]
