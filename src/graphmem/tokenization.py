from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from .text import estimate_tokens as heuristic_tokens


TOKENIZER_ENV = "GRAPHMEM_TOKENIZER_PATH"
_CACHE: dict[tuple[str, str], "TokenCounter"] = {}


class TokenCounter(Protocol):
    """Counts backbone tokens for a piece of evidence or a prompt."""

    backend: str
    model_id: str
    exact: bool

    def count(self, text: str) -> int: ...
    def count_many(self, texts: Sequence[str]) -> list[int]: ...
    def describe(self) -> dict[str, str | bool]: ...


@dataclass(slots=True)
class HeuristicTokenCounter:
    """The V5.4/V5.5 word-count estimate, kept only as an explicit fallback.

    It runs about 25% high on conversational English, so any budget enforced
    with it is neither tight nor safe.  A run that falls back to this must say
    so in its manifest rather than presenting the number as a token count.
    """

    model_id: str = "heuristic"
    backend: str = "heuristic_words_x1.3"
    exact: bool = False

    def count(self, text: str) -> int:
        return heuristic_tokens(text)

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [heuristic_tokens(text) for text in texts]

    def describe(self) -> dict[str, str | bool]:
        return {"backend": self.backend, "model_id": self.model_id, "exact": self.exact}


class ExactTokenCounter:
    """Counts with the backbone's own vocabulary, loaded from a local file.

    Uses ``tokenizers`` directly rather than ``transformers`` so counting never
    needs the model weights, a network round trip, or a GPU.
    """

    backend = "tokenizers.Tokenizer"
    exact = True

    def __init__(self, model_id: str, tokenizer_path: Path) -> None:
        from tokenizers import Tokenizer  # imported lazily: optional dependency

        self.model_id = model_id
        self.tokenizer_path = tokenizer_path
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()[:16]
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if not text:
            return 0
        cached = self._cache.get(text)
        if cached is None:
            cached = len(self._tokenizer.encode(text, add_special_tokens=False).ids)
            # Turn text repeats across candidates, proof units and the answer
            # prompt; bound the cache so a long corpus cannot grow it forever.
            if len(self._cache) < 200_000:
                self._cache[text] = cached
        return cached

    def count_many(self, texts: Sequence[str]) -> list[int]:
        pending = [text for text in dict.fromkeys(texts) if text and text not in self._cache]
        if pending:
            for text, encoded in zip(pending, self._tokenizer.encode_batch(
                    list(pending), add_special_tokens=False)):
                if len(self._cache) < 200_000:
                    self._cache[text] = len(encoded.ids)
        return [self.count(text) for text in texts]

    def describe(self) -> dict[str, str | bool]:
        return {
            "backend": self.backend, "model_id": self.model_id, "exact": True,
            "tokenizer_path": str(self.tokenizer_path), "tokenizer_sha256_prefix": self._digest,
        }


def _hub_directories() -> list[Path]:
    roots: list[Path] = []
    for value in (os.environ.get("HF_HOME"), os.environ.get("HUGGINGFACE_HUB_CACHE"),
                  "~/.cache/huggingface"):
        if not value:
            continue
        root = Path(value).expanduser()
        roots.append(root if root.name == "hub" else root / "hub")
    return [row for row in dict.fromkeys(roots) if row.is_dir()]


def find_tokenizer_file(model_id: str) -> Path | None:
    """Locate ``tokenizer.json`` for a model id without contacting the hub.

    Snapshot directories are sorted so that a cache holding several revisions
    resolves to the same one on every run.
    """
    override = os.environ.get(TOKENIZER_ENV)
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_dir():
            candidate = candidate / "tokenizer.json"
        return candidate if candidate.is_file() else None
    slug = "models--" + re.sub(r"[^0-9A-Za-z._-]", "--", model_id)
    for hub in _hub_directories():
        snapshots = hub / slug / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in sorted(snapshots.iterdir(), reverse=True):
            candidate = snapshot / "tokenizer.json"
            if candidate.is_file():
                return candidate
    return None


def resolve_token_counter(model_id: str, *, require_exact: bool = False) -> TokenCounter:
    """Return a counter for ``model_id``, preferring the model's real vocabulary.

    ``require_exact`` turns a missing tokenizer into an error.  Gate enforcement
    and any reported token figure must set it; exploratory code need not.
    """
    key = (model_id, os.environ.get(TOKENIZER_ENV, ""))
    cached = _CACHE.get(key)
    if cached is not None:
        if require_exact and not cached.exact:
            raise RuntimeError(f"no exact tokenizer available for {model_id!r}")
        return cached
    path = find_tokenizer_file(model_id)
    counter: TokenCounter
    if path is None:
        if require_exact:
            raise RuntimeError(
                f"no local tokenizer.json for {model_id!r}; set {TOKENIZER_ENV} or populate the "
                "Hugging Face cache before enforcing a token budget")
        counter = HeuristicTokenCounter()
    else:
        try:
            counter = ExactTokenCounter(model_id, path)
        except ImportError as error:
            if require_exact:
                raise RuntimeError(
                    "the 'tokenizers' package is required for exact token counting") from error
            counter = HeuristicTokenCounter()
    _CACHE[key] = counter
    return counter


def total_tokens(counter: TokenCounter, texts: Iterable[str]) -> int:
    rows = list(texts)
    return sum(counter.count_many(rows)) if rows else 0
