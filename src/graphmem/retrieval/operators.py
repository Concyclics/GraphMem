"""Operator AST for the V5.6 query compiler.

V5.5 compiled a question to a single ``QueryOperator`` enum by racing string
tests in a fixed order, so a question like "How many places have Alice and Bob
both visited?" collapsed to ``COUNT_DISTINCT`` and lost the intersection
entirely.  An AST keeps the composition, and every node names exactly the
operands it consumes so the scheduler and packer can reason about coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, TypeAlias

from ..domain import QueryOperator


@dataclass(frozen=True, slots=True)
class FactSet:
    """The leaf: every fact bound to one operand."""

    operand_id: str


@dataclass(frozen=True, slots=True)
class Lookup:
    child: "OperatorNode"


@dataclass(frozen=True, slots=True)
class UnionDistinct:
    children: tuple["OperatorNode", ...]
    distinct_by: str = "value"


@dataclass(frozen=True, slots=True)
class IntersectionDistinct:
    children: tuple["OperatorNode", ...]
    distinct_by: str = "value"


@dataclass(frozen=True, slots=True)
class GroupByOwner:
    children: tuple["OperatorNode", ...]
    distinct_by: str = "value"


@dataclass(frozen=True, slots=True)
class CountDistinct:
    child: "OperatorNode"
    distinct_by: str = "value"


@dataclass(frozen=True, slots=True)
class ExistsAll:
    children: tuple["OperatorNode", ...]


@dataclass(frozen=True, slots=True)
class Ordinal:
    """The k-th element under ``order``.

    ``index`` is 1-based; ``-1`` means "the last one".  V5.5 parsed the ordinal
    word but never carried k, so "second" and "first" resolved identically.
    """

    child: "OperatorNode"
    index: int = 1
    order: str = "ascending"

    def __post_init__(self) -> None:
        if self.index == 0:
            raise ValueError("ordinal index is 1-based; 0 is not a position")
        if self.order not in {"ascending", "descending"}:
            raise ValueError(f"invalid ordinal order: {self.order!r}")


@dataclass(frozen=True, slots=True)
class ArgMinTime:
    child: "OperatorNode"


@dataclass(frozen=True, slots=True)
class ArgMaxTime:
    child: "OperatorNode"


@dataclass(frozen=True, slots=True)
class DateDifference:
    left: "OperatorNode"
    right: "OperatorNode"
    unit: str = "days"


@dataclass(frozen=True, slots=True)
class LatestState:
    child: "OperatorNode"


OperatorNode: TypeAlias = (
    FactSet | Lookup | UnionDistinct | IntersectionDistinct | GroupByOwner | CountDistinct
    | ExistsAll | Ordinal | ArgMinTime | ArgMaxTime | DateDifference | LatestState
)

_ROOT_OPERATOR: dict[type, QueryOperator] = {
    FactSet: QueryOperator.LOOKUP,
    Lookup: QueryOperator.LOOKUP,
    UnionDistinct: QueryOperator.UNION_DISTINCT,
    IntersectionDistinct: QueryOperator.INTERSECTION_DISTINCT,
    GroupByOwner: QueryOperator.GROUP_BY_OWNER,
    CountDistinct: QueryOperator.COUNT_DISTINCT,
    ExistsAll: QueryOperator.EXISTS_ALL,
    Ordinal: QueryOperator.ORDINAL,
    ArgMinTime: QueryOperator.ARGMIN_TIME,
    ArgMaxTime: QueryOperator.ARGMAX_TIME,
    DateDifference: QueryOperator.DATE_DIFFERENCE,
    LatestState: QueryOperator.LATEST_STATE,
}

# What the answer composer may emit without an LLM once the certificate closes.
CLOSED_FORM_KINDS = frozenset({
    "count", "list", "group", "existence", "date_difference", "state", "ordinal",
})

_ANSWER_KIND: dict[type, str] = {
    FactSet: "lookup",
    Lookup: "lookup",
    UnionDistinct: "list",
    IntersectionDistinct: "list",
    GroupByOwner: "group",
    CountDistinct: "count",
    ExistsAll: "existence",
    Ordinal: "ordinal",
    ArgMinTime: "ordinal",
    ArgMaxTime: "ordinal",
    DateDifference: "date_difference",
    LatestState: "state",
}


def children_of(node: OperatorNode) -> tuple[OperatorNode, ...]:
    if isinstance(node, FactSet):
        return ()
    if isinstance(node, DateDifference):
        return (node.left, node.right)
    if isinstance(node, (Lookup, CountDistinct, Ordinal, ArgMinTime, ArgMaxTime, LatestState)):
        return (node.child,)
    return tuple(node.children)


def walk(node: OperatorNode) -> Iterator[OperatorNode]:
    """Pre-order traversal, parents before children, deterministic."""
    yield node
    for child in children_of(node):
        yield from walk(child)


def operand_ids(node: OperatorNode) -> tuple[str, ...]:
    """Operands the node consumes, in first-appearance order."""
    return tuple(dict.fromkeys(row.operand_id for row in walk(node) if isinstance(row, FactSet)))


def root_operator(node: OperatorNode) -> QueryOperator:
    """The single enum V5.5 used, kept for traces and back-compatible metrics."""
    return _ROOT_OPERATOR[type(node)]


def answer_kind(node: OperatorNode) -> str:
    return _ANSWER_KIND[type(node)]


def is_closed_form(node: OperatorNode) -> bool:
    """Whether the algebra alone determines the answer value."""
    return answer_kind(node) in CLOSED_FORM_KINDS


def distinct_by(node: OperatorNode, default: str = "value") -> str:
    for row in walk(node):
        value = getattr(row, "distinct_by", None)
        if value:
            return str(value)
    return default


def requires_exhaustive_scope(node: OperatorNode) -> bool:
    """Operators whose answer is wrong unless the collection is fully enumerated."""
    return any(isinstance(row, (UnionDistinct, IntersectionDistinct, GroupByOwner,
                                CountDistinct, Ordinal))
               for row in walk(node))


def describe(node: OperatorNode) -> str:
    """Compact, stable rendering for traces and test assertions."""
    if isinstance(node, FactSet):
        return f"FactSet({node.operand_id})"
    if isinstance(node, Ordinal):
        return f"Ordinal[{node.index},{node.order}]({describe(node.child)})"
    if isinstance(node, DateDifference):
        return f"DateDifference[{node.unit}]({describe(node.left)},{describe(node.right)})"
    inner = ",".join(describe(child) for child in children_of(node))
    return f"{type(node).__name__}({inner})"


def flatten_operands(nodes: Sequence[OperatorNode]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for node in nodes for item in operand_ids(node)))
