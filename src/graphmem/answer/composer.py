"""Closed-form answers derived from the algebra, with no LLM call.

A count, a list, a group, a date difference or a latest-state answer is fully
determined by ``AlgebraResult`` once the certificate closes.  Composing it
directly costs zero tokens and, more importantly, cannot hallucinate a member
the algebra never produced.

The draft is still labelled a *proposal* when it reaches the prompt: until the
post-pack certificate is trustworthy, an unverified closed form must not
outrank the cited turns.  ``AnswerDraft.certified`` records which of the two
regimes produced it.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..domain import AlgebraResult, EvidenceCertificate


@dataclass(frozen=True, slots=True)
class AnswerDraft:
    answer_kind: str
    text: str
    #: True only when the algebra closed its scope *and* the certificate agrees.
    #: A count over a scope known to be partial is a lower bound, not an answer.
    certified: bool
    member_keys: tuple[str, ...] = ()
    witness_binding_ids: tuple[str, ...] = ()
    degradations: tuple[str, ...] = ()


def _members_text(result: AlgebraResult) -> str:
    values = [member.value for member in result.members if member.value]
    # dict.fromkeys keeps first-seen order, which the algebra already fixed.
    return ", ".join(dict.fromkeys(values))


def compose(result: AlgebraResult | None,
            certificate: EvidenceCertificate | None = None) -> AnswerDraft | None:
    """Return a closed-form draft, or ``None`` when the operator has no such form."""
    if result is None or result.answer_kind not in {
            "count", "list", "group", "existence", "date_difference", "state", "ordinal"}:
        return None
    kind = result.answer_kind
    certified = bool(result.scope_complete and not result.degradations
                     and (certificate is None or certificate.post_pack_complete))
    witnesses = tuple(dict.fromkeys(
        binding_id for member in result.members for binding_id in member.witness_binding_ids))
    keys = tuple(member.member_key for member in result.members)

    if kind == "count":
        # ``count`` is authoritative only when the scope is closed; otherwise the
        # distinct members found are a floor and saying so is the honest form.
        total = result.count if result.count is not None else len(result.members)
        text = str(total) if certified else f"at least {total}"
    elif kind == "list":
        text = _members_text(result)
    elif kind == "group":
        rows = [f"{owner}: {', '.join(values)}"
                for owner, values in sorted(result.groups.items())]
        text = "; ".join(rows)
    elif kind == "existence":
        # An unclosed scope cannot prove absence, so only a positive witness set
        # may answer here.
        if not result.members and not certified:
            return None
        text = "yes" if result.members else "no"
    elif kind == "date_difference":
        endpoints = sorted(result.temporal_endpoints, key=lambda row: row.key.sort_key)
        if len(endpoints) < 2:
            return None
        days = endpoints[-1].key.days_between(endpoints[0].key)
        if days is None:
            return None
        text = f"{days} days"
        witnesses = tuple(row.binding_id for row in endpoints)
    elif kind == "state":
        state = result.state_result
        if state is None or not state.current_value:
            return None
        text = state.current_value
        witnesses = (state.current_binding_id,)
    else:  # ordinal
        if not result.members:
            return None
        text = result.members[0].value
        witnesses = tuple(result.members[0].witness_binding_ids)

    if not text:
        return None
    return AnswerDraft(answer_kind=kind, text=text, certified=certified,
                       member_keys=keys, witness_binding_ids=witnesses,
                       degradations=tuple(result.degradations))
