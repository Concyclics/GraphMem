import re, collections
from .calendar_source_resolver import source_date

TEMPORAL = re.compile(
    r"\b(when|before|after|ago|since|date|dates|years?|months?|weeks?|days?|hours?|duration|order|earliest|latest|first|last|how long)\b",
    re.I,
)

ORDER = re.compile(r"\b(order|earliest|latest|first|before|after)\b", re.I)

STOP = set(
    "when did do does was were is are have has had the a an in on at to of for and with from how long many what which before after first last order earliest latest days day weeks week months month years year ago since i my me".split()
)


def terms(text):
    return set(re.findall(r"[a-z]{3,}", text.lower())) - STOP


def display_session(turn):
    return (
        int(turn["display_session"])
        if "display_session" in turn
        else int(turn["session_id"].split("_")[-1])
    )


def render(turns, annotations=None, repeat_dates=False):
    groups = collections.defaultdict(list)
    for turn in turns:
        groups[display_session(turn)].append(turn)
    output = []
    for sid in sorted(groups):
        ts = sorted(groups[sid], key=lambda t: t["turn_index"])
        assert len({t["timestamp"] for t in ts}) == 1
        output.append(f"Session {sid} — {ts[0]['timestamp']}\n")
        for turn in ts:
            label = str(turn["turn_index"] + 1)
            if repeat_dates:
                label += " @ " + str(source_date(turn["timestamp"]))
            output.append(f"[{label}] {turn['speaker']}: {turn['raw_text']}\n")
            for annotation in (annotations or {}).get(turn["turn_id"], []):
                output.append(annotation + "\n")
    return "".join(output)
