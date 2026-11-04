import re, calendar
from datetime import date, timedelta

NUMBERS = dict(
    zip(
        "one two three four five six seven eight nine ten eleven twelve".split(),
        range(1, 13),
    )
)

NUMBERS.update(a=1, an=1)
PATTERN = re.compile(
    r"\b(yesterday|today|tomorrow|(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an) (?:days?|weeks?|months?|years?) ago|last (?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.I,
)

STOP = set(
    "when did do does was were is are have has had the a an in on at to of for and with from how long what which before after".split()
)


def terms(s):
    return set(re.findall(r"[a-z]{3,}", s.lower())) - STOP


def month_shift(d, n):
    y, m = divmod(d.year * 12 + d.month - 1 + n, 12)
    return date(y, m + 1, min(d.day, calendar.monthrange(y, m + 1)[1]))


def resolve(raw, d):
    s = raw.lower()
    if s in ["yesterday", "today", "tomorrow"]:
        z = d + timedelta(days={"yesterday": -1, "today": 0, "tomorrow": 1}[s])
        return z, z
    if s == "last week":
        end = d - timedelta(days=d.weekday() + 1)
        return end - timedelta(days=6), end
    if s == "last month":
        end = d.replace(day=1) - timedelta(days=1)
        return end.replace(day=1), end
    if s == "last year":
        return date(d.year - 1, 1, 1), date(d.year - 1, 12, 31)
    if s.startswith("last "):
        weekday = list(calendar.day_name).index(s[5:].capitalize())
        z = d - timedelta(days=(d.weekday() - weekday) % 7 or 7)
        return z, z
    n, unit, _ = s.split()
    n = int(n) if n.isdigit() else NUMBERS[n]
    if unit.startswith("day"):
        z = d - timedelta(days=n)
    elif unit.startswith("week"):
        z = d - timedelta(weeks=n)
    elif unit.startswith("month"):
        z = month_shift(d, -n)
    else:
        z = month_shift(d, -12 * n)
    return z, z
