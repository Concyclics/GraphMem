"""Pure calendar resolver extracted unchanged from the prior inline-time reader.
Source SHA256: 619c2905dca8b81c7c4a01d09e70c2216a5cbf114667867d69e52bb13f2245a9
"""

from datetime import date, datetime, timedelta
import calendar
import re

NUMBERS = dict(
    zip(
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
        range(1, 21),
    )
)
NUMBERS.update(a=1, an=1)
N = "(?:\\d+|" + "|".join(NUMBERS) + ")"
UNIT = "(?:days?|weeks?|months?|years?)"
WEEKDAYS = "|".join((day.lower() for day in calendar.day_name))
RELATIVE = re.compile(
    "\\b(?:the day before yesterday|the day after tomorrow|yesterday|today|tomorrow|(?:last|next) (?:week|month|year|"
    + WEEKDAYS
    + ")|this (?:week|month|year)|"
    + N
    + " "
    + UNIT
    + " ago|in "
    + N
    + " "
    + UNIT
    + ")\\b",
    re.I,
)


def source_date(text):
    match = re.match("(\\d{4})[/-](\\d{2})[/-](\\d{2})", text)
    if match:
        return date(*map(int, match.groups()))
    return datetime.strptime(text.split(" on ")[-1], "%d %B, %Y").date()


def shift_months(day, amount):
    year, month = divmod(day.year * 12 + day.month - 1 + amount, 12)
    return date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


def resolve(phrase, anchor):
    phrase = phrase.lower()
    offsets = {
        "the day before yesterday": -2,
        "yesterday": -1,
        "today": 0,
        "tomorrow": 1,
        "the day after tomorrow": 2,
    }
    if phrase in offsets:
        value = anchor + timedelta(days=offsets[phrase])
        return (value, value)
    parts = phrase.split()
    if parts[0] in ["last", "this", "next"]:
        direction = {"last": -1, "this": 0, "next": 1}[parts[0]]
        unit = parts[1]
        if unit == "week":
            start = (
                anchor - timedelta(days=anchor.weekday()) + timedelta(weeks=direction)
            )
            return (start, start + timedelta(days=6))
        if unit == "month":
            start = shift_months(anchor.replace(day=1), direction)
            return (
                start,
                date(
                    start.year,
                    start.month,
                    calendar.monthrange(start.year, start.month)[1],
                ),
            )
        if unit == "year":
            year = anchor.year + direction
            return (date(year, 1, 1), date(year, 12, 31))
        weekday = list(calendar.day_name).index(unit.capitalize())
        delta = (weekday - anchor.weekday()) % 7 or 7
        value = (
            anchor + timedelta(days=delta)
            if direction == 1
            else anchor - timedelta(days=(anchor.weekday() - weekday) % 7 or 7)
        )
        return (value, value)
    future = parts[0] == "in"
    raw_n, unit = (parts[1], parts[2]) if future else (parts[0], parts[1])
    amount = int(raw_n) if raw_n.isdigit() else NUMBERS[raw_n]
    amount *= 1 if future else -1
    if unit.startswith("day"):
        value = anchor + timedelta(days=amount)
    elif unit.startswith("week"):
        value = anchor + timedelta(weeks=amount)
    elif unit.startswith("month"):
        value = shift_months(anchor, amount)
    else:
        value = shift_months(anchor, 12 * amount)
    return (value, value)
