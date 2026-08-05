from graphmem.build.temporal import extract_time_expression, normalize_time, observed_interval


def test_absolute_natural_date_normalizes_to_interval() -> None:
    row = normalize_time("5:51 pm on 21 October, 2023", None, "turn:1")
    assert row.kind == "absolute"
    assert row.start == "2023-10-21T17:51:00"
    assert row.precision == "minute"


def test_relative_time_anchors_to_observation_date() -> None:
    row = normalize_time("yesterday", "2023/03/27 (Mon) 15:15", "turn:2")
    assert row.kind == "relative"
    assert row.start == "2023-03-26T00:00:00"
    assert row.anchor_turn_id == "turn:2"


def test_unresolved_time_never_creates_false_ordering_endpoint() -> None:
    row = normalize_time("sometime eventually", "2023-03-27", "turn:3")
    assert row.kind == "unresolved"
    assert row.start is None and row.end is None
    observed = observed_interval("2023-03-27", "turn:3")
    assert observed and observed.kind == "observed"


def test_rule_extractor_resolves_relative_phrase_against_turn_timestamp() -> None:
    phrase = extract_time_expression("We held the birthday dinner about a month ago at home.")
    assert phrase == "about a month ago"
    row = normalize_time(phrase, "2023-05-27", "turn:4")
    assert row.kind == "relative"
    assert row.start == "2023-04-01T00:00:00"


def test_last_weekday_resolves_backwards_not_forwards() -> None:
    phrase = extract_time_expression("I won the final last Friday.")
    assert phrase == "last Friday"
    row = normalize_time(phrase, "2023-05-29", "turn:5")
    assert row.start == "2023-05-26T00:00:00"


def test_duration_produces_approximate_start_month() -> None:
    phrase = extract_time_expression("I've been playing professionally for just under a year now.")
    assert phrase == "for just under a year now"
    row = normalize_time(phrase, "2023-12-06", "turn:6")
    assert row.kind == "relative"
    assert row.precision == "month"
    assert row.start == "2023-01-01T00:00:00"
