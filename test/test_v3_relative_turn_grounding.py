from datetime import date

from graphmem_demo.v3.state_temporal_operators import _turn_event_day


def test_turn_relative_time_is_anchored_to_observation_date() -> None:
    observed = date(2023, 3, 15)
    assert _turn_event_day(observed, "I got it today.") == observed
    assert _turn_event_day(observed, "I met her yesterday.") == date(2023, 3, 14)
    assert _turn_event_day(observed, "I ran into her a few days ago.") == date(2023, 3, 12)
    assert _turn_event_day(observed, "I bought it 10 days ago.") == date(2023, 3, 5)
    assert _turn_event_day(observed, "I got it a few weeks ago.") == date(2023, 2, 22)
