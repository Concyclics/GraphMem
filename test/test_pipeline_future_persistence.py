from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from graphmem_demo.pipeline import _completed_question_results


def test_completed_question_results_yields_successes_before_raising_failures() -> None:
    failed = Future()
    failed.set_exception(ValueError("transient provider failure"))
    succeeded = Future()
    succeeded.set_result(("answer", ["embedding"], []))

    failed_case = SimpleNamespace(question_id="q-failed")
    succeeded_case = SimpleNamespace(question_id="q-succeeded")
    results = _completed_question_results(
        {
            failed: failed_case,
            succeeded: succeeded_case,
        }
    )

    yielded = []
    with pytest.raises(RuntimeError, match="q-failed") as raised:
        while True:
            yielded.append(next(results))

    assert yielded == [
        (succeeded_case, ("answer", ["embedding"], [])),
    ]
    assert isinstance(raised.value.__cause__, ValueError)


def test_completed_question_results_does_not_swallow_process_interrupts() -> None:
    interrupted = Future()
    interrupted.set_exception(KeyboardInterrupt())
    case = SimpleNamespace(question_id="q-interrupted")

    with pytest.raises(KeyboardInterrupt):
        next(_completed_question_results({interrupted: case}))
