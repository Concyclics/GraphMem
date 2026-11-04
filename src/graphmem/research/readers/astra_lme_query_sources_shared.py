"""Same final Astra reader; source preparation becomes answer-independent."""

from .astra_lme_elapsed_shared import instruction as previous_instruction
from .astra_stage2_shared import INSTRUCTION, request_body


def instruction(branch):
    return (
        INSTRUCTION if branch in [None, "base_reader"] else previous_instruction(branch)
    )


def payload(row):
    body = request_body(row)
    body["instructions"] = instruction(row.get("branch"))
    return body
