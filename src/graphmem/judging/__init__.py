"""The only pieces of the retired V2-V4 demo package the V5 line still needs.

Both benchmarks are judged with mem0's own prompts, verified by sha256 against
upstream, so the prompt module is kept byte-identical rather than rewritten.
"""
from .clients import OpenAICompatibleClient
from .mem0_longmemeval_prompts import get_judge_prompt

__all__ = ["OpenAICompatibleClient", "get_judge_prompt"]
