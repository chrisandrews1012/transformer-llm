"""
Minimal SFT prompt formatting helpers.
"""

from __future__ import annotations


def format_sft_prompt(instruction: str, input_text: str = "") -> str:
    """
    Format a tiny instruction/input prefix for decoder-only SFT examples.

    :param instruction: The instruction to follow.
    :type instruction: str
    :param input_text: Optional additional input context.
    :type input_text: str
    :returns: Formatted prompt string with "Instruction:", optional "Input:",
        and "Response:" sections.
    :rtype: str
    """
    instruction = instruction.strip()
    input_text = input_text.strip()

    if input_text:
        return f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    return f"Instruction: {instruction}\nResponse:"
