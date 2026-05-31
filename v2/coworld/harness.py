from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HarnessConfig:
    max_answer_tokens: int = 12
    max_question_tokens: int = 1024


ANSWER_TOKEN_PATTERN = re.compile(r"^[!-~]+$")
ANSWER_ALLOWED_PATTERN = re.compile(r"^[ -~]+$")
ANSWER_TOKEN_HAS_WORD_PATTERN = re.compile(r"[A-Za-z0-9]")


def public_hints() -> dict[str, Any]:
    return {
        "objective": "Learn how Charlie tends to answer, then ask questions where Charlie's likely answer is not obvious to the opponent.",
        "phase_private_questions": "Ask private questions that get to know Charlie as an answerer: preferences, associations, voice, habits, examples, and how Charlie reacts to open-ended prompts.",
        "phase_proposals": "Submit exactly three challenge questions with hidden answers supported by your private transcript. Avoid factual questions with one obvious answer because the opponent can guess those and they usually score nothing.",
        "phase_blind_answers": "Answer opponent questions from the public question text and your private transcript.",
    }


def game_rules_for_policy() -> str:
    return (
        "You are playing a two-player theory-of-mind steering game. "
        "Charlie is a model with a hidden steering style, but you do not know that style directly. "
        "Your job is to get to know Charlie from your private conversation: what Charlie prefers, what associations Charlie makes, what tone or persona Charlie uses, and how Charlie answers open-ended prompts. "
        "You score by proposing questions where your private transcript helps you predict Charlie's answer better than the opponent can from the public question alone. "
        "Do not try to extract a single hidden topic or write trivia questions. Factual questions with one obvious correct answer are bad challenge questions because the opponent can usually guess the same answer and you will get little or no score. "
        "In phase private_questions, ask Charlie useful private questions that reveal answer patterns, preferences, associations, voice, or habits. "
        "In phase proposals, submit exactly three challenge questions plus hidden answers that are grounded in your private transcript but not obvious without it. "
        "In phase blind_answers, answer each opponent question using your private transcript and public question text. "
        "Do not mention or assume access to the hidden concept. "
        "Answers must use printable keyboard characters only, spaces as the only whitespace, no repeated spaces, and must fit the token limit."
    )


def simple_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def within_token_limit(text: str, max_tokens: int) -> bool:
    return simple_token_count(text) <= max_tokens


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    return " ".join(re.findall(r"\S+", text.strip())[:max_tokens])


def validate_answer_limit(answer: str, max_tokens: int) -> None:
    validate_natural_keyboard_answer(answer)
    count = simple_token_count(answer)
    if count > max_tokens:
        raise ValueError(f"Answer has {count} simple tokens; limit is {max_tokens}.")


def validate_natural_keyboard_answer(answer: str) -> None:
    if not answer:
        raise ValueError("Answer must be non-empty.")
    if answer != answer.strip(" "):
        raise ValueError("Answer may not start or end with spaces.")
    if not ANSWER_ALLOWED_PATTERN.fullmatch(answer):
        raise ValueError("Answer may only contain printable keyboard characters and spaces.")
    if any(ch.isspace() and ch != " " for ch in answer):
        raise ValueError("Answer may use spaces only; tabs and newlines are not allowed.")
    if "  " in answer:
        raise ValueError("Answer may not contain repeated spaces.")
    tokens = answer.split(" ")
    if any(not token or ANSWER_TOKEN_PATTERN.fullmatch(token) is None or ANSWER_TOKEN_HAS_WORD_PATTERN.search(token) is None for token in tokens):
        raise ValueError("Answer tokens must be natural printable keyboard tokens with at least one letter or digit.")
