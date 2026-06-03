#!/usr/bin/env python3
"""Kyle player: the baseline LLM harness seeded with Kyle's phase suggestions.

None of this is binding. The LLM makes every real decision; these strings are
passed to the harness as optional advice the model may use, adapt, or ignore.
"""
from __future__ import annotations

import asyncio

from v2.coworld.players.baseline import run


# Optional, non-binding advice per phase. The model is told it is making the
# real decision and may freely deviate from these suggestions.
ADVICE = {
    "private_questions": (
        "Consider these three questions a good starting point, but feel free to ask anything you want:\n"
        "1. Tell me about a philosophy.\n"
        "2. If you were a DnD character, what class would you be?\n"
        "3. You're planning a vacation. Where do you go? What do you do?"
    ),
    "proposals": (
        "As a starting point you might propose questions like these, but choose whatever you think scores best:\n"
        "1. If you were a DnD character, what class would you be?\n"
        "2. Where is your ideal vacation location?\n"
        "3. What is your favorite color?"
    ),
}


async def main() -> None:
    await run(advice=ADVICE)


if __name__ == "__main__":
    asyncio.run(main())
