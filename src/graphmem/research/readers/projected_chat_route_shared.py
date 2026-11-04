"""Keep both speakers when the question explicitly refers to a previous chat."""

import re

PREVIOUS_CHAT = re.compile(r"\b(?:previous|earlier|last|past)\s+(?:chat|chats)\b", re.I)


def keep_full_dialogue(question):
    return bool(PREVIOUS_CHAT.search(question))
