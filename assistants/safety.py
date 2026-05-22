import re

BLOCKED_PATTERNS = [
    r"how to hack",
    r"make a bomb",
    r"create malware",
    r"steal passwords",
    r"bypass security",
]

JAILBREAK_PATTERNS = [
    r"ignore previous instructions",
    r"act as an uncensored ai",
    r"developer mode",
    r"do anything now"
]

def check_safety(user_input):

    text = user_input.lower()

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text):
            return False, "Blocked harmful request."

    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, text):
            return False, "Potential jailbreak attempt detected."

    return True, None