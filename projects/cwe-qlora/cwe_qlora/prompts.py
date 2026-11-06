import re

SYSTEM_PROMPT = (
    "You are a vulnerability analyst. Read the CVE description and reply with the single most "
    "specific CWE identifier that describes the underlying weakness, in the form CWE-<number>. "
    "Reply with the identifier only."
)
MAX_DESCRIPTION_CHARS = 1500


def clean_description(text, max_chars: int = MAX_DESCRIPTION_CHARS) -> str:
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    return collapsed[:max_chars]


def build_messages(description: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": clean_description(description)},
    ]
