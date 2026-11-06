import re

INVALID = "INVALID"

_CWE_PATTERN = re.compile(r"CWE[\s_:\-]*0*(\d+)", re.IGNORECASE)
_BARE_NUMBER = re.compile(r"^\s*0*(\d+)\s*\.?\s*$")


def canonical(value) -> str | None:
    text = "" if value is None else str(value).strip()
    match = _CWE_PATTERN.search(text) or _BARE_NUMBER.match(text)
    if match is None:
        return None
    return f"CWE-{int(match.group(1))}"


def parse_prediction(text: str) -> str:
    return canonical(text) or INVALID
