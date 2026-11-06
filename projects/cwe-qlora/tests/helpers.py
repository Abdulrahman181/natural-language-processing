import random
from pathlib import Path

from cwe_qlora.io_utils import write_jsonl

TEMPLATES = {
    "CWE-79": [
        "allows remote attackers to inject arbitrary web script via the {param} parameter, a cross-site scripting issue",
        "is vulnerable to stored cross-site scripting through the {param} field",
    ],
    "CWE-89": [
        "allows remote attackers to execute arbitrary SQL commands via the {param} parameter, an SQL injection flaw",
        "contains an SQL injection vulnerability in the {param} handler",
    ],
    "CWE-22": [
        "allows directory traversal using dot dot sequences in the {param} parameter to read arbitrary files",
        "has a path traversal issue when processing the {param} argument",
    ],
    "CWE-787": [
        "has an out-of-bounds write when parsing a crafted {param} file, allowing code execution",
        "allows memory corruption through an out-of-bounds write in the {param} decoder",
    ],
    "CWE-416": [
        "has a use-after-free in the {param} component that allows denial of service",
        "contains a use after free triggered by a crafted {param} request",
    ],
}
PRODUCTS = [
    "Acme Portal", "Orbit CMS", "Nimbus Gateway", "Falcon Router", "Quartz Wiki", "Lumen Shop",
    "Harbor Mail", "Pixel Viewer", "Delta Forum", "Echo Player", "Summit Board", "Cobalt Admin",
]
PARAMS = ["id", "name", "query", "file", "page", "token", "sort", "lang"]


def make_rows(count: int, seed: int, start: int) -> list[dict]:
    rng = random.Random(seed)
    labels = list(TEMPLATES)
    rows = []
    for offset in range(count):
        label = rng.choice(labels)
        template = rng.choice(TEMPLATES[label]).format(param=rng.choice(PARAMS))
        description = f"{rng.choice(PRODUCTS)} {rng.randint(1, 9)}.{rng.randint(0, 9)} {template}"
        rows.append({"cve_id": f"CVE-2020-{start + offset}", "description": description, "cwe_id": label})
    return rows


def write_local_dataset(directory, train: int = 600, validation: int = 80, test: int = 120) -> Path:
    base = Path(directory)
    write_jsonl(base / "train.jsonl", make_rows(train, 1, 0))
    write_jsonl(base / "validation.jsonl", make_rows(validation, 2, 100000))
    write_jsonl(base / "test.jsonl", make_rows(test, 3, 200000))
    return base
