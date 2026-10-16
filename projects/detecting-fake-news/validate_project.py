"""Static validation for the fake-news notebook (no dataset or ML runtime required)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "detecting-fake-news.ipynb"
REQUIRED_FILES = ("README.md", "requirements.txt", ".gitignore", NOTEBOOK.name)


def main() -> int:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"Missing required project files: {', '.join(missing)}")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise SystemExit("Notebook is not a valid nbformat 4 document")

    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if not code_cells:
        raise SystemExit("Notebook contains no code cells")
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source, filename=f"{NOTEBOOK.name}:cell-{index}")
        except SyntaxError as error:
            raise SystemExit(f"Syntax error in code cell {index}: {error}") from error
        if cell.get("outputs"):
            raise SystemExit(f"Code cell {index} contains stale outputs")
        if cell.get("execution_count") is not None:
            raise SystemExit(f"Code cell {index} has a stale execution count")
        if any(marker in source for marker in ("/content/", "C:\\\\", "\\\\Users\\\\")):
            raise SystemExit(f"Non-portable path found in code cell {index}")

    all_source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    required_markers = (
        "FAKE_NEWS_DATA_DIR", "train_test_split", "stratify=", "fit_transform(text_train)",
        "transform(text_val)", "transform(text_test)", "random_state=SEED",
    )
    missing_markers = [marker for marker in required_markers if marker not in all_source]
    if missing_markers:
        raise SystemExit(f"Notebook is missing methodology markers: {', '.join(missing_markers)}")
    print(f"OK: {len(code_cells)} code cells compile; no stale outputs; methodology markers present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
