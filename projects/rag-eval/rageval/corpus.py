import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

API_URL = "https://ar.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "rageval/0.1 (retrieval evaluation script; add your contact email here)"
_STOP_HEADINGS = {
    "\u0645\u0631\u0627\u062c\u0639",
    "\u0627\u0644\u0645\u0631\u0627\u062c\u0639",
    "\u0648\u0635\u0644\u0627\u062a \u062e\u0627\u0631\u062c\u064a\u0629",
    "\u0631\u0648\u0627\u0628\u0637 \u062e\u0627\u0631\u062c\u064a\u0629",
    "\u0627\u0646\u0638\u0631 \u0623\u064a\u0636\u0627",
    "\u0627\u0646\u0638\u0631 \u0623\u064a\u0636\u0627\u064b",
    "\u0627\u0646\u0638\u0631 \u0623\u064a\u0636\u064b\u0627",
    "\u0647\u0648\u0627\u0645\u0634",
    "\u0627\u0644\u0645\u0635\u0627\u062f\u0631",
    "\u0645\u0635\u0627\u062f\u0631",
    "\u0645\u0644\u0627\u062d\u0638\u0627\u062a",
}


@dataclass(frozen=True)
class Document:
    title: str
    text: str


def clean_extract(text: str) -> str:
    kept = []
    for line in text.splitlines():
        heading = line.strip().strip("=").strip()
        if heading in _STOP_HEADINGS:
            break
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _request_page(http, headers: dict, title: str, retries: int, sleep) -> dict | None:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "redirects": 1,
        "titles": title,
    }
    last_status = None
    for attempt in range(retries):
        response = http.get(API_URL, params=params, headers=headers, timeout=30)
        last_status = response.status_code
        if response.status_code == 200:
            pages = response.json().get("query", {}).get("pages", [])
            if not pages:
                return None
            page = pages[0]
            if page.get("missing") or "extract" not in page:
                return None
            return page
        if response.status_code in (429, 500, 502, 503, 504):
            sleep(2 ** attempt)
            continue
        break
    raise RuntimeError(f"Wikipedia request failed for a title with HTTP status {last_status}")


def fetch_wikipedia(
    titles: list[str],
    user_agent: str = DEFAULT_USER_AGENT,
    min_chars: int = 3000,
    pause: float = 0.3,
    session=None,
    retries: int = 3,
    sleep=time.sleep,
) -> tuple[list[Document], list[tuple[str, str]]]:
    http = session if session is not None else requests.Session()
    headers = {"User-Agent": user_agent}
    documents: dict[str, Document] = {}
    skipped: list[tuple[str, str]] = []
    for title in titles:
        page = _request_page(http, headers, title, retries, sleep)
        if page is None:
            skipped.append((title, "not_found"))
        else:
            text = clean_extract(page["extract"])
            if len(text) < min_chars:
                skipped.append((title, "too_short"))
            elif page["title"] in documents:
                skipped.append((title, "duplicate_redirect"))
            else:
                documents[page["title"]] = Document(page["title"], text)
        sleep(pause)
    return list(documents.values()), skipped


def save_corpus(documents: list[Document], path: Path) -> None:
    if not documents:
        raise ValueError("Cannot save an empty corpus")
    titles = [document.title for document in documents]
    if len(titles) != len(set(titles)):
        raise ValueError("Corpus document titles must be unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps({"title": document.title, "text": document.text}, ensure_ascii=False) + "\n")


def load_corpus(path: Path) -> list[Document]:
    documents = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                title = str(record["title"]).strip()
                text = str(record["text"]).strip()
                if not title or not text:
                    raise ValueError(f"Invalid corpus record in {path}: title and text are required")
                documents.append(Document(title, text))
    if not documents:
        raise ValueError(f"Corpus is empty: {path}")
    if len({document.title for document in documents}) != len(documents):
        raise ValueError(f"Duplicate document titles in corpus: {path}")
    return documents


def load_titles(path: Path) -> list[str]:
    titles = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            title = line.strip()
            if title and not title.startswith("#"):
                titles.append(title)
    return titles


def export_text_files(documents: list[Document], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index, document in enumerate(documents, start=1):
        safe = re.sub(r'[<>:"/\\|?*\s]+', "_", document.title)[:60].strip("_")
        target = out_dir / f"{index:03d}_{safe}.txt"
        target.write_text(document.text, encoding="utf-8")
        written.append(target)
    return written
