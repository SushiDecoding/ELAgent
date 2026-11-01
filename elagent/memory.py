"""global.memory: a plain-text, human-editable long-term experience file.

Write path: after a task finishes, Python asks the model for 3-5 bullets, cleans
them, dedupes and appends.  Read path: on a new task the agent can recall the
few most relevant entries (keyword overlap), never the whole file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import prompts
from .tokens import estimate_tokens, truncate_to_tokens

HEADER = "# ELAgent Global Memory\n\n> 由 ELAgent 维护，可手工编辑。每次任务追加一条经验。\n"


@dataclass
class Entry:
    header: str
    body: str

    @property
    def text(self) -> str:
        return f"## {self.header}\n{self.body}".strip()


def _cjk_bigrams(text: str) -> set[str]:
    grams: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        for i in range(len(run) - 1):
            grams.add(run[i : i + 2])
    return grams


def _keywords(text: str) -> set[str]:
    words = {w.lower() for w in re.findall(r"[A-Za-z_][\w\-.]{2,}", text or "")}
    return words | _cjk_bigrams(text or "")


class GlobalMemory:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > len(HEADER)

    def read_raw(self) -> str:
        if not self.path.exists():
            return ""
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def entries(self) -> list[Entry]:
        raw = self.read_raw()
        if not raw.strip():
            return []
        chunks = re.split(r"(?m)^##\s+", raw)
        result: list[Entry] = []
        for chunk in chunks[1:]:
            lines = chunk.strip().splitlines()
            if not lines:
                continue
            header = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            if body:
                result.append(Entry(header, body))
        return result

    # ------------------------------------------------------------------
    def append(self, task: str, bullets: list[str]) -> bool:
        cleaned = []
        seen = set()
        for bullet in bullets:
            text = re.sub(r"^\s*[-*+]\s*", "", str(bullet)).strip()
            if not text or text in ("无", "none", "N/A"):
                continue
            key = re.sub(r"\s+", "", text).lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text[:200])
        if not cleaned:
            return False

        existing = {re.sub(r"\s+", "", line.lower()) for line in self.read_raw().splitlines()}
        fresh = [b for b in cleaned if re.sub(r"\s+", "", ("- " + b).lower()) not in existing]
        if not fresh:
            return False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(HEADER, encoding="utf-8")
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        title = re.sub(r"\s+", " ", task.strip())[:60]
        block = f"\n## {stamp} · {title}\n" + "\n".join(f"- {b}" for b in fresh) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(block)
        self._trim()
        return True

    def _trim(self, keep: int = 200) -> None:
        if self.path.stat().st_size < 200_000:
            return
        entries = self.entries()[-keep:]
        self.path.write_text(HEADER + "\n" + "\n\n".join(e.text for e in entries) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    def recall(self, query: str, k: int = 3) -> list[Entry]:
        entries = self.entries()
        if not entries:
            return []
        query_keys = _keywords(query)
        scored: list[tuple[float, int, Entry]] = []
        for index, entry in enumerate(entries):
            keys = _keywords(entry.text)
            overlap = len(query_keys & keys)
            recency = index / max(1, len(entries))
            score = overlap / (len(keys) ** 0.5 + 1) + recency * 0.15
            if overlap > 0 or index >= len(entries) - 2:
                scored.append((score, index, entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [entry for _, _, entry in scored[:k]]

    def render(self, entries: list[Entry], limit_tokens: int = 700) -> str:
        if not entries:
            return ""
        text = "\n\n".join(entry.text for entry in entries)
        return truncate_to_tokens(text, limit_tokens)

    def stats(self) -> dict[str, int]:
        entries = self.entries()
        return {
            "entries": len(entries),
            "tokens": estimate_tokens(self.read_raw()),
            "bytes": self.path.stat().st_size if self.path.exists() else 0,
        }

    # ------------------------------------------------------------------
    def distill(self, llm, task: str, summary: str) -> list[str]:
        try:
            response = llm.chat(
                [{"role": "user", "content": prompts.memory_prompt(task, truncate_to_tokens(summary, 1200))}],
                stream=False,
                temperature=0.2,
                max_tokens=300,
            )
        except Exception:
            return []
        bullets = []
        for line in (response.text or "").splitlines():
            text = line.strip()
            if text.startswith(("-", "*", "•")):
                bullets.append(text.lstrip("-*• ").strip())
        return bullets[:5]
