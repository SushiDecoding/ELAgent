"""Session persistence: one JSON file per session, atomic writes, resume support."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _slug(text: str, limit: int = 40) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text or "").strip("-")
    return cleaned[:limit] or "session"


@dataclass
class Session:
    id: str
    cwd: str
    model: str
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    title: str = ""
    goal_mode: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    memory_asked: bool = False
    memory_enabled: bool = True

    def touch(self) -> None:
        self.updated = _now()

    def add_usage(self, prompt: int, completion: int) -> None:
        self.usage["prompt"] = int(self.usage.get("prompt", 0)) + int(prompt or 0)
        self.usage["completion"] = int(self.usage.get("completion", 0)) + int(completion or 0)
        self.usage["total"] = self.usage["prompt"] + self.usage["completion"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class SessionStore:
    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.dir = self.home / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.home / "state.json"

    # ------------------------------------------------------------------
    def new(self, cwd: str, model: str, title: str = "") -> Session:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session = Session(
            id=f"{stamp}-{uuid.uuid4().hex[:4]}",
            cwd=str(cwd),
            model=model,
            title=title,
        )
        self.save(session)
        return session

    def path_for(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def save(self, session: Session) -> None:
        session.touch()
        target = self.path_for(session.id)
        tmp = target.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, target)
        except OSError:
            # last resort: never crash the loop because a session could not be saved
            try:
                target.write_text(json.dumps(session.to_dict(), ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
        self._set_last(session.cwd, session.id)

    def load(self, session_id: str) -> Session | None:
        target = self.path_for(session_id)
        if not target.exists():
            return None
        try:
            return Session.from_dict(json.loads(target.read_text(encoding="utf-8")))
        except Exception:
            return None

    def list(self, cwd: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for file in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if cwd and str(data.get("cwd", "")).lower() != str(cwd).lower():
                continue
            rows.append(
                {
                    "id": data.get("id", file.stem),
                    "title": data.get("title") or "(未命名)",
                    "updated": data.get("updated", ""),
                    "messages": len(data.get("messages") or []),
                    "rounds": len(data.get("rounds") or []),
                    "cwd": data.get("cwd", ""),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    # ------------------------------------------------------------------
    def _read_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {}
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _set_last(self, cwd: str, session_id: str) -> None:
        state = self._read_state()
        last = state.setdefault("last", {})
        last[str(cwd).lower()] = session_id
        try:
            self.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass

    def latest(self, cwd: str) -> Session | None:
        state = self._read_state()
        session_id = (state.get("last") or {}).get(str(cwd).lower())
        if session_id:
            session = self.load(session_id)
            if session:
                return session
        rows = self.list(cwd=cwd, limit=1)
        return self.load(rows[0]["id"]) if rows else None

    # ------------------------------------------------------------------
    def export_markdown(self, session: Session) -> str:
        lines = [
            f"# ELAgent Session {session.id}",
            "",
            f"- 工作目录: `{session.cwd}`",
            f"- 模型: `{session.model}`",
            f"- 创建: {session.created}",
            f"- 更新: {session.updated}",
            f"- 用量: {session.usage.get('total', 0)} tokens",
            "",
        ]
        if session.rounds:
            lines.append("## 轮次")
            for item in session.rounds:
                lines.append(
                    f"- 第 {item.get('n')} 轮: status={item.get('status')} "
                    f"steps={item.get('steps')} reason={item.get('reason', '')}"
                )
            lines.append("")
        lines.append("## 对话")
        for message in session.messages:
            role = message.get("role", "?")
            content = message.get("content", "")
            lines.append(f"### {role}")
            lines.append("")
            lines.append(str(content))
            lines.append("")
        return "\n".join(lines)

    def save_markdown(self, session: Session, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.export_markdown(session), encoding="utf-8")
        return path
