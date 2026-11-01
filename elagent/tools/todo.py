"""A tiny persistent plan store.

Long-horizon work on a small model needs an external plan: the model forgets,
the file does not.  The plan is also injected into the system prompt so the
model sees its own commitments every turn without spending many tokens.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .base import Tool, ToolContext, ToolResult

MAX_ITEMS = 20
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)、])\s*(?:\[[ xX]\]\s*)?")


def _store_path(ctx: ToolContext) -> Path:
    directory = Path(ctx.config.sessions_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{ctx.session_id}.todo.json"


def load_items(ctx: ToolContext) -> list[dict]:
    path = _store_path(ctx)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("items") if isinstance(data, dict) else data
        return [item for item in (items or []) if isinstance(item, dict) and item.get("text")]
    except Exception:
        return []


def save_items(ctx: ToolContext, items: list[dict]) -> None:
    path = _store_path(ctx)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"items": items[:MAX_ITEMS]}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def normalize_items(value) -> list[str]:
    if isinstance(value, list):
        raw = [str(v) for v in value]
    else:
        raw = str(value or "").splitlines()
    result = []
    for line in raw:
        text = _BULLET_RE.sub("", line).strip()
        if text:
            result.append(text[:160])
    return result[:MAX_ITEMS]


def render(items: list[dict], limit: int = 12) -> str:
    if not items:
        return ""
    done = sum(1 for item in items if item.get("done"))
    lines = [f"计划（{done}/{len(items)} 完成）"]
    for index, item in enumerate(items[:limit], 1):
        mark = "x" if item.get("done") else " "
        lines.append(f"[{mark}] {index}. {item['text']}")
    if len(items) > limit:
        lines.append(f"...另有 {len(items) - limit} 项")
    return "\n".join(lines)


class TodoTool(Tool):
    name = "todo"
    description = "维护任务计划：set/add/done/remove/list"
    params = {"action": "list", "items": "[]", "id": "0"}

    def run(self, args, ctx: ToolContext) -> ToolResult:
        action = self.arg_str(args, "action", "list").strip().lower() or "list"
        items = load_items(ctx)
        item_id = self.arg_int(args, "id", 0)

        if action == "set":
            texts = normalize_items(args.get("items"))
            if not texts:
                return ToolResult(False, "error: set 需要 items 列表", display="items 为空")
            items = [{"text": text, "done": False} for text in texts]
        elif action == "add":
            texts = normalize_items(args.get("items"))
            if not texts:
                return ToolResult(False, "error: add 需要 items", display="items 为空")
            for text in texts:
                if len(items) < MAX_ITEMS:
                    items.append({"text": text, "done": False})
        elif action == "done":
            if not 1 <= item_id <= len(items):
                return ToolResult(False, f"error: id 超出范围 1..{len(items)}", display="id 无效")
            items[item_id - 1]["done"] = True
        elif action == "remove":
            if not 1 <= item_id <= len(items):
                return ToolResult(False, f"error: id 超出范围 1..{len(items)}", display="id 无效")
            items.pop(item_id - 1)
        elif action == "clear":
            items = []
        elif action != "list":
            return ToolResult(False, f"error: 未知 action={action}，可用 set/add/done/remove/list/clear", display="action 无效")

        save_items(ctx, items)
        if not items:
            return ToolResult(True, "计划为空", display="0 项")
        return ToolResult(True, render(items), display=f"{sum(1 for i in items if i.get('done'))}/{len(items)} 完成")
