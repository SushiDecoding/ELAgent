"""Tool protocol: one class per tool, a registry, and a single dispatch path."""

from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class ToolError(Exception):
    """Expected, model-recoverable tool failure."""


@dataclass
class ToolResult:
    ok: bool
    content: str
    display: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls(False, f"error: {message}", display=message[:80])


AskUserFn = Callable[[str], "str | None"]
SubagentFn = Callable[[str, int], "tuple[bool, str, list[str]]"]
WorkflowFn = Callable[["list[str]"], "tuple[bool, str, list[str]]"]


@dataclass
class ToolContext:
    workspace: Path
    config: Any
    session_id: str = "default"
    cancel: threading.Event | None = None
    confirm: Callable[[str], bool] | None = None
    ask_user: AskUserFn | None = None
    subagent: SubagentFn | None = None
    workflow: WorkflowFn | None = None
    depth: int = 0

    def resolve(self, raw: str, *, must_exist: bool = False) -> Path:
        if raw is None or str(raw).strip() == "":
            raw = "."
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate.absolute()
        root = self.workspace.resolve()
        if not self.config.allow_outside_workspace:
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ToolError(
                    f"路径越界: {raw}（仅允许工作目录 {root} 内；如需放开请设置 allow_outside_workspace）"
                ) from exc
        if must_exist and not resolved.exists():
            hint = self._suggest(resolved)
            message = f"路径不存在: {raw}"
            if hint:
                message += f"；是否指 {hint}？下一步用它重试，不要重复原路径"
            else:
                message += "；先 ls 查看目录，再照抄返回的名称"
            raise ToolError(message)
        return resolved

    def _suggest(self, path: Path) -> str:
        """Closest existing sibling name; breaks the classic dot-prefix loop."""
        try:
            parent = path.parent
            if not parent.is_dir():
                return ""
            names = []
            for child in parent.iterdir():
                names.append(child.name)
                if len(names) >= 400:
                    break
        except OSError:
            return ""
        import difflib

        matches = difflib.get_close_matches(path.name, names, n=1, cutoff=0.6)
        if not matches:
            return ""
        return self.rel(parent / matches[0])

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.workspace.resolve()).as_posix()
        except Exception:
            return str(path)

    def cancelled(self) -> bool:
        return bool(self.cancel and self.cancel.is_set())

    def ask(self, question: str) -> bool:
        if self.confirm is None:
            return False
        try:
            return bool(self.confirm(question))
        except Exception:
            return False


class Tool:
    name: str = ""
    description: str = ""
    params: dict[str, str] = {}
    danger: bool = False

    def signature(self) -> str:
        parts = []
        for key, default in self.params.items():
            parts.append(key if default == "" else f"{key}={default}")
        return f"{self.name}({', '.join(parts)})"

    def line(self) -> str:
        return f"{self.signature()} {self.description}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:  # pragma: no cover
        raise NotImplementedError

    # helpers ---------------------------------------------------------
    @staticmethod
    def arg_str(args: dict[str, Any], key: str, default: str = "") -> str:
        value = args.get(key, default)
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def arg_int(args: dict[str, Any], key: str, default: int) -> int:
        value = args.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def arg_bool(args: dict[str, Any], key: str, default: bool = False) -> bool:
        value = args.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(value)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("tool 缺少 name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def without(self, *names: str) -> "ToolRegistry":
        """Copy without the named tools (used to keep sub-agents leaf-only)."""
        drop = set(names)
        return ToolRegistry([tool for name, tool in self._tools.items() if name not in drop])

    def describe(self) -> str:
        return "\n".join(self._tools[name].line() for name in self.names())

    def describe_full(self) -> str:
        rows = []
        for name in self.names():
            tool = self._tools[name]
            params = ", ".join(f"{k}={v}" for k, v in tool.params.items())
            rows.append(f"- {name}({params}): {tool.description}")
        return "\n".join(rows)

    def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(f"未知工具 {name}；可用工具: {', '.join(self.names())}")
        if not isinstance(args, dict):
            args = {}
        started = time.monotonic()
        try:
            result = tool.run(args, ctx)
            if not isinstance(result, ToolResult):
                result = ToolResult(True, str(result))
        except ToolError as exc:
            result = ToolResult.error(str(exc))
        except FileNotFoundError as exc:
            result = ToolResult.error(f"文件不存在: {exc.filename or exc}")
        except PermissionError as exc:
            result = ToolResult.error(f"权限不足: {exc}")
        except UnicodeDecodeError:
            result = ToolResult.error("文件不是 UTF-8 文本（可能是二进制文件）")
        except Exception as exc:  # noqa: BLE001 - never let a tool kill the loop
            traceback.print_exc()
            result = ToolResult.error(f"{type(exc).__name__}: {exc}")
        result.meta.setdefault("ms", int((time.monotonic() - started) * 1000))
        limit = int(getattr(ctx.config, "tool_output_limit", 6000))
        if len(result.content) > limit * 4:
            result.content = result.content[: limit * 4] + "\n...（输出过长已截断）"
        return result
