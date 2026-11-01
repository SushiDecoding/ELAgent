"""Rich-based rendering.  The UI only consumes events, never agent internals."""

from __future__ import annotations

import json
import time

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from ..events import Event

ACCENT = "#7aa2f7"
DIM = "#565f89"
OK = "#9ece6a"
ERR = "#f7768e"
WARN = "#e0af68"
TOOL = "#bb9af7"
FG = "#c0caf5"


def _short(value, limit: int = 120) -> str:
    if isinstance(value, (dict, list)):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Renderer:
    def __init__(self, console: Console, *, verbose: bool = False) -> None:
        self.console = console
        self.verbose = verbose
        self.show_reasoning = True
        self._live: Live | None = None
        self._tokens = 0
        self._reasoning_tokens = 0
        self._turn_started = 0.0
        self.last_text = ""
        self.ctx_tokens = 0
        self.ctx_limit = 0

    # ------------------------------------------------------------------
    def header(self, model: str, cwd: str, session_id: str, resumed: bool = False) -> None:
        title = Text()
        title.append("ELAgent", style=f"bold {ACCENT}")
        title.append("  local agent harness", style=DIM)
        body = Text()
        body.append("模型  ", style=DIM)
        body.append(model + "\n", style=FG)
        body.append("目录  ", style=DIM)
        body.append(cwd + "\n", style=FG)
        body.append("会话  ", style=DIM)
        body.append(session_id + ("  (已恢复)" if resumed else "") + "\n", style=FG)
        body.append("提示  ", style=DIM)
        body.append("/help 命令 · /goal 目标模式 · Ctrl+C 中断", style=FG)
        self.console.print(Panel(Group(title, Text(""), body), border_style=ACCENT, padding=(0, 1)))

    def rule(self, label: str, style: str = DIM) -> None:
        self.console.rule(Text(label, style=style), style=DIM)

    # ------------------------------------------------------------------
    def _start_live(self) -> None:
        self._stop_live()
        self._tokens = 0
        self._reasoning_tokens = 0
        self._turn_started = time.monotonic()
        self._live = Live(
            Spinner("dots", text=Text("生成中…", style=DIM)),
            console=self.console,
            refresh_per_second=8,
            transient=True,
        )
        self._live.start()

    def _update_live(self) -> None:
        if not self._live:
            return
        elapsed = time.monotonic() - self._turn_started
        text = Text(f"生成中… {self._tokens} tok · {elapsed:.1f}s", style=DIM)
        if self.show_reasoning and self._reasoning_tokens:
            text.append(f" · 思考 {self._reasoning_tokens} tok", style="#7dcfff")
        try:
            self._live.update(Spinner("dots", text=text))
        except Exception:
            pass

    def stop(self) -> None:
        """Public safety valve: never leave a Live region running."""
        self._stop_live()

    def _stop_live(self) -> None:
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    # ------------------------------------------------------------------
    def handle(self, event: Event) -> None:
        kind, data = event.kind, event.data
        if kind == "turn_start":
            self._start_live()
        elif kind == "delta":
            self._tokens += max(1, len(data.get("text", "")) // 3)
            self._update_live()
        elif kind == "reasoning":
            self._reasoning_tokens += max(1, len(data.get("text", "")) // 3)
        elif kind == "turn_end":
            self._stop_live()
            self.last_text = data.get("text", "")
            self.ctx_tokens = int(data.get("ctx_tokens") or 0)
            self.ctx_limit = int(data.get("limit") or 0)
            reasoning = str(data.get("reasoning") or "").strip()
            if self.show_reasoning and reasoning:
                body = reasoning if len(reasoning) <= 800 else reasoning[:800] + " …（已截断）"
                self.console.print(Text.assemble(("  ◦ 思考 ", DIM), (body.replace("\n", " "), DIM)))
            if self.verbose and self.last_text.strip():
                self.console.print(Text(self.last_text.strip()[:1200], style=DIM))
        elif kind == "tool_start":
            self._render_tool_start(data)
        elif kind == "tool_end":
            self._render_tool_end(data)
        elif kind == "notice":
            style = WARN if data.get("level") == "warn" else DIM
            self.console.print(Text("  ! ", style=style) + Text(str(data.get("text", "")), style=style))
        elif kind == "error":
            self.console.print(Text("  x ", style=ERR) + Text(str(data.get("text", "")), style=ERR))
        elif kind == "compact":
            self.console.print(
                Text(
                    f"  ~ 上下文压缩 {data.get('before')} → {data.get('after')} tok",
                    style=DIM,
                )
            )
        elif kind == "round_start":
            total = data.get("max", -1)
            label = f"第 {data.get('n')} 轮" + ("" if total and total > 0 else " / 无限")
            self.rule(label, ACCENT)
        elif kind == "verdict":
            status = data.get("status")
            if status == 1:
                self.console.print(Text("  ✓ 评审通过", style=OK))
            elif status == 0:
                self.console.print(Text("  x 评审未通过: ", style=ERR) + Text(str(data.get("reason", "")), style=FG))
            else:
                self.console.print(Text("  ? 评审输出无法解析", style=WARN))
        elif kind == "goal_done":
            if data.get("ok"):
                self.console.print(Text(f"  ✓ 目标达成，共 {data.get('rounds')} 轮", style=OK))
            else:
                self.console.print(Text(f"  x 目标未达成，共 {data.get('rounds')} 轮", style=WARN))
        elif kind == "final":
            self._render_final(data)

    # ------------------------------------------------------------------
    def _render_tool_start(self, data: dict) -> None:
        name = data.get("name", "?")
        args = data.get("args") or {}
        if name == "terminalbash" and args.get("command"):
            detail = str(args["command"])
        elif name == "read" and args.get("path"):
            detail = str(args["path"])
            if args.get("offset"):
                detail += f" from {args['offset']}"
        elif name in ("write", "edit") and args.get("path"):
            detail = str(args["path"])
        else:
            detail = _short(args, 100)
        line = Text("  ▸ ", style=TOOL)
        line.append(name, style=f"bold {TOOL}")
        line.append("  " + detail, style=FG)
        self.console.print(line)

    def _render_tool_end(self, data: dict) -> None:
        ok = data.get("ok", True)
        mark = "✓" if ok else "x"
        style = OK if ok else ERR
        summary = data.get("display") or ("ok" if ok else "error")
        ms = data.get("ms") or 0
        line = Text("    └ ", style=DIM)
        line.append(mark, style=style)
        line.append(f" {summary} · {ms}ms", style=DIM)
        self.console.print(line)
        if self.verbose:
            content = str(data.get("content", ""))
            preview = "\n".join(content.splitlines()[:12])
            if preview.strip():
                self.console.print(Text(preview, style=DIM))

    def _render_final(self, data: dict) -> None:
        text = str(data.get("text", "")).strip()
        if not text:
            return
        degraded = bool(data.get("degraded"))
        border = WARN if degraded else OK
        title = "未完成" if degraded else "完成"
        self.console.print(Panel(Markdown(text), title=title, border_style=border, padding=(0, 1)))
        meta = f"{data.get('steps', 0)} 步"
        if data.get("elapsed"):
            meta += f" · {data['elapsed']:.1f}s"
        self.console.print(Text("  " + meta, style=DIM))

    # ------------------------------------------------------------------
    def show_help(self, rows: list[tuple[str, str]]) -> None:
        table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
        table.add_column(style=ACCENT, no_wrap=True)
        table.add_column(style=FG)
        for command, description in rows:
            table.add_row(command, description)
        self.console.print(Panel(table, title="命令", border_style=DIM, padding=(0, 1)))

    def show_tools(self, text: str) -> None:
        self.console.print(Panel(Text(text, style=FG), title="工具", border_style=DIM, padding=(0, 1)))
