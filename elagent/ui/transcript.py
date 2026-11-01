"""Transcript model for the full-screen TUI.

Each block is a Rich renderable.  Blocks are rendered through a Rich console
with `force_terminal=True`, so we get ANSI text that prompt_toolkit's `ANSI`
parser turns into styled fragments.  Rendering is cached per width: a resize
re-renders everything, a normal frame is a slice of cached lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import Any

from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .render import ACCENT, DIM, ERR, FG, OK, TOOL, WARN

Fragment = tuple[str, str]


def render_ansi(renderable: Any, width: int) -> list[list[Fragment]]:
    """Render a Rich object to a list of styled lines (already wrapped)."""
    width = max(12, int(width))
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=width,
        height=24,  # Rich falls back to the detected size when height is None
        force_terminal=True,
        color_system="truecolor",
        legacy_windows=False,
        highlight=False,
        soft_wrap=False,
        emoji=False,
    )
    console.print(renderable)
    raw = buffer.getvalue()
    if raw.endswith("\n"):
        raw = raw[:-1]
    if not raw:
        return [[]]
    return [to_formatted_text(ANSI(line)) for line in raw.split("\n")]


def slice_lines(lines: list[Any], height: int, scroll: int) -> tuple[list[Any], int]:
    """Take the visible window.  scroll=0 sticks to the bottom."""
    height = max(1, height)
    max_scroll = max(0, len(lines) - height)
    scroll = max(0, min(scroll, max_scroll))
    start = max(0, len(lines) - height - scroll)
    return lines[start : start + height], scroll


@dataclass
class Block:
    renderable: Any
    lines_: list[list[Fragment]] = field(default_factory=list, repr=False)
    width: int = 0

    def lines(self, width: int) -> list[list[Fragment]]:
        if width != self.width or not self.lines_:
            self.lines_ = render_ansi(self.renderable, width)
            self.width = width
        return self.lines_


class Transcript:
    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.live = ""          # streaming assistant output, shown dimly at the tail
        self.reasoning_live = ""  # streaming reasoning, shown dimly above the output

    # -- mutation -------------------------------------------------------
    def add(self, renderable: Any) -> None:
        self.blocks.append(Block(renderable))
        self.live = ""
        self.reasoning_live = ""

    def clear(self) -> None:
        self.blocks.clear()
        self.live = ""
        self.reasoning_live = ""

    def set_live(self, text: str) -> None:
        self.live = text or ""

    def set_reasoning(self, text: str) -> None:
        self.reasoning_live = text or ""

    def clear_live(self) -> None:
        self.live = ""
        self.reasoning_live = ""

    # -- rendering ------------------------------------------------------
    def lines(self, width: int) -> list[list[Fragment]]:
        out: list[list[Fragment]] = []
        for block in self.blocks:
            out.extend(block.lines(width))
        if self.reasoning_live.strip():
            tail = self.reasoning_live.strip()[-500:].replace("\n", " ")
            fragment = render_ansi(Text("… 思考 " + tail, style=DIM), width)
            out.extend(fragment[-5:])
        if self.live.strip():
            tail = self.live.strip()[-600:].replace("\n", " ")
            fragment = render_ansi(Text("… " + tail, style=DIM), width)
            out.extend(fragment[-6:])
        return out


# ----------------------------------------------------------------------
# block builders (used by the app and by tests)
# ----------------------------------------------------------------------
def user_block(text: str) -> Text:
    return Text.assemble(("› ", f"bold {ACCENT}"), (text, FG))


def assistant_block(text: str) -> Text:
    return Text(text, style=FG)


def final_block(text: str, degraded: bool = False) -> Panel:
    return Panel(
        Markdown(text),
        title="未完成" if degraded else "完成",
        border_style=WARN if degraded else OK,
        padding=(0, 1),
    )


def tool_start_block(name: str, detail: str) -> Text:
    line = Text()
    line.append("  ▸ ", style=TOOL)
    line.append(name, style=f"bold {TOOL}")
    line.append("  " + detail, style=FG)
    return line


def tool_end_block(ok: bool, summary: str, ms: int = 0) -> Text:
    line = Text("    └ ", style=DIM)
    line.append("✓" if ok else "x", style=OK if ok else ERR)
    suffix = f" {summary}" if summary else ""
    line.append(f"{suffix} · {ms}ms", style=DIM)
    return line


def thinking_block(text: str, limit: int = 1500) -> Text:
    body = str(text).strip()
    if len(body) > limit:
        body = body[:limit] + " …（已截断）"
    return Text.assemble(("  ◦ 思考 ", DIM), (body.replace("\n", " "), DIM))


def notice_block(text: str, level: str = "warn") -> Text:
    style = WARN if level == "warn" else DIM
    return Text.assemble(("  ! ", style), (str(text), style))


def error_block(text: str) -> Text:
    return Text.assemble(("  x ", ERR), (str(text), ERR))


def rule_block(label: str) -> Rule:
    return Rule(Text(label, style=ACCENT), style=DIM)


def verdict_block(status: int, reason: str) -> Text:
    if status == 1:
        return Text("  ✓ 评审通过", style=OK)
    if status == 0:
        return Text.assemble(("  x 评审未通过: ", ERR), (str(reason), FG))
    return Text("  ? 评审输出无法解析", style=WARN)


def kv_table(rows: list[tuple[str, str]], title: str | None = None) -> Table:
    table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    table.add_column(style=DIM, no_wrap=True)
    table.add_column(style=FG)
    for key, value in rows:
        table.add_row(key, str(value))
    if title:
        return Panel(table, title=title, border_style=DIM, padding=(0, 1))
    return table


def command_table(rows: list[tuple[str, str]]) -> Panel:
    table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    table.add_column(style=ACCENT, no_wrap=True)
    table.add_column(style=FG)
    for command, description in rows:
        table.add_row(command, description)
    return Panel(table, title="命令", border_style=DIM, padding=(0, 1))


def welcome_block(model: str, cwd: str, session_id: str, resumed: bool, error: str = "") -> Panel:
    body = Text()
    body.append("模型  ", style=DIM)
    body.append(model + "\n", style=FG)
    body.append("目录  ", style=DIM)
    body.append(cwd + "\n", style=FG)
    body.append("会话  ", style=DIM)
    body.append(session_id + ("  (已恢复)" if resumed else "") + "\n", style=FG)
    body.append("提示  ", style=DIM)
    body.append("/help 命令 · /goal 目标模式 · Ctrl+C 中断 · Ctrl+D 退出", style=FG)
    pieces: list[Any] = [Text.assemble(("ELAgent", f"bold {ACCENT}"), ("  local agent harness", DIM)), Text("")]
    pieces.append(body)
    if error:
        pieces.append(Text(""))
        pieces.append(Text.assemble(("连接失败: ", ERR), (error, ERR)))
    return Panel(Group(*pieces), border_style=ACCENT, padding=(0, 1))
