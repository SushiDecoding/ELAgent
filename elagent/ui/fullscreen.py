"""Full-screen TUI, modelled on Pi's terminal UI.

Layout:
    +----------------------------------------------+
    | header:  ELAgent · model · cwd · session     |
    | transcript: scrollable, streams live         |
    | modal (conditional): y/n confirmations       |
    | status:  mode · spinner · ctx · tokens       |
    | input:   > one-line editor                   |
    +----------------------------------------------+

The agent runs on a worker thread; every event is marshalled onto the asyncio
loop with `call_soon_threadsafe`, so the UI never blocks and never crashes on a
tool failure.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style
from rich.panel import Panel
from rich.text import Text

from .. import __version__
from ..agent import GoalRunner, MainAgent
from ..config import Config
from ..context import ContextManager
from ..events import Event, EventBus
from ..llm import LLMClient
from ..memory import GlobalMemory
from ..session import Session, SessionStore
from ..tools import ToolContext, build_registry
from ..tools.shell import BashTool
from . import commands
from .render import DIM, ERR, OK
from .transcript import (
    Transcript,
    assistant_block,
    error_block,
    final_block,
    notice_block,
    rule_block,
    slice_lines,
    thinking_block,
    tool_end_block,
    tool_start_block,
    user_block,
    verdict_block,
    welcome_block,
)
from .tui import ElCompleter

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_STYLE = Style.from_dict(
    {
        "": "#c0caf5 bg:#1a1b26",
        "header": "bg:#1f2335 #7aa2f7 bold",
        "body": "bg:#1a1b26",
        "status": "bg:#1f2335 #a9b1d6",
        "status.running": "bg:#1f2335 #e0af68",
        "input": "bg:#1a1b26",
        "prompt": "#7aa2f7 bold",
        "modal": "bg:#3b2f1f #e0af68 bold",
        "dim": "#565f89",
        "accent": "#7aa2f7",
        "ok": "#9ece6a",
        "warn": "#e0af68",
        "err": "#f7768e",
        "tool": "#bb9af7",
    }
)


class TranscriptControl(FormattedTextControl):
    """Renders the visible slice; learns its real size on every frame."""

    def __init__(self, app: "FullApp") -> None:
        self.app = app
        self.width = 80
        self.height = 20
        super().__init__(text=self._get_text, focusable=False)

    def _get_text(self):
        return self.app.visible_fragments(self.width, self.height)

    def create_content(self, width: int, height: int):
        if width:
            self.width = width
        if height:
            self.height = height
        self.app.on_viewport(self.width, self.height)
        return super().create_content(width, height)


class FullApp:
    def __init__(self, cfg: Config, resume: bool = False) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self.bus = EventBus()
        self.llm = LLMClient(cfg)
        self.tools = build_registry()
        self.store = SessionStore(cfg.home)
        self.memory = GlobalMemory(cfg.memory_file)

        self.model_error = ""
        try:
            self.model = self.llm.resolve_model()
        except Exception as exc:  # noqa: BLE001 - TUI must still open
            self.model = cfg.model
            self.model_error = str(exc)

        session = self.store.latest(cfg.workspace) if resume else None
        self.resumed = session is not None
        self.session: Session = session or self.store.new(cfg.workspace, self.model)
        self.history: list[dict] = self.session.messages
        self.goal_mode = bool(self.session.goal_mode)
        self.max_rounds = cfg.max_rounds
        self.verbose = cfg.verbose
        self.memory_on = cfg.memory_enabled and self.session.memory_enabled
        self.memory_block = ""
        self.show_thinking = True

        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.running = False
        self.quit = False

        self.transcript = Transcript()
        self.scroll = 0
        self._total = 0
        self._height = 20
        self._width = 100
        self.spinner = 0
        self.turn_tokens = 0
        self.reasoning_tokens = 0
        self.turn_started = 0.0
        self.last_ctx = 0
        self._live_text = ""
        self._reasoning_text = ""
        self.last_task = ""
        self.last_final = ""
        self.last_ok = False

        self.modal_visible = False
        self.modal_text = ""
        self.modal_default = "y"
        self.modal_callback = None
        self.pending_question: str | None = None
        self._question_event: threading.Event | None = None
        self._question_answer: dict[str, str] | None = None
        self.loop = None
        self.app: Application | None = None

        self.bus.subscribe(self._on_event)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.buffer = Buffer(
            multiline=False,
            completer=ElCompleter(self.cfg.workspace),
            complete_while_typing=True,
            history=FileHistory(str(self.cfg.home / "history")),
            accept_handler=self._accept,
            enable_history_search=True,
        )
        header = Window(FormattedTextControl(self._header), height=1, style="class:header", dont_extend_height=True)
        body = Window(self.transcript_control_ref(), wrap_lines=False, style="class:body")
        modal = ConditionalContainer(
            Window(
                FormattedTextControl(self._modal_body),
                height=3,
                style="class:modal",
                dont_extend_height=True,
            ),
            filter=Condition(lambda: self.modal_visible),
        )
        status = Window(FormattedTextControl(self._status), height=1, style="class:status", dont_extend_height=True)
        prompt = Window(
            BufferControl(buffer=self.buffer),
            height=1,
            style="class:input",
            get_line_prefix=self._prompt_prefix,
            dont_extend_height=True,
        )

        root = FloatContainer(
            content=HSplit([header, body, modal, status, prompt]),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8, scroll_offset=1),
                )
            ],
        )

        self._layout = Layout(root, focused_element=prompt)
        self._bindings = self._key_bindings()

    def _create_application(self) -> None:
        """Application creation needs a real console, so it is deferred to run()."""
        self.app = Application(
            layout=self._layout,
            key_bindings=self._bindings,
            style=_STYLE,
            full_screen=True,
            mouse_support=True,
        )

    def transcript_control_ref(self) -> TranscriptControl:
        if not hasattr(self, "_transcript_control"):
            self._transcript_control = TranscriptControl(self)
        return self._transcript_control

    # ------------------------------------------------------------------
    # rendering callbacks
    # ------------------------------------------------------------------
    def on_viewport(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def visible_fragments(self, width: int, height: int):
        lines = self.transcript.lines(width)
        chunk, self.scroll = slice_lines(lines, height, self.scroll)
        self._total = len(lines)
        out = []
        for index, fragments in enumerate(chunk):
            if index:
                out.append(("", "\n"))
            out.extend(fragments)
        return out

    @property
    def page(self) -> int:
        return max(1, self._height - 2)

    def scroll_up(self, amount: int) -> None:
        self.scroll += amount
        self.invalidate()

    def scroll_down(self, amount: int) -> None:
        self.scroll = max(0, self.scroll - amount)
        self.invalidate()

    def scroll_top(self) -> None:
        self.scroll = 10 ** 6
        self.invalidate()

    def scroll_bottom(self) -> None:
        self.scroll = 0
        self.invalidate()

    def _header(self):
        width = max(40, shutil.get_terminal_size().columns)
        text = f" ELAgent {__version__}   {self.model}   {self.cfg.workspace}   {self.session.id}"
        if len(text) > width:
            text = f" ELAgent {__version__}   {self.cfg.workspace}   {self.session.id}"
        if len(text) > width:
            text = f" ELAgent {__version__}   {self.session.id}"
        return [("class:header", text[:width].ljust(width))]

    def _status(self):
        width = max(40, shutil.get_terminal_size().columns)
        mode = "目标" if self.goal_mode else "标准"
        ctx = self.last_ctx or self.estimate_ctx()
        pct = int(100 * ctx / max(1, self.cfg.context_limit))
        rounds = "∞" if self.max_rounds < 0 else str(self.max_rounds)
        model = self.model if len(self.model) <= 24 else self.model[:23] + "…"
        total = self.session.usage.get("total", 0)
        fragments = [("class:status", f" {mode} ")]
        if self.running:
            elapsed = time.monotonic() - self.turn_started
            extra = f"  思考 {self.reasoning_tokens}" if self.show_thinking and self.reasoning_tokens else ""
            fragments.append(
                (
                    "class:status.running",
                    f" {SPINNER[self.spinner % len(SPINNER)]} 生成中 {elapsed:.1f}s · {self.turn_tokens} tok{extra} ",
                )
            )
        else:
            fragments.append(("class:status", " 就绪 "))
        tail = f"│ ctx {pct}% │ {total} tok │ 轮 {rounds} │ {model} "
        if self.scroll:
            tail = f"│ ↑{self.scroll} " + tail
        remaining = width - sum(_width(text) for _, text in fragments)
        fragments.append(("class:status", tail[: max(0, remaining)].ljust(max(0, remaining))))
        return fragments

    def _prompt_prefix(self, line_number: int, wrap_count: int):
        if self.pending_question is not None:
            return [("class:prompt", "答 › ")]
        if self.running:
            return [("class:dim", "│ ")]
        return [("class:prompt", "目标 › " if self.goal_mode else "› ")]

    def _modal_body(self):
        hint = "[Y] 是   [N] 否   [Esc] 默认"
        if self.modal_default == "n":
            hint = "[Y] 是   [N] 否   [Esc] 默认否"
        return [("class:modal", f" ! {self.modal_text}    {hint} ")]

    # ------------------------------------------------------------------
    # public helpers shared with the line-mode app
    # ------------------------------------------------------------------
    def write(self, text: str, style: str = "") -> None:
        self.transcript.add(Text(str(text), style=style) if style else Text(str(text)))
        self.invalidate()

    def write_renderable(self, renderable) -> None:
        self.transcript.add(renderable)
        self.invalidate()

    def clear_transcript(self) -> None:
        self.transcript.clear()
        self.invalidate()

    def notify(self, text: str, level: str = "dim") -> None:
        self.write(text, f"class:{level}" if level in ("dim", "warn", "err", "ok") else level)

    def request_quit(self) -> None:
        self.quit = True
        if self.app:
            self.app.exit()

    def estimate_ctx(self) -> int:
        try:
            from ..tokens import estimate_messages

            return estimate_messages(self.history)
        except Exception:
            return 0

    def invalidate(self) -> None:
        if self.app:
            self.app.invalidate()

    def post(self, fn, *args) -> None:
        if self.loop is None:
            try:
                fn(*args)
            except Exception:
                pass
            return
        try:
            self.loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # input handling
    # ------------------------------------------------------------------
    def _accept(self, buff: Buffer) -> bool:
        text = buff.text.strip()
        if self.pending_question is not None:
            self.write(f"  答：{text or '（空）'}", "dim")
            self.pending_question = None
            if self._question_answer is not None:
                self._question_answer["text"] = text
            if self._question_event is not None:
                self._question_event.set()
            self.invalidate()
            return False
        if not text:
            return False
        if self.running:
            self.write("任务执行中，Esc 中断后再提交。", "warn")
            return True
        self.transcript.add(user_block(text))
        if text.startswith("/"):
            if not commands.dispatch(self, text):
                self.write(f"未知命令: {text.split()[0]}（/help 查看）", "warn")
        elif text.startswith("!"):
            self.run_shell(text[1:].strip())
        else:
            self.start_task(self.expand_mentions(text).strip() or text)
        return False

    # ------------------------------------------------------------------
    # task execution
    # ------------------------------------------------------------------
    def handle_task(self, task: str) -> None:
        self.start_task(task)

    def start_task(self, task: str) -> None:
        task = (task or "").strip()
        if not task or self.running:
            return
        self.running = True
        self.cancel.clear()
        self._live_text = ""
        self._reasoning_text = ""
        self.turn_tokens = 0
        self.reasoning_tokens = 0
        self.turn_started = time.monotonic()
        self.transcript.clear_live()
        if not self.session.title:
            self.session.title = task[:40]
        self.maybe_recall(task)
        self.thread = threading.Thread(target=self._task_worker, args=(task,), daemon=True)
        self.thread.start()
        self.invalidate()

    def _task_worker(self, task: str) -> None:
        ok = False
        final = ""
        try:
            if self.goal_mode:
                self.cfg.max_rounds = self.max_rounds
                runner = GoalRunner(
                    self.cfg, self.llm, self.tools, self.bus, self.session, self.store,
                    self.memory, cancel=self.cancel, confirm=self.confirm,
                    ask_user=self.ask_user,
                )
                result = runner.run(task, memory_block=self.memory_block)
            else:
                agent = MainAgent(
                    self.cfg, self.llm, self.tools, self.bus, self.session,
                    cancel=self.cancel, confirm=self.confirm,
                    ask_user=self.ask_user,
                )
                result = agent.run(task, messages=self.history, memory_block=self.memory_block)
            ok, final = result.ok, result.final
        except Exception as exc:  # noqa: BLE001 - never crash the UI
            self.post(self.write, f"{type(exc).__name__}: {exc}", "err")
        finally:
            try:
                self.store.save(self.session)
            except Exception:
                pass
            self.post(self._task_done, ok, final, task)

    def _task_done(self, ok: bool, final: str, task: str) -> None:
        self.running = False
        self.transcript.clear_live()
        self._live_text = ""
        self._reasoning_text = ""
        # 记忆不再自动写入；由 /memory save 手动记录
        self.last_task = task
        self.last_final = final
        self.last_ok = ok
        self.invalidate()

    def record_memory(self) -> None:
        """`/memory save`: distill the last finished task into global.memory."""
        if self.running:
            self.write("  任务执行中，稍后再记录。", "warn")
            return
        if not getattr(self, "last_final", ""):
            self.write("  还没有已完成的任务可记录。", "dim")
            return
        if not self.last_ok:
            self.write("  最近一次任务未完成，通常不值得记录；仍要记录请再执行一次。", "dim")
            return
        self.write("  正在提炼经验…", "dim")
        threading.Thread(target=self._memory_worker, args=(self.last_task, self.last_final), daemon=True).start()

    def _memory_worker(self, task: str, final: str) -> None:
        try:
            bullets = self.memory.distill(self.llm, task, f"结论: {final[:1200]}")
            if bullets and self.memory.append(task, bullets):
                self.post(self.write, f"  已写入 {len(bullets)} 条经验 → {self.cfg.memory_file}", "ok")
            else:
                self.post(self.write, "  本次没有可复用的新经验。", "dim")
        except Exception as exc:  # noqa: BLE001
            self.post(self.write, f"  记忆写入失败: {exc}", "warn")

    def maybe_recall(self, task: str) -> None:
        self.memory_block = ""
        if not (self.cfg.memory_enabled and self.memory_on and self.memory.exists()):
            return
        entries = self.memory.recall(task, self.cfg.memory_max_entries)
        self.memory_block = self.memory.render(entries)
        if entries:
            self.write(f"  已载入 {len(entries)} 条历史经验", "dim")

    def request_cancel(self) -> None:
        if self.running:
            self.cancel.set()
            self.write("正在中断…", "warn")

    # ------------------------------------------------------------------
    # modal (y/n) support
    # ------------------------------------------------------------------
    def ask_modal(self, question: str, default: str, callback) -> None:
        self.modal_text = str(question).replace("\n", " ")[:280]
        self.modal_default = default
        self.modal_callback = callback
        self.modal_visible = True
        try:
            self.buffer.read_only = True
        except Exception:
            pass
        self.invalidate()

    def _answer_modal(self, value: str) -> None:
        if not self.modal_visible:
            return
        self.modal_visible = False
        try:
            self.buffer.read_only = False
        except Exception:
            pass
        callback, self.modal_callback = self.modal_callback, None
        self.invalidate()
        if callback:
            try:
                callback(value)
            except Exception:
                pass

    def confirm(self, question: str) -> bool:
        """Synchronous confirmation for the worker thread."""
        event = threading.Event()
        holder: dict[str, str] = {}

        def show() -> None:
            self.ask_modal(question, "n", lambda ans: (holder.update(v=ans), event.set()))

        self.post(show)
        while not event.wait(0.1):
            if self.cancel.is_set():
                self.post(self._answer_modal, "n")
                return False
        return str(holder.get("v", "n")).lower().startswith("y")

    def ask_user(self, question: str) -> str | None:
        """Blocking free-text prompt for the worker thread (ask tool)."""
        if self.quit:
            return None
        event = threading.Event()
        holder: dict[str, str] = {}

        def show() -> None:
            self.pending_question = str(question)
            self._question_event = event
            self._question_answer = holder
            self.write(f"  ? {question}", "warn")
            try:
                self.buffer.reset()
            except Exception:
                pass
            self.invalidate()

        self.post(show)
        while not event.wait(0.1):
            if self.cancel.is_set():
                self.post(self._clear_question)
                return None
        return holder.get("text")

    def _clear_question(self) -> None:
        self.pending_question = None
        self._question_event = None
        self._question_answer = None
        self.invalidate()

    # ------------------------------------------------------------------
    # events from the agent
    # ------------------------------------------------------------------
    def _on_event(self, event: Event) -> None:
        self.post(self._handle_event, event)

    def _handle_event(self, event: Event) -> None:
        kind, data = event.kind, event.data
        if kind == "turn_start":
            self.turn_tokens = 0
            self.reasoning_tokens = 0
            self.turn_started = time.monotonic()
        elif kind == "delta":
            self.turn_tokens += max(1, len(data.get("text", "")) // 3)
            self._live_text += data.get("text", "")
            self.transcript.set_live(self._live_text)
        elif kind == "reasoning":
            self.reasoning_tokens += max(1, len(data.get("text", "")) // 3)
            if self.show_thinking:
                self._reasoning_text += data.get("text", "")
                self.transcript.set_reasoning(self._reasoning_text)
        elif kind == "turn_end":
            self.last_ctx = int(data.get("ctx_tokens") or 0)
            self.transcript.clear_live()
            self._live_text = ""
            reasoning = str(data.get("reasoning") or "").strip()
            if self.show_thinking and reasoning:
                self.write_renderable(thinking_block(reasoning))
            self._reasoning_text = ""
            if self.verbose and str(data.get("text", "")).strip():
                self.write(str(data.get("text", "")).strip()[:1200], "dim")
        elif kind == "tool_start":
            self.transcript.clear_live()
            self._live_text = ""
            self._reasoning_text = ""
            self.write_renderable(tool_start_block(data.get("name", "?"), self._tool_detail(data.get("args") or {})))
        elif kind == "tool_end":
            self.write_renderable(
                tool_end_block(bool(data.get("ok")), str(data.get("display") or ""), int(data.get("ms") or 0))
            )
            if self.verbose:
                preview = "\n".join(str(data.get("content", "")).splitlines()[:12])
                if preview.strip():
                    self.write(preview, "dim")
        elif kind == "notice":
            self.write_renderable(notice_block(data.get("text", ""), data.get("level", "warn")))
        elif kind == "error":
            self.write_renderable(error_block(data.get("text", "")))
        elif kind == "compact":
            self.write(f"  ~ 上下文压缩 {data.get('before')} → {data.get('after')} tok", "dim")
        elif kind == "round_start":
            total = data.get("max", -1)
            label = f"第 {data.get('n')} 轮" + ("" if total and total > 0 else " / 无限")
            self.write_renderable(rule_block(label))
        elif kind == "verdict":
            self.write_renderable(verdict_block(int(data.get("status", -1)), data.get("reason", "")))
        elif kind == "goal_done":
            if data.get("ok"):
                self.write(f"  ✓ 目标达成，共 {data.get('rounds')} 轮", "ok")
            else:
                self.write(f"  x 目标未达成，共 {data.get('rounds')} 轮", "warn")
        elif kind == "final":
            self.transcript.clear_live()
            self._live_text = ""
            self.write_renderable(final_block(str(data.get("text", "")), bool(data.get("degraded"))))
            meta = f"  {data.get('steps', 0)} 步"
            if data.get("elapsed"):
                meta += f" · {data['elapsed']:.1f}s"
            self.write(meta, "dim")
        self.invalidate()

    @staticmethod
    def _tool_detail(args: dict) -> str:
        import json as _json

        if args.get("command"):
            return str(args["command"])
        if args.get("path"):
            detail = str(args["path"])
            if args.get("offset"):
                detail += f" from {args['offset']}"
            return detail
        try:
            detail = _json.dumps(args, ensure_ascii=False)
        except Exception:
            detail = str(args)
        return detail if len(detail) <= 100 else detail[:99] + "…"

    # ------------------------------------------------------------------
    # session helpers
    # ------------------------------------------------------------------
    def new_session(self) -> None:
        self.store.save(self.session)
        self.session = self.store.new(self.cfg.workspace, self.model)
        self.history = self.session.messages
        self.goal_mode = False
        self.memory_block = ""
        self.last_ctx = 0
        self.write(f"  新会话: {self.session.id}", "dim")

    def resume_session(self, arg: str) -> None:
        rows = self.store.list(limit=15)
        if not rows:
            self.write("  暂无历史会话。", "dim")
            return
        if not arg:
            self.write_renderable(self._sessions_table())
            self.write("  用 /resume <序号|ID> 恢复", "dim")
            return
        if arg.isdigit() and 1 <= int(arg) <= len(rows):
            arg = rows[int(arg) - 1]["id"]
        matches = [row for row in rows if row["id"].startswith(arg)]
        if not matches:
            self.write(f"  未找到会话: {arg}", "warn")
            return
        session = self.store.load(matches[0]["id"])
        if not session:
            self.write("  会话文件损坏，无法恢复。", "warn")
            return
        self.load_session(session)
        self.write(f"  已恢复会话 {session.id}（{len(session.messages)} 条消息）", "dim")

    def load_session(self, session: Session) -> None:
        self.store.save(self.session)
        self.session = session
        self.history = session.messages
        self.goal_mode = bool(session.goal_mode)
        self.memory_block = ""
        self.last_ctx = 0
        self.invalidate()

    def sessions_table(self):
        from rich.table import Table

        rows = self.store.list(limit=15)
        table = Table(box=None, show_header=True, header_style="bold #565f89", padding=(0, 2))
        table.add_column("#", style="dim", width=3)
        table.add_column("会话 ID")
        table.add_column("标题")
        table.add_column("消息", justify="right")
        table.add_column("更新时间", style="dim")
        for index, row in enumerate(rows, 1):
            table.add_row(str(index), row["id"], (row["title"] or "")[:36], str(row["messages"]), row["updated"])
        return table

    def compact_context(self) -> None:
        if self.running:
            self.write("  任务执行中，稍后再压缩。", "warn")
            return
        self.write("  正在压缩…", "dim")

        def worker() -> None:
            try:
                manager = ContextManager(self.history, self.cfg, self.llm, self.bus)
                changed = manager.compact(force=True)
                self.post(self.write, "  上下文已压缩。" if changed else "  当前上下文无需压缩。", "dim")
            except Exception as exc:  # noqa: BLE001
                self.post(self.write, f"  压缩失败: {exc}", "warn")

        threading.Thread(target=worker, daemon=True).start()

    def run_shell(self, command: str) -> None:
        if not command or self.running:
            return
        self.running = True
        self.cancel.clear()
        self.invalidate()

        def worker() -> None:
            try:
                ctx = ToolContext(Path(self.cfg.workspace), self.cfg, self.session.id, self.cancel, self.confirm)
                result = BashTool().run({"command": command, "timeout": 120}, ctx)
                border = OK if result.ok else ERR
                self.post(
                    self.write_renderable,
                    Panel(Text(result.content[:4000], style="white"), border_style=border, padding=(0, 1)),
                )
                self.history.append({"role": "user", "content": f"[shell] $ {command}\n{result.content[:3000]}"})
            finally:
                self.post(self._shell_done)

        threading.Thread(target=worker, daemon=True).start()

    def _shell_done(self) -> None:
        self.running = False
        self.invalidate()

    def expand_mentions(self, task: str) -> str:
        import re

        def replace(match: "re.Match[str]") -> str:
            raw = match.group(1).rstrip("/")
            path = Path(raw)
            if not path.is_absolute():
                path = Path(self.cfg.workspace) / raw
            if not path.exists() or not path.is_file():
                return match.group(0)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")[:4000]
            except Exception:
                return match.group(0)
            return f"\n[附件 {raw}]\n```\n{content}\n```\n"

        return re.sub(r"@([\w./\\\-]+)", replace, task)

    # ------------------------------------------------------------------
    # key bindings
    # ------------------------------------------------------------------
    def _key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        modal = Condition(lambda: self.modal_visible)
        running = Condition(lambda: self.running)

        @kb.add("enter", filter=modal, eager=True)
        def _(event) -> None:
            self._answer_modal(self.modal_default)

        @kb.add("escape", filter=modal, eager=True)
        def _(event) -> None:
            self._answer_modal("n")

        @kb.add("y", filter=modal, eager=True)
        def _(event) -> None:
            self._answer_modal("y")

        @kb.add("n", filter=modal, eager=True)
        def _(event) -> None:
            self._answer_modal("n")

        @kb.add("enter", filter=~has_completions & ~modal)
        def _(event) -> None:
            event.current_buffer.validate_and_handle()

        @kb.add("escape", filter=running & ~modal)
        def _(event) -> None:
            self.request_cancel()

        @kb.add("c-c")
        def _(event) -> None:
            if self.modal_visible:
                self._answer_modal("n")
            elif self.running:
                self.request_cancel()
            elif event.current_buffer.text:
                event.current_buffer.reset()
            else:
                self.write("按 Ctrl+D 退出", "dim")

        @kb.add("c-d")
        def _(event) -> None:
            if not event.current_buffer.text:
                self.request_quit()

        @kb.add("pageup")
        def _(event) -> None:
            self.scroll_up(self.page)

        @kb.add("pagedown")
        def _(event) -> None:
            self.scroll_down(self.page)

        @kb.add("c-home")
        def _(event) -> None:
            self.scroll_top()

        @kb.add("c-end")
        def _(event) -> None:
            self.scroll_bottom()

        @kb.add(Keys.ScrollUp)
        def _(event) -> None:
            self.scroll_up(3)

        @kb.add(Keys.ScrollDown)
        def _(event) -> None:
            self.scroll_down(3)

        return kb

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _pre_run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.write_renderable(
            welcome_block(self.model, self.cfg.workspace, self.session.id, self.resumed, self.model_error)
        )
        self._replay_history()
        self.app.create_background_task(self._ticker())
        # 记忆不再主动询问或写入；参考 /memory on|off 与 /memory save
        stats = self.memory.stats()
        if stats["entries"] and self.memory_on:
            self.write(f"  将参考历史经验（{stats['entries']} 条，/memory off 关闭 · /memory save 记录）", "dim")
        self.invalidate()

    def _replay_history(self, limit: int = 10) -> None:
        interesting = [
            message
            for message in self.history
            if message.get("role") in ("user", "assistant")
            and not str(message.get("content", "")).startswith("OBS[")
            and not str(message.get("content", "")).startswith("[shell]")
        ]
        if not interesting:
            return
        for message in interesting[-limit:]:
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if message.get("role") == "user":
                self.write_renderable(user_block(content[:300]))
            else:
                self.write_renderable(assistant_block(content[:600]))
        self.write("  —— 以上为本次恢复的最近历史 ——", "dim")

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(0.12)
            if self.running:
                self.spinner = (self.spinner + 1) % len(SPINNER)
                self.app.invalidate()

    def run(self) -> None:
        self._create_application()
        try:
            self.app.run(pre_run=self._pre_run)
        finally:
            try:
                self.store.save(self.session)
            except Exception:
                pass
            self.llm.close()


def _width(text: str) -> int:
    try:
        from prompt_toolkit.utils import get_cwidth

        return get_cwidth(text)
    except Exception:
        return len(text)
