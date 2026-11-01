"""Interactive TUI: prompt_toolkit input + rich transcript."""

from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..agent import GoalRunner, MainAgent
from ..config import Config
from ..context import ContextManager
from ..events import EventBus
from ..llm import LLMClient
from ..memory import GlobalMemory
from ..session import Session, SessionStore
from ..tools import ToolContext, build_registry
from ..tools.shell import BashTool
from ..tokens import estimate_messages
from .commands import COMMANDS, dispatch
from .render import DIM, ERR, OK, WARN, Renderer

_MENTION_RE = re.compile(r"@([\w./\\\-]+)")

_STYLE = Style.from_dict(
    {
        "prompt": "#7aa2f7 bold",
        "bottom-toolbar": "bg:#1f2335 #a9b1d6",
        "completion-menu.completion": "bg:#1f2335 #c0caf5",
        "completion-menu.completion.current": "bg:#7aa2f7 #1a1b26",
        "scrollbar.background": "bg:#1f2335",
        "scrollbar.button": "bg:#7aa2f7",
    }
)


class ElCompleter(Completer):
    def __init__(self, workspace: str) -> None:
        self.workspace = Path(workspace)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text or " " in text:
            return
        if text.startswith("/"):
            for command, _ in COMMANDS:
                if command.startswith(text):
                    yield Completion(command, start_position=-len(text))
        elif text.startswith("@"):
            fragment = text[1:]
            try:
                matches = sorted(self.workspace.glob(fragment + "*"))[:20]
            except Exception:
                return
            for path in matches:
                suffix = "/" if path.is_dir() else ""
                yield Completion("@" + path.name + suffix, start_position=-len(text))


class App:
    def __init__(self, cfg: Config, resume: bool = False) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self.console = Console(highlight=False)
        self.bus = EventBus()
        self.renderer = Renderer(self.console, verbose=cfg.verbose)
        self.bus.subscribe(self.renderer.handle)

        self.llm = LLMClient(cfg)
        self.tools = build_registry()
        self.store = SessionStore(cfg.home)
        self.memory = GlobalMemory(cfg.memory_file)

        self.model_error = ""
        try:
            self.model = self.llm.resolve_model()
        except Exception as exc:  # noqa: BLE001 - the TUI must still open
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
        self.cancel = threading.Event()
        self.quit = False
        self.last_task = ""
        self.last_final = ""
        self.last_ok = False

        self.ps: PromptSession | None = None
        try:
            self.ps = PromptSession(
                history=FileHistory(str(cfg.home / "history")),
                completer=ElCompleter(cfg.workspace),
                complete_while_typing=True,
                style=_STYLE,
                bottom_toolbar=self.status_line,
                enable_history_search=True,
                reserve_space_for_menu=6,
            )
        except Exception:
            # No real Windows console (piped input, some terminals). Fall back to
            # a plain line reader so the agent stays usable everywhere.
            self.ps = None

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.renderer.header(self.model, self.cfg.workspace, self.session.id, self.resumed)
        if self.model_error:
            self.console.print(
                Panel(
                    Text(f"无法连接模型服务: {self.model_error}\n请确认 {self.cfg.base_url} 已启动，或 /model 切换。", style=ERR),
                    border_style=ERR,
                )
            )
        stats = self.memory.stats()
        if stats["entries"]:
            self.console.print(
                Text(f"  记忆: {stats['entries']} 条 · {self.cfg.memory_file}", style=DIM)
            )
        if self.history:
            self.console.print(Text(f"  已恢复 {len(self.history)} 条历史消息（/new 开始新会话）", style=DIM))

        while not self.quit:
            try:
                line = self.read_line("目标 › " if self.goal_mode else "› ")
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            line = (line or "").strip()
            if not line:
                continue
            if line.startswith("/"):
                if not dispatch(self, line):
                    self.console.print(f"  未知命令: {line.split()[0]}（/help 查看）", style=WARN)
                continue
            if line.startswith("!"):
                self.run_shell(line[1:].strip())
                continue
            self.handle_task(line)

        self.store.save(self.session)
        self.llm.close()
        self.console.print(Text("  已退出。", style=DIM))

    # ------------------------------------------------------------------
    def status_line(self) -> str:
        mode = "目标" if self.goal_mode else "标准"
        ctx = self.renderer.ctx_tokens or self.estimate_ctx()
        pct = int(100 * ctx / max(1, self.cfg.context_limit))
        rounds = "∞" if self.max_rounds < 0 else str(self.max_rounds)
        model = self.model if len(self.model) <= 26 else self.model[:25] + "…"
        total = self.session.usage.get("total", 0)
        return f" {mode} │ ctx {pct}% │ {total} tok │ 轮 {rounds} │ {model} "

    @property
    def last_ctx(self) -> int:
        return self.renderer.ctx_tokens

    @property
    def show_thinking(self) -> bool:
        return self.renderer.show_reasoning

    @show_thinking.setter
    def show_thinking(self, value: bool) -> None:
        self.renderer.show_reasoning = bool(value)

    # shared UI interface (see commands.py) ---------------------------
    def write(self, text: str, style: str = "") -> None:
        self.console.print(Text(str(text), style=style) if style else str(text))

    def write_renderable(self, renderable) -> None:
        self.console.print(renderable)

    def clear_transcript(self) -> None:
        self.console.clear()

    def request_quit(self) -> None:
        self.quit = True

    def set_model(self, name: str) -> None:
        self.cfg.model = name
        self.llm.set_model(name)
        self.model = name

    def sessions_table(self):
        from rich.table import Table

        rows = self.store.list(limit=15)
        table = Table(box=None, show_header=True, header_style="bold dim", padding=(0, 2))
        table.add_column("#", style="dim", width=3)
        table.add_column("会话 ID")
        table.add_column("标题")
        table.add_column("消息", justify="right")
        table.add_column("更新时间", style="dim")
        for index, row in enumerate(rows, 1):
            table.add_row(str(index), row["id"], (row["title"] or "")[:36], str(row["messages"]), row["updated"])
        return table

    def resume_session(self, arg: str) -> None:
        rows = self.store.list(limit=15)
        if not rows:
            self.console.print("  暂无历史会话。", style="dim")
            return
        if not arg:
            self.write_renderable(self.sessions_table())
            self.console.print("  用 /resume <序号|ID> 恢复", style="dim")
            return
        if arg.isdigit() and 1 <= int(arg) <= len(rows):
            arg = rows[int(arg) - 1]["id"]
        matches = [row for row in rows if row["id"].startswith(arg)]
        if not matches:
            self.console.print(f"  未找到会话: {arg}", style=WARN)
            return
        session = self.store.load(matches[0]["id"])
        if not session:
            self.console.print("  会话文件损坏，无法恢复。", style=WARN)
            return
        self.load_session(session)
        self.console.print(f"  已恢复会话 {session.id}（{len(session.messages)} 条消息）", style="dim")

    def compact_context(self) -> None:
        if self.compact_now():
            self.console.print("  上下文已压缩。", style="dim")
        else:
            self.console.print("  当前上下文无需压缩。", style="dim")

    def estimate_ctx(self) -> int:
        try:
            return estimate_messages(self.history)
        except Exception:
            return 0

    def read_line(self, prompt: str) -> str:
        if self.ps is not None:
            return self.ps.prompt(prompt)
        sys.stdout.write(prompt)
        sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line.rstrip("\r\n")

    def ask(self, prompt: str, default: str = "") -> str:
        try:
            if self.ps is not None:
                return self.ps.prompt(prompt, default=default)
            answer = self.read_line(prompt)
            return answer if answer.strip() else default
        except (KeyboardInterrupt, EOFError):
            return default

    def confirm(self, question: str) -> bool:
        answer = self.ask(question + " [y/N] ", default="n")
        return answer.strip().lower() in ("y", "yes", "是")

    def ask_user(self, question: str) -> str | None:
        """Free-text prompt used by the ask tool (line mode is synchronous)."""
        try:
            if self.ps is not None:
                return self.ps.prompt(str(question) + " ")
            return self.read_line(str(question) + " ")
        except (KeyboardInterrupt, EOFError):
            return None

    # ------------------------------------------------------------------
    def handle_task(self, task: str) -> None:
        task = self.expand_mentions(task).strip()
        if not task:
            return
        self.last_task = task
        if not self.session.title:
            self.session.title = task[:40]
        self.cancel.clear()
        self.maybe_recall(task)
        started = time.monotonic()
        ok = False
        final = ""
        files: list[str] = []

        try:
            if self.goal_mode:
                self.cfg.max_rounds = self.max_rounds
                runner = GoalRunner(
                    self.cfg, self.llm, self.tools, self.bus, self.session, self.store,
                    self.memory, cancel=self.cancel, confirm=self.confirm,
                    ask_user=self.ask_user,
                )
                result = runner.run(task, memory_block=self.memory_block)
                ok, final = result.ok, result.final
            else:
                agent = MainAgent(
                    self.cfg, self.llm, self.tools, self.bus, self.session,
                    cancel=self.cancel, confirm=self.confirm,
                    ask_user=self.ask_user,
                )
                result = agent.run(task, messages=self.history, memory_block=self.memory_block)
                ok, final = result.ok, result.final
                files = result.changed_files
        except KeyboardInterrupt:
            self.cancel.set()
            self.console.print(Text("  已中断。", style=WARN))
        except Exception as exc:  # noqa: BLE001 - a UI crash is worse than a failed task
            self.console.print(Panel(Text(f"{type(exc).__name__}: {exc}", style=ERR), border_style=ERR))
        finally:
            self.renderer.stop()
            try:
                self.store.save(self.session)
            except Exception:
                pass

        elapsed = time.monotonic() - started
        if files:
            self.console.print(Text(f"  修改: {', '.join(files[:8])}", style=DIM))
        self.console.print(Text(f"  用时 {elapsed:.1f}s · {self.session.usage.get('total', 0)} tok 累计", style=DIM))

        # 记忆不再自动写入；用 /memory save 手动记录
        self.last_task = task
        self.last_final = final
        self.last_ok = ok

    # ------------------------------------------------------------------
    def maybe_recall(self, task: str) -> None:
        if not (self.cfg.memory_enabled and self.memory_on and self.memory.exists()):
            self.memory_block = ""
            return
        entries = self.memory.recall(task, self.cfg.memory_max_entries)
        self.memory_block = self.memory.render(entries)
        if entries:
            self.console.print(Text(f"  已载入 {len(entries)} 条历史经验", style=DIM))

    def record_memory(self) -> None:
        """`/memory save`: 记录最近一次已完成任务的经验。"""
        if self.running:
            self.console.print(Text("  任务执行中，稍后再记录。", style=WARN))
            return
        if not getattr(self, "last_final", ""):
            self.console.print(Text("  还没有已完成的任务可记录。", style=DIM))
            return
        if not self.last_ok:
            self.console.print(Text("  最近一次任务未完成，通常不值得记录；仍要记录请再执行一次。", style=DIM))
            return
        self.write_memory(self.last_task, self.last_final)

    def write_memory(self, task: str, final: str) -> None:
        self.console.print(Text("  正在提炼经验写入 global.memory…", style=DIM))
        summary = f"结论: {final[:1200]}"
        try:
            bullets = self.memory.distill(self.llm, task, summary)
            if bullets and self.memory.append(task, bullets):
                self.console.print(Text(f"  已写入 {len(bullets)} 条经验 → {self.cfg.memory_file}", style=OK))
            else:
                self.console.print(Text("  本次没有可复用的新经验。", style=DIM))
        except Exception as exc:  # noqa: BLE001
            self.console.print(Text(f"  记忆写入失败: {exc}", style=DIM))

    # ------------------------------------------------------------------
    def expand_mentions(self, task: str) -> str:
        def replace(match: re.Match) -> str:
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

        return _MENTION_RE.sub(replace, task)

    def run_shell(self, command: str) -> None:
        if not command:
            return
        ctx = ToolContext(Path(self.cfg.workspace), self.cfg, self.session.id, self.cancel, self.confirm)
        result = BashTool().run({"command": command, "timeout": 120}, ctx)
        border = OK if result.ok else ERR
        self.console.print(Panel(Text(result.content[:4000], style="white"), border_style=border, padding=(0, 1)))
        self.history.append({"role": "user", "content": f"[shell] $ {command}\n{result.content[:3000]}"})

    # ------------------------------------------------------------------
    def new_session(self) -> None:
        self.store.save(self.session)
        self.session = self.store.new(self.cfg.workspace, self.model)
        self.history = self.session.messages
        self.goal_mode = False
        self.memory_block = ""
        self.renderer.ctx_tokens = 0
        self.console.print(Text(f"  新会话: {self.session.id}", style=DIM))

    def load_session(self, session: Session) -> None:
        self.store.save(self.session)
        self.session = session
        self.history = session.messages
        self.goal_mode = bool(session.goal_mode)
        self.memory_block = ""
        self.renderer.ctx_tokens = self.estimate_ctx()

    def compact_now(self) -> bool:
        ctx = ContextManager(self.history, self.cfg, self.llm, self.bus)
        return ctx.compact(force=True)
