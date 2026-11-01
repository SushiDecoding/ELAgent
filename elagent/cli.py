"""Command line entry point: `ela` opens the TUI, `ela -p "task"` runs once."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

from . import __version__
from .agent import GoalRunner, MainAgent
from .config import Config
from .events import EventBus
from .llm import LLMClient
from .memory import GlobalMemory
from .prompts import budget_report
from .session import SessionStore
from .tools import build_registry
from .tokens import estimate_messages

BANNER = f"ELAgent {__version__}"


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ela", description="ELAgent - local long-horizon agent")
    parser.add_argument("task", nargs="*", help="一次性任务（省略则进入 TUI）")
    parser.add_argument("-p", "--print", action="store_true", dest="print_mode", help="一次性执行并输出结果")
    parser.add_argument("--goal", action="store_true", help="以目标模式执行（主 Agent + 评审循环）")
    parser.add_argument("--rounds", type=int, default=None, help="目标模式最大轮数，-1 为无限")
    parser.add_argument("--resume", action="store_true", help="恢复当前目录最近的会话（默认每次新建）")
    parser.add_argument("--new", action="store_true", help="新建会话（默认行为，保留兼容）")
    parser.add_argument("--model", default=None, help="覆盖模型名")
    parser.add_argument("--base-url", default=None, help="覆盖 OpenAI 兼容端点")
    parser.add_argument("--shell", default=None, choices=["auto", "bash", "cmd", "powershell"])
    parser.add_argument("--yolo", action="store_true", help="跳过危险命令确认（谨慎）")
    parser.add_argument("--verbose", action="store_true", help="显示工具完整输出")
    parser.add_argument("--no-memory", action="store_true", help="本次不读写 global.memory")
    parser.add_argument("--doctor", action="store_true", help="检查端点、模型与提示词预算")
    parser.add_argument("--tools", action="store_true", help="列出工具")
    parser.add_argument("--version", action="version", version=BANNER)
    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8()
    args = build_parser().parse_args(argv)

    cfg = Config.load()
    if args.model:
        cfg.model = args.model
    if args.base_url:
        cfg.base_url = args.base_url
    if args.shell:
        cfg.shell = args.shell
    if args.yolo:
        cfg.yolo = True
    if args.verbose:
        cfg.verbose = True
    if args.rounds is not None:
        cfg.max_rounds = args.rounds
    if args.no_memory:
        cfg.memory_enabled = False
        cfg.memory_on_complete = False

    if args.doctor:
        return doctor(cfg)
    if args.tools:
        print(build_registry().describe_full())
        return 0

    task = " ".join(args.task).strip()
    if task:
        return run_once(cfg, task, goal=args.goal, resume=args.resume)
    if args.print_mode:
        print("缺少任务内容。用法: ela -p \"任务\"", file=sys.stderr)
        return 2

    try:
        _launch_tui(cfg, resume=args.resume and not args.new)
    except KeyboardInterrupt:
        print("\n已退出。")
    return 0


def _launch_tui(cfg: Config, resume: bool) -> None:
    """Prefer the full-screen UI; fall back to the line UI without a console."""
    if sys.stdout.isatty():
        try:
            from .ui.fullscreen import FullApp

            FullApp(cfg, resume=resume).run()
            return
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            recoverable = name in ("NoConsoleScreenBufferError",) or "console" in str(exc).lower()
            if not recoverable:
                raise
            print(f"全屏界面不可用（{name}），退回行式界面。", file=sys.stderr)
    from .ui.tui import App

    App(cfg, resume=resume).run()


# ----------------------------------------------------------------------
def run_once(cfg: Config, task: str, *, goal: bool, resume: bool) -> int:
    from rich.console import Console

    from .ui.render import Renderer

    console = Console(highlight=False)
    bus = EventBus()
    renderer = Renderer(console, verbose=cfg.verbose)
    bus.subscribe(renderer.handle)

    llm = LLMClient(cfg)
    tools = build_registry()
    store = SessionStore(cfg.home)
    memory = GlobalMemory(cfg.memory_file)
    session = (store.latest(cfg.workspace) if resume else None) or store.new(cfg.workspace, cfg.model)
    if not session.title:
        session.title = task[:40]

    try:
        model = llm.resolve_model()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]无法连接模型服务: {exc}[/red]")
        return 3
    session.model = model
    renderer.header(model, cfg.workspace, session.id, resume)

    memory_block = ""
    if cfg.memory_enabled and memory.exists():
        entries = memory.recall(task, cfg.memory_max_entries)
        memory_block = memory.render(entries)
        if entries:
            console.print(f"[dim]载入 {len(entries)} 条历史经验[/dim]")

    def deny(question: str) -> bool:
        console.print("[yellow]非交互模式拒绝危险命令（如需执行请加 --yolo）[/yellow]")
        return False

    started = time.monotonic()
    try:
        if goal:
            runner = GoalRunner(cfg, llm, tools, bus, session, store, memory, confirm=deny)
            result = runner.run(task, memory_block=memory_block)
            ok, final = result.ok, result.final
        else:
            agent = MainAgent(cfg, llm, tools, bus, session, confirm=deny)
            result = agent.run(task, messages=session.messages, memory_block=memory_block)
            ok, final = result.ok, result.final
    except KeyboardInterrupt:
        console.print("[yellow]已中断[/yellow]")
        return 130
    finally:
        renderer.stop()
        store.save(session)

    elapsed = time.monotonic() - started
    console.print(
        f"[dim]{'完成' if ok else '未完成'} · {elapsed:.1f}s · {session.usage.get('total', 0)} tok[/dim]"
    )

    if ok and final and cfg.memory_on_complete:
        bullets = memory.distill(llm, task, f"结论: {final[:1200]}")
        if bullets and memory.append(task, bullets):
            console.print(f"[dim]已写入 {len(bullets)} 条经验 → {cfg.memory_file}[/dim]")

    store.save_markdown(session, Path(cfg.workspace) / f"elagent-{session.id}.md")
    llm.close()
    return 0 if ok else 1


# ----------------------------------------------------------------------
def doctor(cfg: Config) -> int:
    from rich.console import Console
    from rich.table import Table

    console = Console(highlight=False)
    console.print(f"[bold]{BANNER}[/bold] · Python {platform.python_version()} · {platform.system()}")

    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    table.add_row("端点", cfg.base_url)
    table.add_row(
        "生成参数",
        (
            f"由 llama.cpp 服务端决定（本地流上限 {cfg.max_stream_chars} 字符）"
            if not cfg.send_generation_params
            else "ELAgent 覆盖 temperature/top_p/max_tokens"
        ),
    )

    llm = LLMClient(cfg)
    try:
        started = time.monotonic()
        info = llm.ping()
        latency = (time.monotonic() - started) * 1000
        model = llm.resolve_model()
        table.add_row("服务", f"[green]OK[/green] {info['status']} · {latency:.0f}ms")
        table.add_row("模型", model)
        raw = info.get("body", {})
        items = raw.get("data") or raw.get("models") or []
        if items:
            meta = (items[0] or {}).get("meta") or {}
            if meta.get("n_ctx"):
                table.add_row("原生 ctx", str(meta["n_ctx"]))
    except Exception as exc:  # noqa: BLE001
        table.add_row("服务", f"[red]FAIL[/red] {exc}")
    console.print(table)

    budget = budget_report()
    worst = max(budget.values())
    color = "green" if worst <= 1000 else "yellow"
    console.print(
        f"提示词预算  system {budget['system']} · judge {budget['judge']} · "
        f"compact {budget['compact']} · memory {budget['memory']}  "
        f"[{color}]（上限 1000）[/{color}]"
    )
    if not cfg.send_generation_params:
        console.print(
            "[dim]提示: 生成长度与采样由 llama.cpp 决定；回复过长请在服务端设置 --n-predict。[/dim]"
        )

    tools = build_registry()
    console.print(f"工具 {len(tools.names())}: {', '.join(tools.names())}")
    console.print(f"会话目录  {cfg.sessions_dir}")
    console.print(f"记忆文件  {cfg.memory_file}")

    store = SessionStore(cfg.home)
    session = store.latest(cfg.workspace)
    if session:
        console.print(f"可恢复会话 {session.id} · {len(session.messages)} 条消息 · 约 {estimate_messages(session.messages)} tok")
    llm.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
