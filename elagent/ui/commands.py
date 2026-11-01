"""Slash command system, shared by the full-screen and line-mode UIs.

Commands never touch the terminal directly: they call `app.write` /
`app.write_renderable`, which each UI implements (Rich transcript block in
full-screen mode, direct console print in line mode).
"""

from __future__ import annotations

from pathlib import Path

from .transcript import command_table, kv_table

COMMANDS: list[tuple[str, str]] = [
    ("/help", "显示命令与快捷键"),
    ("/goal [任务]", "进入目标模式；/goal off 关闭"),
    ("/rounds N", "最大轮数，-1 为无限（默认）"),
    ("/new", "新建会话"),
    ("/resume [id]", "恢复历史会话"),
    ("/sessions", "列出最近会话"),
    ("/compact", "压缩上下文"),
    ("/memory [save|on|off|show]", "save 记录上次任务经验；on/off 参考开关；show 查看"),
    ("/model [name]", "查看或切换模型"),
    ("/status", "模型、会话、上下文与用量"),
    ("/tools", "列出可用工具"),
    ("/verbose", "切换工具输出详情"),
    ("/thinking", "展开/收起思考内容（默认展开）"),
    ("/export [path]", "导出 Markdown 会话"),
    ("/clear", "清空对话区"),
    ("/quit", "退出"),
]


def dispatch(app, line: str) -> bool:
    name, _, arg = line.partition(" ")
    name = name.strip().lower()
    arg = arg.strip()
    handler = _HANDLERS.get(name)
    if handler is None:
        return False
    handler(app, arg)
    return True


# ----------------------------------------------------------------------
def _help(app, arg: str) -> None:
    app.write_renderable(command_table(COMMANDS))
    app.write(
        "  !命令   执行 shell 并把输出加入上下文\n"
        "  @路径   把文件内容附加到任务\n"
        "  PageUp/PageDown 滚动对话 · Ctrl+Home/End 到顶部/底部\n"
        "  Ctrl+C  中断任务 · Ctrl+D 退出",
        "dim",
    )


def _goal(app, arg: str) -> None:
    if arg.lower() in ("off", "0", "false", "关闭"):
        app.goal_mode = False
        app.session.goal_mode = False
        app.write("  目标模式已关闭，回到标准模式。", "dim")
        return
    app.goal_mode = True
    app.session.goal_mode = True
    if arg:
        app.write("  目标模式已开启。", "dim")
        app.handle_task(arg)
    else:
        app.write("  目标模式已开启。请输入任务，我会循环执行并由子 Agent 评审。", "dim")


def _rounds(app, arg: str) -> None:
    if not arg:
        app.write(f"  当前最大轮数: {app.max_rounds}（-1 为无限）", "dim")
        return
    try:
        value = -1 if arg.lower() in ("inf", "infinite", "无限") else int(arg)
    except ValueError:
        app.write("  用法: /rounds 5 或 /rounds -1", "warn")
        return
    app.max_rounds = value
    app.cfg.max_rounds = value
    app.write(f"  最大轮数已设为 {value}", "dim")


def _new(app, arg: str) -> None:
    app.new_session()


def _sessions(app, arg: str) -> None:
    app.write_renderable(app.sessions_table())


def _resume(app, arg: str) -> None:
    app.resume_session(arg)


def _compact(app, arg: str) -> None:
    app.compact_context()


def _memory(app, arg: str) -> None:
    action = (arg or "show").lower()
    if action in ("save", "记录", "write", "学习"):
        app.record_memory()
        return
    if action in ("on", "off"):
        app.memory_on = action == "on"
        app.cfg.memory_enabled = app.memory_on
        app.session.memory_enabled = app.memory_on
        if not app.memory_on:
            app.memory_block = ""
        app.write(f"  全局记忆参考已{'开启' if app.memory_on else '关闭'}（记录请用 /memory save）", "dim")
        return
    stats = app.memory.stats()
    if not stats["entries"]:
        app.write("  global.memory 为空。完成任务后用 /memory save 记录经验。", "dim")
        return
    app.write(f"  文件: {app.cfg.memory_file}", "dim")
    app.write(f"  条目: {stats['entries']} · 约 {stats['tokens']} tok · {stats['bytes']} 字节", "dim")
    for entry in app.memory.entries()[-3:]:
        app.write(f"  · {entry.header}", "dim")
        for line in entry.body.splitlines()[:3]:
            app.write(f"    {line[:100]}", "dim")
    app.write("  /memory save 记录上次任务 · /memory on|off 参考开关", "dim")


def _model(app, arg: str) -> None:
    if arg:
        app.cfg.model = arg
        app.set_model(arg)
        try:
            app.cfg.save()
        except Exception:
            pass
        app.write(f"  模型已切换为 {arg}", "dim")
        return
    app.write(f"  模型: {app.model}", "dim")
    app.write(f"  Base URL: {app.cfg.base_url}", "dim")
    app.write(f"  ctx: {app.cfg.context_limit} tok", "dim")


def _status(app, arg: str) -> None:
    ctx_tokens = app.last_ctx or app.estimate_ctx()
    pct = int(100 * ctx_tokens / max(1, app.cfg.context_limit))
    usage = app.session.usage
    app.write_renderable(
        kv_table(
            [
                ("会话", f"{app.session.id}  ({len(app.session.messages)} 条消息)"),
                ("模式", ("目标" if app.goal_mode else "标准") + f" · 最大轮数 {app.max_rounds}"),
                ("模型", app.model),
                ("上下文", f"{ctx_tokens} / {app.cfg.context_limit} tok ({pct}%)"),
                (
                    "用量",
                    f"{usage.get('total', 0)} tok (prompt {usage.get('prompt', 0)}"
                    f" / completion {usage.get('completion', 0)})",
                ),
                ("记忆", ("开" if app.memory_on else "关") + f" · {app.memory.stats()['entries']} 条"),
                ("工作目录", app.cfg.workspace),
            ],
            title="状态",
        )
    )


def _tools(app, arg: str) -> None:
    rows = []
    for name in app.tools.names():
        tool = app.tools.get(name)
        rows.append((tool.signature(), tool.description))
    app.write_renderable(command_table(rows))


def _verbose(app, arg: str) -> None:
    app.verbose = not app.verbose
    app.write(f"  工具输出详情: {'开' if app.verbose else '关'}", "dim")


def _thinking(app, arg: str) -> None:
    app.show_thinking = not getattr(app, "show_thinking", False)
    app.write(f"  思考内容显示: {'开' if app.show_thinking else '关'}", "dim")


def _export(app, arg: str) -> None:
    path = Path(arg) if arg else Path(app.cfg.workspace) / f"elagent-{app.session.id}.md"
    if not path.is_absolute():
        path = Path(app.cfg.workspace) / path
    try:
        app.store.save_markdown(app.session, path)
    except Exception as exc:
        app.write(f"  导出失败: {exc}", "warn")
        return
    app.write(f"  已导出: {path}", "dim")


def _clear(app, arg: str) -> None:
    app.clear_transcript()


def _quit(app, arg: str) -> None:
    app.request_quit()


_HANDLERS = {
    "/help": _help,
    "/?": _help,
    "/goal": _goal,
    "/rounds": _rounds,
    "/new": _new,
    "/resume": _resume,
    "/sessions": _sessions,
    "/compact": _compact,
    "/memory": _memory,
    "/model": _model,
    "/status": _status,
    "/tools": _tools,
    "/verbose": _verbose,
    "/thinking": _thinking,
    "/export": _export,
    "/clear": _clear,
    "/quit": _quit,
    "/exit": _quit,
}
