"""Agent-level tools: time context, asking the user, sub-agents, workflows.

These four tools need services the filesystem tools do not: the wall clock and
session counters, a blocking prompt to the human, and a bounded child agent.
The services are injected through ToolContext by MainAgent/GoalRunner, so the
tool classes stay small and the UI stays out of the core.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .base import Tool, ToolContext, ToolError, ToolResult

_WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


class TimeTool(Tool):
    name = "time"
    description = "时间上下文：当前时间、时区、会话耗时与用量"
    params = {}

    def run(self, args, ctx: ToolContext) -> ToolResult:
        now = datetime.now().astimezone()
        offset = now.strftime("%z")
        zone = now.tzname() or ""
        lines = [
            now.strftime("%Y-%m-%d %H:%M:%S") + f" {offset} {_WEEKDAYS[now.weekday()]}",
            f"时区 {zone} ({offset})",
            f"工作目录 {ctx.config.workspace}",
        ]
        summary = self._session_summary(ctx, now)
        if summary:
            lines.append(summary)
        text = "\n".join(lines)
        return ToolResult(True, text, display=now.strftime("%H:%M:%S"))

    @staticmethod
    def _session_summary(ctx: ToolContext, now: datetime) -> str:
        try:
            path = Path(ctx.config.sessions_dir) / f"{ctx.session_id}.json"
            if not path.exists():
                return ""
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - time must never fail a task
            return ""
        usage = data.get("usage") or {}
        created = str(data.get("created") or "")
        elapsed = ""
        try:
            start = datetime.strptime(created, "%Y-%m-%d %H:%M:%S")
            minutes = max(0, int((now.replace(tzinfo=None) - start).total_seconds() // 60))
            elapsed = f" · 已用 {minutes} 分钟" if minutes else ""
        except Exception:
            pass
        return (
            f"会话 {ctx.session_id}{elapsed} · 消息 {len(data.get('messages') or [])} 条"
            f" · 用量 {usage.get('total', 0)} tok"
        )


class AskTool(Tool):
    name = "ask"
    description = "向用户提问并等待回答（仅交互模式，谨慎使用）"
    params = {"question": ""}

    def run(self, args, ctx: ToolContext) -> ToolResult:
        question = self.arg_str(args, "question").strip()
        if not question:
            raise ToolError("question 不能为空")
        if ctx.ask_user is None:
            return ToolResult(
                False,
                "error: 当前是非交互模式，无法向用户提问；请根据已有信息自行决定并继续",
                display="非交互模式",
            )
        answer = ctx.ask_user(question)
        if answer is None:
            return ToolResult(
                False,
                "error: 用户未回答（取消或界面不可用）；请自行决定并继续",
                display="未回答",
            )
        answer = str(answer).strip()
        body = f"用户回答：{answer}" if answer else "用户回答：（空）"
        return ToolResult(True, body, display=f"回答 {answer[:40] or '（空）'}")


class SubagentTool(Tool):
    name = "subagent"
    description = "启动子代理独立完成一个子任务并返回结论（成本高）"
    params = {"task": "", "max_steps": "8"}

    def run(self, args, ctx: ToolContext) -> ToolResult:
        task = self.arg_str(args, "task").strip()
        if not task:
            raise ToolError("task 不能为空；subagent 需要一个自包含的子任务")
        if ctx.subagent is None:
            return ToolResult(False, "error: 当前模式不支持子代理", display="不可用")
        steps = self.arg_int(args, "max_steps", int(getattr(ctx.config, "subagent_max_steps", 8)))
        ok, text, changed = ctx.subagent(task, steps)
        text = text.strip() or "子代理未产出结论"
        return ToolResult(
            ok,
            text,
            display=f"子代理 · {'完成' if ok else '失败'}",
            meta={"changed_files": changed},
        )


class WorkflowTool(Tool):
    name = "workflow"
    description = "顺序执行 2-4 个步骤，每步交给子代理，返回汇总"
    params = {"steps": "[]"}

    def run(self, args, ctx: ToolContext) -> ToolResult:
        raw = args.get("steps", args.get("tasks", args.get("items")))
        steps = _normalise_steps(raw)
        if not steps:
            raise ToolError("steps 不能为空：传字符串列表，例如 [\"读代码\", \"写测试\"]")
        limit = max(1, int(getattr(ctx.config, "workflow_max_steps", 4)))
        if len(steps) > limit:
            steps = steps[:limit]
        if ctx.workflow is None:
            return ToolResult(False, "error: 当前模式不支持 workflow", display="不可用")
        ok, text, changed = ctx.workflow(steps)
        text = text.strip() or "workflow 未产出结果"
        return ToolResult(
            ok,
            text,
            display=f"workflow {len(steps)} 步 · {'完成' if ok else '失败'}",
            meta={"changed_files": changed},
        )


def _normalise_steps(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except Exception:
                return [line.strip() for line in stripped.splitlines() if line.strip()]
        else:
            return [line.strip() for line in stripped.splitlines() if line.strip()]
    if not isinstance(value, list):
        return []
    steps: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("task") or item.get("step") or item.get("title") or "").strip()
        else:
            text = str(item).strip()
        if text:
            steps.append(text[:500])
    return steps
