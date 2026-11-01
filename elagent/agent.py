"""Agent core: ReAct loop, judge sub-agent, and the goal-mode round controller.

Control flow is explicit.  The model only ever produces one of two things per
turn: a tool call envelope or a final answer.  Everything else (retry, nudge,
compaction, round limits, verdict parsing, abort conditions) is Python.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import prompts
from .config import Config
from .context import ContextManager
from .events import EventBus
from .llm import LLMClient
from .memory import GlobalMemory
from .protocol import Action, Verdict, loads_lenient, parse_action, parse_verdict
from .session import Session, SessionStore
from .tools import ToolContext, ToolRegistry, load_items, render_todo


class Cancelled(BaseException):
    """Raised inside the stream callback to abort a run (BaseException on purpose)."""


@dataclass
class RunResult:
    final: str = ""
    ok: bool = False
    steps: int = 0
    tool_calls: int = 0
    changed_files: list[str] = field(default_factory=list)
    error: str = ""
    interrupted: bool = False
    tokens: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GoalResult:
    ok: bool
    rounds: int
    final: str = ""
    reasons: list[str] = field(default_factory=list)
    error: str = ""
    interrupted: bool = False


class MainAgent:
    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        tools: ToolRegistry,
        bus: EventBus,
        session: Session,
        *,
        cancel: threading.Event | None = None,
        confirm=None,
        ask_user=None,
        depth: int = 0,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.tools = tools
        self.bus = bus
        self.session = session
        self.cancel = cancel
        self.confirm = confirm
        self.tool_ctx = ToolContext(
            workspace=Path(cfg.workspace),
            config=cfg,
            session_id=session.id,
            cancel=cancel,
            confirm=confirm,
            ask_user=ask_user,
            subagent=self._run_subagent,
            workflow=self._run_workflow,
            depth=depth,
        )

    # ------------------------------------------------------------------
    def run(
        self,
        task: str,
        messages: list[dict[str, Any]] | None = None,
        *,
        feedback: str = "",
        history: str = "",
        round_no: int = 1,
        memory_block: str = "",
    ) -> RunResult:
        todo_text = render_todo(load_items(self.tool_ctx))
        system = prompts.system_prompt(
            self.cfg.workspace,
            self.tools.describe(),
            shell=str(self.cfg.shell),
            todo=todo_text,
            memory=memory_block,
        )
        if messages is None:
            messages = []
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": system}
        else:
            messages.insert(0, {"role": "system", "content": system})
        # The task prompt is appended by Python on every run: the model never has
        # to remember that a new user turn started.
        messages.append(
            {"role": "user", "content": prompts.task_prompt(task, feedback=feedback, history=history)}
        )

        ctx = ContextManager(messages, self.cfg, self.llm, self.bus)
        result = RunResult()
        invalid = 0
        repeat_counts: dict[str, int] = {}
        tool_names = self.tools.names()
        started = time.monotonic()

        for step in range(1, max(1, self.cfg.max_steps) + 1):
            if self._cancelled():
                result.interrupted = True
                result.error = "用户中断"
                return result
            result.steps = step

            if ctx.needs_compaction():
                try:
                    ctx.compact()
                except Exception as exc:  # noqa: BLE001
                    self.bus.emit("notice", text=f"压缩跳过: {exc}", level="warn")

            self.bus.emit("turn_start", step=step, ctx_tokens=ctx.tokens(), limit=self.cfg.context_limit)
            try:
                response = self.llm.chat(
                    ctx.messages,
                    on_delta=self._on_delta,
                    on_reasoning=self._on_reasoning,
                )
            except Cancelled:
                result.interrupted = True
                result.error = "用户中断"
                return result
            except Exception as exc:  # noqa: BLE001 - transport already retried
                result.error = str(exc)
                self.bus.emit("error", text=f"模型请求失败: {exc}")
                return result

            self.session.add_usage(response.prompt_tokens, response.completion_tokens)
            result.tokens += response.total_tokens
            self.bus.emit(
                "turn_end",
                text=response.text,
                reasoning=response.reasoning,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                ctx_tokens=ctx.tokens(),
                limit=self.cfg.context_limit,
                latency=response.latency,
            )

            action = self._parse(response, tool_names)

            if action.kind == "final":
                final_text = action.text.strip()
                if not final_text:
                    invalid += 1
                    ctx.add("assistant", response.text.strip()[:1000] or "{}")
                    ctx.add("user", prompts.final_retry())
                    continue
                messages.append({"role": "assistant", "content": final_text})
                result.final = final_text
                result.ok = True
                self.bus.emit("final", text=final_text, steps=step, elapsed=time.monotonic() - started)
                return result

            if action.kind == "invalid":
                invalid += 1
                self.bus.emit("notice", text=f"动作解析失败({invalid}/{self.cfg.max_invalid_actions}): {action.reason}", level="warn")
                ctx.add("assistant", (response.text or "").strip()[:1500] or "(空响应)")
                ctx.add("user", prompts.nudge(action.reason, tool_names))
                if invalid >= self.cfg.max_invalid_actions:
                    return self._forced_final(ctx, result, reason="连续输出无法解析")
                continue

            # ---- tool call -------------------------------------------
            invalid = 0
            raw_assistant = (response.text or "").strip()
            if not raw_assistant:
                raw_assistant = json.dumps({"tool": action.tool, "args": action.args}, ensure_ascii=False)
            ctx.add("assistant", raw_assistant)

            # Weak models loop by repeating the exact same call.  Let it happen
            # once, warn on the second occurrence, and end the round on the
            # third instead of burning the whole step budget.
            signature = self._signature(action)
            repeats = repeat_counts.get(signature, 0)
            if repeats:
                limit = max(1, int(getattr(self.cfg, "max_repeat_actions", 3)))
                if repeats + 1 >= limit:
                    self.bus.emit(
                        "notice",
                        text=f"重复动作达到上限，停止本轮: {action.tool}",
                        level="warn",
                    )
                    result.error = "重复动作"
                    return self._forced_final(ctx, result, reason="检测到重复动作")
                repeat_counts[signature] = repeats + 1
                self.bus.emit("notice", text=f"重复动作已拦下: {action.tool}", level="warn")
                ctx.add("user", prompts.loop_nudge(action.tool, self._args_text(action.args)))
                continue
            repeat_counts[signature] = 1

            self.bus.emit("tool_start", name=action.tool, args=action.args, step=step)
            tool_result = self.tools.execute(action.tool, action.args, self.tool_ctx)
            result.tool_calls += 1
            self.bus.emit(
                "tool_end",
                name=action.tool,
                ok=tool_result.ok,
                display=tool_result.display,
                content=tool_result.content,
                ms=tool_result.meta.get("ms", 0),
            )
            ctx.add_observation(action.tool, tool_result.ok, tool_result.content)
            if action.extra:
                self.bus.emit(
                    "notice",
                    text=f"忽略本轮多余的 {action.extra} 个动作（每轮只能输出一个 JSON）",
                    level="warn",
                )
                ctx.add("user", prompts.extra_actions(action.extra))

            changed = tool_result.meta.get("changed")
            if changed:
                # The workspace changed, so a previously repeated call may now
                # be legitimate (e.g. re-run the tests after an edit).
                repeat_counts.clear()
                if changed not in result.changed_files:
                    result.changed_files.append(str(changed))
            for path in tool_result.meta.get("changed_files") or []:
                # Sub-agents/workflows report files changed in their own runs,
                # so the judge sees them too.
                if path not in result.changed_files:
                    result.changed_files.append(str(path))
            if action.tool == "terminalbash":
                result.evidence.append(
                    {
                        "cmd": str(action.args.get("command", ""))[:300],
                        "exit": tool_result.meta.get("exit"),
                        "out": tool_result.content[:800],
                    }
                )
                result.evidence = result.evidence[-8:]
            if self._cancelled():
                result.interrupted = True
                result.error = "用户中断"
                return result

        return self._forced_final(ctx, result, reason="达到本轮步数上限")

    # ------------------------------------------------------------------
    def _parse(self, response, tool_names: list[str]) -> Action:
        if response.tool_calls:
            first = response.tool_calls[0]
            fn = first.get("function") or {}
            name = str(fn.get("name") or "").strip().lower()
            raw_args = fn.get("arguments") or "{}"
            args = loads_lenient(raw_args) if isinstance(raw_args, str) else raw_args
            if name in tool_names and isinstance(args, dict):
                if getattr(response, "truncated", False):
                    return Action("invalid", reason="output_truncated")
                return Action("tool", tool=name, args=args)
        action = parse_action(response.text or "", tool_names)
        if getattr(response, "truncated", False) and action.kind != "invalid":
            # A half-written JSON action can silently corrupt a file; let the
            # normal invalid-action nudge ask the model to finish it.
            return Action("invalid", reason="output_truncated", raw=action.raw)
        return action

    @staticmethod
    def _args_text(args: dict[str, Any]) -> str:
        try:
            text = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except Exception:  # noqa: BLE001 - diagnostics only
            text = str(args)
        return text if len(text) <= 200 else text[:199] + "…"

    @classmethod
    def _signature(cls, action: Action) -> str:
        return f"{action.tool}|{cls._args_text(action.args)[:400]}"

    # ------------------------------------------------------------------
    # sub-agents (bounded; silent bus so the parent transcript stays clean)
    # ------------------------------------------------------------------
    def _run_subagent(self, task: str, max_steps: int = 0) -> tuple[bool, str, list[str]]:
        depth = int(getattr(self.tool_ctx, "depth", 0))
        max_depth = max(0, int(getattr(self.cfg, "subagent_max_depth", 1)))
        if depth >= max_depth:
            return False, "error: 子代理嵌套已达上限；请直接在当前代理里完成"
        ceiling = max(1, int(getattr(self.cfg, "subagent_max_steps", 8)))
        limit = max(1, min(int(max_steps or ceiling), ceiling))
        sub_cfg = replace(self.cfg, max_steps=limit)
        self.bus.emit("notice", text=f"子代理开始（≤{limit} 步）: {str(task)[:60]}", level="dim")
        sub = MainAgent(
            sub_cfg,
            self.llm,
            self.tools.without("ask", "subagent", "workflow", "todo"),
            EventBus(),
            self.session,
            cancel=self.cancel,
            confirm=self.confirm,
            depth=depth + 1,
        )
        result = sub.run(str(task), messages=None, memory_block="")
        text = (result.final or result.error or "").strip()
        self.bus.emit(
            "notice",
            text=f"子代理{'完成' if result.ok else '失败'}（{result.steps} 步）",
            level="dim",
        )
        return bool(result.ok and result.final), text or "子代理未产出结论", list(result.changed_files)

    def _run_workflow(self, steps: list[str]) -> tuple[bool, str, list[str]]:
        limit = max(1, int(getattr(self.cfg, "workflow_max_steps", 4)))
        steps = [str(step).strip() for step in steps if str(step).strip()][:limit]
        if not steps:
            return False, "error: workflow 没有有效步骤", []
        rows: list[str] = []
        context = ""
        all_ok = True
        changed_files: list[str] = []
        for index, step in enumerate(steps, 1):
            task = step
            if context:
                task += f"\n\n【前序步骤结果，仅供参考】\n{context[-3000:]}"
            ok, text, files = self._run_subagent(task, int(getattr(self.cfg, "subagent_max_steps", 8)))
            rows.append(f"{index}. {'完成' if ok else '失败'}：{text[:1200]}")
            for path in files:
                if path not in changed_files:
                    changed_files.append(path)
            if not ok:
                all_ok = False
                break
            context += f"\n第 {index} 步：{text[:800]}"
        return all_ok, "\n".join(rows), changed_files

    def _forced_final(self, ctx: ContextManager, result: RunResult, *, reason: str) -> RunResult:
        """One last attempt to get a written conclusion; otherwise synthesise one."""
        ctx.add("user", prompts.force_final())
        try:
            response = self.llm.chat(ctx.messages, on_delta=self._on_delta, stream=True, max_tokens=500)
            action = parse_action(response.text or "", self.tools.names())
            self.session.add_usage(response.prompt_tokens, response.completion_tokens)
            if action.kind == "final" and action.text.strip():
                result.final = action.text.strip()
                ctx.add("assistant", result.final)
                self.bus.emit("final", text=result.final, steps=result.steps, elapsed=0.0, degraded=True)
                return result
        except Cancelled:
            result.interrupted = True
        except Exception as exc:  # noqa: BLE001
            self.bus.emit("notice", text=f"收尾请求失败: {exc}", level="warn")

        result.ok = False
        result.error = result.error or reason
        result.final = self._synthesise(ctx, reason)
        self.bus.emit("final", text=result.final, steps=result.steps, elapsed=0.0, degraded=True)
        return result

    @staticmethod
    def _synthesise(ctx: ContextManager, reason: str) -> str:
        observations = [m["content"] for m in ctx.messages if str(m.get("content", "")).startswith("OBS[")]
        tail = observations[-3:] if observations else []
        lines = [f"任务未完成：{reason}。"]
        if tail:
            lines.append("最近的工具结果：")
            for item in tail:
                lines.append(item[:400])
        lines.append("可以继续对话让我接着做，或调整 /rounds 后重试。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _on_delta(self, text: str) -> None:
        if self._cancelled():
            raise Cancelled()
        self.bus.emit("delta", text=text)

    def _on_reasoning(self, text: str) -> None:
        if self._cancelled():
            raise Cancelled()
        self.bus.emit("reasoning", text=text)

    def _cancelled(self) -> bool:
        return bool(self.cancel and self.cancel.is_set())


class JudgeAgent:
    """Strict reviewer.  Outputs status 1/0; -1 means the verdict was unparseable."""

    def __init__(self, cfg: Config, llm: LLMClient, bus: EventBus) -> None:
        self.cfg = cfg
        self.llm = llm
        self.bus = bus

    def evaluate(self, task: str, result: RunResult, session: Session, todo_text: str = "") -> Verdict:
        files = self._files(result.changed_files)
        evidence = self._evidence(result.evidence)
        prompt = prompts.judge_prompt(task, result.final, files, evidence, todo_text)
        messages = [{"role": "user", "content": prompt}]
        last = Verdict(-1, "", "", parse_error=True)

        for attempt in range(max(1, self.cfg.judge_retries)):
            try:
                response = self.llm.chat(
                    messages,
                    stream=False,
                    temperature=0.0,
                    max_tokens=self.cfg.judge_max_tokens,
                    fallback_reasoning=True,
                )
                session.add_usage(response.prompt_tokens, response.completion_tokens)
            except Exception as exc:  # noqa: BLE001
                self.bus.emit("notice", text=f"评审请求失败({attempt + 1}): {exc}", level="warn")
                time.sleep(0.6 * (attempt + 1))
                continue
            verdict = parse_verdict(response.text or "")
            self.bus.emit("verdict", status=verdict.status, reason=verdict.reason, raw=response.text or "")
            if verdict.status >= 0:
                return verdict
            last = verdict
            messages.append({"role": "assistant", "content": (response.text or "")[:500]})
            messages.append({"role": "user", "content": prompts.judge_retry()})

        return last

    def _files(self, paths: list[str]) -> str:
        rows = []
        root = Path(self.cfg.workspace)
        for rel in paths[:10]:
            path = Path(rel)
            if not path.is_absolute():
                path = root / path
            try:
                if path.exists():
                    text = path.read_text(encoding="utf-8", errors="replace")
                    rows.append(f"{rel} ({len(text.splitlines())} 行, {len(text)} 字符)")
                else:
                    rows.append(f"{rel} (不存在!)")
            except Exception:
                rows.append(rel)
        return "\n".join(rows)

    @staticmethod
    def _evidence(items: list[dict[str, Any]]) -> str:
        if not items:
            return ""
        rows = []
        for item in items[-6:]:
            rows.append(f"$ {item.get('cmd', '')}\nexit={item.get('exit')}\n{str(item.get('out', ''))[:500]}")
        return "\n---\n".join(rows)


class GoalRunner:
    """Runs main agent -> judge -> repair rounds until pass or a hard limit."""

    def __init__(
        self,
        cfg: Config,
        llm: LLMClient,
        tools: ToolRegistry,
        bus: EventBus,
        session: Session,
        store: SessionStore,
        memory: GlobalMemory | None = None,
        *,
        cancel: threading.Event | None = None,
        confirm=None,
        ask_user=None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.tools = tools
        self.bus = bus
        self.session = session
        self.store = store
        self.memory = memory
        self.cancel = cancel
        self.confirm = confirm
        self.ask_user = ask_user
        self.judge = JudgeAgent(cfg, llm, bus)

    def run(self, task: str, *, memory_block: str = "") -> GoalResult:
        reasons: list[str] = []
        history_rows: list[str] = []
        round_no = 0
        consecutive_errors = 0
        judge_errors = 0
        last_result: RunResult | None = None

        while self.cfg.max_rounds < 0 or round_no < self.cfg.max_rounds:
            if self.cancel and self.cancel.is_set():
                return GoalResult(False, round_no, last_result.final if last_result else "", reasons, "用户中断", True)
            round_no += 1
            self.bus.emit("round_start", n=round_no, max=self.cfg.max_rounds)

            agent = MainAgent(
                self.cfg, self.llm, self.tools, self.bus, self.session,
                cancel=self.cancel, confirm=self.confirm, ask_user=self.ask_user,
            )
            result = agent.run(
                task,
                messages=None,
                feedback=reasons[-1] if reasons else "",
                history="\n".join(history_rows[-8:]),
                round_no=round_no,
                memory_block=memory_block,
            )
            last_result = result

            if result.interrupted:
                return GoalResult(False, round_no, result.final, reasons, "用户中断", True)
            if result.error and not result.final:
                consecutive_errors += 1
                self.bus.emit("error", text=f"第 {round_no} 轮失败: {result.error}")
                if consecutive_errors >= 2:
                    return GoalResult(False, round_no, "", reasons, result.error)
            else:
                consecutive_errors = 0

            todo_text = render_todo(load_items(agent.tool_ctx))
            if not result.final:
                # Python decides: a round without a conclusion is a failed round,
                # regardless of what the reviewer might say.
                verdict = Verdict(0, result.error or "主 Agent 未产出结论")
                self.bus.emit("verdict", status=0, reason=verdict.reason, raw="")
            else:
                verdict = self.judge.evaluate(task, result, self.session, todo_text)

            self.session.rounds.append(
                {
                    "n": round_no,
                    "status": verdict.status,
                    "steps": result.steps,
                    "reason": verdict.reason,
                    "final": result.final[:500],
                    "files": result.changed_files,
                }
            )
            self.store.save(self.session)

            if verdict.status == 1:
                self.bus.emit("goal_done", rounds=round_no, ok=True)
                return GoalResult(True, round_no, result.final, reasons)

            if verdict.status == 0:
                judge_errors = 0
                reason = verdict.reason or "评审未给出具体原因"
                reasons.append(reason)
                history_rows.append(f"第 {round_no} 轮未通过：{reason[:120]}")
            else:
                judge_errors += 1
                self.bus.emit("notice", text=f"评审输出无法解析（{judge_errors}/3）", level="warn")
                if judge_errors >= 3:
                    return GoalResult(False, round_no, result.final, reasons, "评审子 Agent 连续输出非法 JSON，已中止")

        self.bus.emit("goal_done", rounds=round_no, ok=False)
        return GoalResult(False, round_no, last_result.final if last_result else "", reasons, "达到最大轮数")
