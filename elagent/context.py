"""Context window management.

The model never decides when to compact.  Python measures the estimate, calls a
dedicated summarisation request, and replaces the middle of the transcript with
a short progress note.  If summarisation fails, a mechanical fallback keeps the
loop alive.
"""

from __future__ import annotations

from typing import Any

from . import prompts
from .events import EventBus
from .tokens import estimate_messages, estimate_tokens, truncate_to_tokens

SUMMARY_HEADER = "【进度摘要】"


class ContextManager:
    def __init__(self, messages: list[dict[str, Any]], cfg, llm, bus: EventBus) -> None:
        self.messages = messages
        self.cfg = cfg
        self.llm = llm
        self.bus = bus
        self.compactions = 0

    # ------------------------------------------------------------------
    def tokens(self) -> int:
        return estimate_messages(self.messages)

    def needs_compaction(self) -> bool:
        used = self.tokens()
        return used > self.cfg.compact_threshold or used + self.cfg.max_tokens > self.cfg.context_limit

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def add_observation(self, tool: str, ok: bool, content: str) -> None:
        limit = int(self.cfg.tool_output_limit)
        body = content if len(content) <= limit else content[:limit] + "\n...（工具输出已截断）"
        status = "ok" if ok else "error"
        self.add("user", f"OBS[{tool}] {status}\n{body}")

    # ------------------------------------------------------------------
    def compact(self, force: bool = False) -> bool:
        if not force and not self.needs_compaction():
            return False
        keep_n = max(2, int(self.cfg.keep_recent_messages))
        if len(self.messages) <= keep_n + 2:
            return False

        system = self.messages[0]
        if system.get("role") != "system":
            system = {"role": "system", "content": str(system.get("content", ""))}
        middle = self.messages[1 : -keep_n]
        tail = self.messages[-keep_n:]
        before = self.tokens()

        summary = self._summarise(middle)
        anchor = {
            "role": "user",
            "content": f"{SUMMARY_HEADER}\n{summary}\n（以上为系统压缩的历史进度，原始记录已保存到会话文件）",
        }
        new_messages = [system, anchor]
        if tail and tail[0].get("role") == "user":
            # avoid two consecutive user turns confusing small chat templates
            new_messages.append({"role": "assistant", "content": "收到，继续。"})
        new_messages.extend(tail)
        self.messages[:] = new_messages
        self.compactions += 1
        after = self.tokens()
        self.bus.emit("compact", before=before, after=after, count=self.compactions)
        return True

    # ------------------------------------------------------------------
    def _summarise(self, middle: list[dict[str, Any]]) -> str:
        transcript_lines = []
        for message in middle:
            role = message.get("role", "?")
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            transcript_lines.append(f"[{role}] {content[:700]}")
        transcript = "\n".join(transcript_lines)
        transcript = truncate_to_tokens(transcript, 3000)

        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompts.compact_prompt(transcript)}],
                stream=False,
                temperature=0.1,
                max_tokens=600,
            )
            text = (response.text or "").strip()
            if text and len(text) > 20:
                return text
        except Exception as exc:  # noqa: BLE001 - fallback is mandatory
            self.bus.emit("notice", text=f"上下文压缩失败，改用机械摘要: {exc}", level="warn")

        return self._mechanical(middle)

    @staticmethod
    def _mechanical(middle: list[dict[str, Any]]) -> str:
        tools: list[str] = []
        last_user = ""
        for message in middle:
            content = str(message.get("content") or "")
            if content.startswith("OBS["):
                name = content[4:].split("]", 1)[0]
                if name not in tools:
                    tools.append(name)
            elif message.get("role") == "user":
                last_user = content[:200]
        parts = [f"（自动压缩）此前共 {len(middle)} 条消息。"]
        if tools:
            parts.append(f"使用过的工具：{', '.join(tools[:8])}。")
        if last_user:
            parts.append(f"最近的请求：{last_user}")
        parts.append("细节已从上下文移除，如需确认请重新 read/grep。")
        return " ".join(parts)
