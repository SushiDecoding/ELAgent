"""OpenAI-compatible client tuned for a slow, occasionally unreliable local model.

Design notes
------------
* Every request is wrapped in a bounded retry loop with jittered backoff.
* Streaming is the default; a partial stream is returned instead of raising so
  the agent loop can still make progress with what it got.
* Usage is estimated locally when the server does not report it (llama.cpp
  usually reports timings, not token usage, for streamed responses).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .tokens import estimate_messages, estimate_tokens

DeltaCallback = Callable[[str], None]


class LLMError(RuntimeError):
    """Base class for model transport errors."""


class LLMConnectionError(LLMError):
    pass


class _EmptyResponse(LLMError):
    """200 OK but no content: the server returned an empty message."""


class _TruncatedOutput(LLMError):
    """Local stream cap reached before the model produced anything usable."""


def _truncated_message() -> str:
    return (
        "模型输出超过本地流上限，已中断（可调 max_stream_chars，"
        "或在 llama.cpp 设置 --n-predict）"
    )


class LLMHTTPError(LLMError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:400]}")
        self.status = status
        self.body = body


@dataclass
class LLMResponse:
    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    truncated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._client = httpx.Client(
            timeout=httpx.Timeout(cfg.request_timeout, connect=10.0),
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            },
            # llama.cpp (and some local gateways) answer 404 to the first POST
            # that reuses a keep-alive connection opened by GET /v1/models.
            # Disabling keep-alive costs ~1ms locally and removes the whole class
            # of stale-connection failures.
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=0),
        )
        self._model: str = "" if cfg.model in ("", "auto") else cfg.model

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def _url(self, suffix: str) -> str:
        return self.cfg.base_url.rstrip("/") + suffix

    def set_model(self, name: str) -> None:
        self._model = name

    def resolve_model(self, retries: int = 2) -> str:
        if self._model:
            return self._model
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = self._client.get(self._url("/models"), timeout=15.0)
                resp.raise_for_status()
                data = resp.json()
                items = data.get("data") or data.get("models") or []
                if items:
                    first = items[0]
                    self._model = first.get("id") or first.get("model") or "local-model"
                    return self._model
            except Exception as exc:  # noqa: BLE001 - transport layer
                last = exc
                time.sleep(0.5 * (attempt + 1))
        if last:
            raise LLMConnectionError(f"无法获取模型列表: {last}") from last
        raise LLMConnectionError("模型列表为空")

    def ping(self) -> dict[str, Any]:
        started = time.monotonic()
        resp = self._client.get(self._url("/models"), timeout=15.0)
        resp.raise_for_status()
        return {"status": resp.status_code, "latency": time.monotonic() - started, "body": resp.json()}

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool | None = None,
        on_delta: DeltaCallback | None = None,
        on_reasoning: DeltaCallback | None = None,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        fallback_reasoning: bool = False,
    ) -> LLMResponse:
        stream = self.cfg.stream if stream is None else stream
        payload: dict[str, Any] = {
            "model": self.resolve_model(),
            "messages": messages,
            "stream": stream,
        }
        # Generation parameters are owned by llama.cpp (server launch flags).
        # ELAgent sends none by default and never changes them mid-run; a user
        # who really wants per-request overrides can set send_generation_params.
        if getattr(self.cfg, "send_generation_params", False):
            payload["temperature"] = self.cfg.temperature if temperature is None else temperature
            payload["top_p"] = self.cfg.top_p
            payload["max_tokens"] = self.cfg.max_tokens if max_tokens is None else max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stop:
            payload["stop"] = stop

        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            started = time.monotonic()
            try:
                if stream:
                    result = self._chat_stream(payload, on_delta, on_reasoning)
                else:
                    result = self._chat_once(payload)
                result.latency = time.monotonic() - started
                if not result.text and result.reasoning and fallback_reasoning:
                    # Reasoning content is a last-resort body for the judge
                    # only; the main loop must not turn it into a final.
                    result.text = result.reasoning
                if not result.text and not result.tool_calls:
                    if result.truncated:
                        raise _TruncatedOutput(_truncated_message())
                    raise _EmptyResponse("模型返回空内容（生成参数由 llama.cpp 决定）")
                if not result.prompt_tokens:
                    result.prompt_tokens = estimate_messages(messages)
                if not result.completion_tokens:
                    result.completion_tokens = estimate_tokens(result.text)
                return result
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = LLMConnectionError(
                    f"无法连接 {self.cfg.base_url}，请确认本地模型服务已启动"
                )
                last_error.__cause__ = exc
            except (httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ReadError) as exc:
                last_error = LLMError(f"连接中断: {exc}")
            except LLMHTTPError as exc:
                if exc.status not in (404, 408, 409, 429, 500, 502, 503, 504):
                    raise
                last_error = exc
            except _TruncatedOutput:
                # Retrying a runaway generation just burns the same minutes.
                raise
            except LLMError as exc:
                last_error = exc
            except Exception as exc:  # noqa: BLE001 - defensive
                last_error = LLMError(f"请求异常: {exc}")

            if attempt < self.cfg.max_retries:
                delay = min(8.0, 0.8 * (2 ** attempt)) + random.uniform(0, 0.3)
                time.sleep(delay)

        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    def _chat_once(self, payload: dict[str, Any]) -> LLMResponse:
        resp = self._client.post(self._url("/chat/completions"), json=payload)
        if resp.status_code >= 400:
            raise LLMHTTPError(resp.status_code, resp.text)
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMError(f"响应不是合法 JSON: {resp.text[:200]}") from exc
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return LLMResponse(
            text=message.get("content") or "",
            reasoning=message.get("reasoning_content") or message.get("reasoning") or "",
            tool_calls=message.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason") or "",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    def _chat_stream(
        self,
        payload: dict[str, Any],
        on_delta: DeltaCallback | None,
        on_reasoning: DeltaCallback | None,
    ) -> LLMResponse:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish_reason = ""
        usage: dict[str, Any] = {}
        broken = False
        capped = False
        char_count = 0
        max_chars = int(getattr(self.cfg, "max_stream_chars", 0) or 0)

        with self._client.stream("POST", self._url("/chat/completions"), json=payload) as resp:
            if resp.status_code >= 400:
                body = resp.read().decode("utf-8", "replace")
                raise LLMHTTPError(resp.status_code, body)
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                if not line.startswith("{"):
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or choice.get("message") or {}

                piece = delta.get("content")
                if piece:
                    text_parts.append(piece)
                    char_count += len(piece)
                    if on_delta:
                        try:
                            on_delta(piece)
                        except Exception:
                            broken = True

                reason = delta.get("reasoning_content") or delta.get("reasoning")
                if reason:
                    reasoning_parts.append(reason)
                    char_count += len(reason)
                    if on_reasoning:
                        try:
                            on_reasoning(reason)
                        except Exception:
                            broken = True

                for call in delta.get("tool_calls") or []:
                    index = int(call.get("index", 0) or 0)
                    slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if call.get("id"):
                        slot["id"] = call["id"]
                    fn = call.get("function") or {}
                    name = fn.get("name") or ""
                    if name and name != slot["name"]:
                        slot["name"] = name if not slot["name"] else slot["name"] + name
                    args = fn.get("arguments") or ""
                    if args:
                        slot["arguments"] += args

                if max_chars and char_count >= max_chars:
                    # Stop reading; closing the response also aborts the server
                    # request instead of waiting for the whole runaway body.
                    capped = True
                    break

        tool_calls = []
        for _, slot in sorted(calls.items()):
            if slot["name"]:
                tool_calls.append(
                    {"id": slot["id"] or "call_0", "function": {"name": slot["name"], "arguments": slot["arguments"]}}
                )

        return LLMResponse(
            text="".join(text_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            truncated=broken or capped,
        )
