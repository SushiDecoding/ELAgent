"""Tolerant parsing of the model's action / verdict protocol.

A 7B local model will *not* reliably emit clean JSON.  Everything in this module
exists so that a malformed envelope degrades into a recoverable nudge instead of
a crashed loop:

    {"tool":"read","args":{"path":"a.py"}}      <- canonical
    ```json {...} ```                            <- fenced
    {'tool': 'read', 'args': {'path': 'a.py'}}  <- python literals
    read({"path": "a.py"})                      <- function call style
    {"read": {"path": "a.py"}}                  <- shorthand
    {"final": "done"}                           <- completion
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)```", re.S)
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'"})
_TOOL_KEYS = ("tool", "name", "action", "function", "tool_name")
_ARG_KEYS = ("args", "arguments", "parameters", "params", "input")
_TOOLCALL_RE = re.compile(
    r"<tool_call>\s*(?P<name>[A-Za-z_][\w\-]*)\s*(?P<body>.*?)</tool_call>", re.S | re.I
)
_ARG_PAIR_RE = re.compile(
    r"<arg_key>\s*(?P<key>.*?)\s*</arg_key>\s*<arg_value>\s*(?P<value>.*?)\s*</arg_value>",
    re.S | re.I,
)
_TOOL_ATTEMPT_RE = re.compile(r"<tool_call|<arg_key|<arg_value|<function=|tool_call\s*[:=]", re.I)


# ----------------------------------------------------------------------
# low level helpers
# ----------------------------------------------------------------------
def strip_fences(text: str) -> str:
    """Remove markdown fences but keep the body; join multiple blocks."""
    if "```" not in text:
        return text
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return "\n".join(block.strip() for block in blocks)
    return text.replace("```", "")


def scan_objects(text: str) -> list[str]:
    """Return every balanced {...} substring, ignoring braces inside strings."""
    results: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, in_str, esc = 0, False, False
        j = i
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        results.append(text[i : j + 1])
                        break
            j += 1
        i = j + 1
    return results


def _balance(text: str) -> str:
    """Append missing closing braces / brackets."""
    stack: list[str] = []
    in_str, esc = False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    if in_str:
        text += '"'
    return text + "".join(reversed(stack))


def _escape_newlines_in_strings(text: str) -> str:
    out: list[str] = []
    in_str, esc = False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            elif ch in "\r\n":
                out.append("\\n")
                continue
        else:
            if ch == '"':
                in_str = True
        out.append(ch)
    return "".join(out)


def _strip_comments(text: str) -> str:
    # line comments only when they start a line (protects URLs)
    text = re.sub(r"(?m)^\s*//.*$", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def repair_json(text: str) -> str:
    text = strip_fences(text).strip().translate(_SMART_QUOTES)
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = _strip_comments(text)
    text = _escape_newlines_in_strings(text)
    text = re.sub(r",\s*([}\]])", r"\1", text)                       # trailing commas
    text = re.sub(r"([{,]\s*)([A-Za-z_][\w\-]*)(\s*:)", r'\1"\2"\3', text)  # bare keys
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bNaN\b", "null", text)
    return _balance(text)


def loads_lenient(text: str) -> Any:
    """Parse JSON, then Python literals, then a repaired variant. None on failure."""
    if not isinstance(text, str):
        return None
    candidates = [text.strip(), strip_fences(text).strip()]
    for candidate in candidates:
        if not candidate:
            continue
        for variant in (candidate, repair_json(candidate)):
            try:
                return json.loads(variant)
            except Exception:
                pass
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return ast.literal_eval(variant)
            except Exception:
                pass
    return None


# ----------------------------------------------------------------------
# action protocol
# ----------------------------------------------------------------------
@dataclass
class Action:
    kind: str                    # tool | final | invalid
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    reason: str = ""
    raw: str = ""
    extra: int = 0               # additional actions found in the same turn


def _coerce_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = loads_lenient(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _call_args(body: str) -> dict[str, Any]:
    """Parse `read(path="a.py", limit=10)` style arguments."""
    body = body.strip()
    if not body:
        return {}
    if body.startswith("{"):
        parsed = loads_lenient(body)
        return parsed if isinstance(parsed, dict) else {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(f"_f({body})", mode="eval")
    except SyntaxError:
        return {}
    call = tree.body
    args: dict[str, Any] = {}
    if isinstance(call, ast.Call):
        for keyword in call.keywords:
            if keyword.arg:
                try:
                    args[keyword.arg] = ast.literal_eval(keyword.value)
                except Exception:
                    continue
        if call.args:
            try:
                positional = ast.literal_eval(call.args[0])
                if isinstance(positional, dict):
                    args.update(positional)
            except Exception:
                pass
    return args


def _action_from_dict(obj: dict[str, Any], tool_names: set[str]) -> Action | None:
    # OpenAI native shape
    calls = obj.get("tool_calls")
    if isinstance(calls, list) and calls:
        first = calls[0] if isinstance(calls[0], dict) else {}
        fn = first.get("function") or {}
        name = fn.get("name") or first.get("name") or ""
        args = _coerce_args(fn.get("arguments") or first.get("arguments") or {})
        if name:
            return _make_tool(name, args, tool_names)

    # {"final": ...} / {"answer": ...}
    for key in ("final", "answer", "conclusion", "result"):
        if key in obj and len(obj) <= 3 and not any(k in obj for k in _TOOL_KEYS):
            value = obj[key]
            if isinstance(value, (str, int, float)):
                return Action("final", text=str(value))
            if isinstance(value, (dict, list)):
                # Weak models often wrap the conclusion in an object of fields.
                return Action("final", text=json.dumps(value, ensure_ascii=False))

    for key in _TOOL_KEYS:
        if key in obj and isinstance(obj[key], str):
            name = obj[key].strip()
            if not name:
                continue
            args: dict[str, Any] = {}
            for arg_key in _ARG_KEYS:
                if arg_key in obj:
                    args = _coerce_args(obj[arg_key])
                    break
            if not args:
                args = {
                    k: v for k, v in obj.items()
                    if k not in _TOOL_KEYS and k not in _ARG_KEYS
                }
            return _make_tool(name, args, tool_names)

    # {"read": {"path": "a.py"}}
    if len(obj) == 1:
        (name, value), = obj.items()
        if name in tool_names:
            return Action("tool", tool=name, args=_coerce_args(value))

    return None


def _make_tool(name: str, args: dict[str, Any], tool_names: set[str]) -> Action:
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in tool_names:
        return Action("invalid", tool=name, reason=f"unknown_tool:{name}")
    return Action("tool", tool=normalized, args=args)


def parse_action(
    text: str,
    tool_names: set[str] | list[str],
    *,
    allow_plain_final: bool = True,
) -> Action:
    names = set(tool_names)
    raw = text or ""
    if not raw.strip():
        return Action("invalid", reason="empty", raw=raw)

    cleaned = strip_fences(raw).strip()

    actions: list[Action] = []
    for candidate in scan_objects(cleaned):
        obj = loads_lenient(candidate)
        if isinstance(obj, dict):
            action = _action_from_dict(obj, names)
            if action is not None:
                action.raw = candidate
                actions.append(action)
    if actions:
        first = actions[0]
        if len(actions) > 1:
            # Execute only the first action (single-action protocol) but tell
            # the loop how many were ignored; silently dropping the second one
            # used to make the model believe its verification command ran.
            first.extra = len(actions) - 1
        return first

    # truncated envelope: model stopped mid-JSON, repair from the first brace
    if "{" in cleaned and not scan_objects(cleaned):
        start = cleaned.find("{")
        repaired = repair_json(cleaned[start:])
        obj = loads_lenient(repaired)
        if isinstance(obj, dict):
            action = _action_from_dict(obj, names)
            if action is not None:
                action.raw = repaired
                return action

    # <tool_call>ls <arg_key>path</arg_key><arg_value>.</arg_value></tool_call>
    # (Hermes / Qwen style; Ling sometimes emits it even when asked for JSON)
    tool_call = _TOOLCALL_RE.search(cleaned)
    if tool_call:
        name = tool_call.group("name").strip().lower().replace("-", "_")
        if name not in names:
            return Action("invalid", reason=f"unknown_tool:{name}", raw=tool_call.group(0))
        body = tool_call.group("body").strip()
        args: dict[str, Any] = {}
        if "<arg_key>" in body:
            for key, raw_value in _ARG_PAIR_RE.findall(body):
                key = key.strip()
                if not key:
                    continue
                parsed_value = loads_lenient(raw_value)
                args[key] = parsed_value if parsed_value is not None else raw_value.strip()
        elif body.startswith("{"):
            parsed = loads_lenient(body)
            if isinstance(parsed, dict):
                args = parsed
        elif body:
            # <tool_call>terminalbash(command="ls", timeout=10)</tool_call>
            inner = body[1:-1] if body.startswith("(") and body.endswith(")") else body
            args = _call_args(inner)
        return Action("tool", tool=name, args=args, raw=tool_call.group(0))

    # <tool name="read">{...}</tool>  /  <read>{...}</read>
    xml = re.search(r"<(?P<tag>[\w\-]+)(?:\s+name=[\"'](?P<name>[\w\-]+)[\"'])?\s*>(?P<body>.*?)</(?P=tag)>", cleaned, re.S)
    if xml:
        name = xml.group("name") or xml.group("tag")
        body = loads_lenient(xml.group("body")) if xml.group("body").strip() else {}
        if name in names:
            return Action("tool", tool=name, args=_coerce_args(body), raw=xml.group(0))

    # read({"path": "a.py"})  /  read(path="a.py")
    call = re.match(r"^\s*([A-Za-z_][\w\-]*)\s*\((.*)\)\s*$", cleaned, re.S)
    if call and call.group(1).lower() in names:
        args = _call_args(call.group(2))
        return Action("tool", tool=call.group(1).lower(), args=args, raw=cleaned)

    # explicit final marker
    marker = re.match(r"^\s*(?:FINAL|ANSWER|最终答案|最终结论|完成)\s*[:：]\s*(.+)", cleaned, re.S | re.I)
    if marker:
        return Action("final", text=marker.group(1).strip(), raw=raw)

    # plain prose: accept as a final answer when it does not look like a broken call
    if allow_plain_final and len(cleaned) >= 12 and not re.search(r"[{}]", cleaned):
        if not _TOOL_ATTEMPT_RE.search(cleaned):
            return Action("final", text=cleaned, reason="plain_text", raw=raw)

    if _TOOL_ATTEMPT_RE.search(cleaned):
        return Action("invalid", reason="unparsed_tool_call", raw=raw)
    if re.search(r"[{}]", cleaned):
        return Action("invalid", reason="json_unparseable", raw=raw)
    return Action("invalid", reason="too_short", raw=raw)


# ----------------------------------------------------------------------
# judge verdict
# ----------------------------------------------------------------------
@dataclass
class Verdict:
    status: int          # 1 pass, 0 fail, -1 unparseable
    reason: str = ""
    raw: str = ""
    parse_error: bool = False


_TRUE_WORDS = ("通过", "合格", "正确", "完成", "成功", "pass", "passed", "true", "yes", "ok", "success")
_FALSE_WORDS = ("不通过", "不合格", "不正确", "失败", "未完成", "缺陷", "错误", "fail", "failed", "false", "no")


def _coerce_status(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("1", "true", "yes", "ok", "pass", "passed", "success"):
            return 1
        if low in ("0", "false", "no", "fail", "failed", "failure"):
            return 0
        for word in _FALSE_WORDS:
            if word in low:
                return 0
        for word in _TRUE_WORDS:
            if word in low:
                return 1
    return -1


def parse_verdict(text: str) -> Verdict:
    raw = text or ""
    cleaned = strip_fences(raw)

    for candidate in scan_objects(cleaned):
        obj = loads_lenient(candidate)
        if not isinstance(obj, dict):
            continue
        for key in ("status", "verdict", "pass", "passed", "result", "ok"):
            if key in obj:
                status = _coerce_status(obj[key])
                if status >= 0:
                    reason = obj.get("reason") or obj.get("issue") or obj.get("feedback") or ""
                    return Verdict(status, str(reason).strip(), candidate)

    match = re.search(r"[\"']?status[\"']?\s*[:=]\s*[\"']?(-?\d+)", cleaned, re.I)
    if match:
        status = _coerce_status(match.group(1))
        if status >= 0:
            reason_match = re.search(r"[\"']?(?:reason|issue|feedback)[\"']?\s*[:=]\s*[\"']([^\"']{1,400})", cleaned, re.I)
            return Verdict(status, reason_match.group(1).strip() if reason_match else "", raw)

    low = cleaned.lower()
    for word in _FALSE_WORDS:
        if word in low:
            return Verdict(0, cleaned.strip()[:200], raw, parse_error=True)
    for word in _TRUE_WORDS:
        if word in low:
            return Verdict(1, "", raw, parse_error=True)

    return Verdict(-1, "", raw, parse_error=True)
