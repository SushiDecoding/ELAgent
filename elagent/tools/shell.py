"""Shell tool: Windows-aware, cancellable, output-capped."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .base import Tool, ToolContext, ToolError, ToolResult

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")

_DANGEROUS = [
    r"\brm\s+-[a-z]*[rf]", r"\bdel\s+/[fsq]", r"\bformat\s+[a-z]:",
    r"\bmkfs(\.\w+)?\b", r"\bdiskpart\b", r"\bshutdown\b", r"\breboot\b",
    r":\(\)\s*\{", r">\s*/dev/sd", r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[a-z]*f", r"\bremove-item\b[^\n]*-recurse",
    r"\breg\s+delete\b", r"\btaskkill\b[^\n]*/f[^\n]*/im\b",
]


def find_bash() -> str | None:
    """Locate Git Bash on Windows.

    `shutil.which("bash")` may return the WSL launcher under System32: slow,
    often blocked by policy, and with a different filesystem view.  Prefer a
    real Git for Windows installation.
    """
    if os.name != "nt":
        return shutil.which("bash")

    candidates: list[Path] = []
    for env in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if not base:
            continue
        candidates.append(Path(base) / "Git" / "bin" / "bash.exe")
        candidates.append(Path(base) / "Programs" / "Git" / "bin" / "bash.exe")
    git = shutil.which("git")
    if git:
        root = Path(git).resolve().parent.parent
        candidates.append(root / "bin" / "bash.exe")
        candidates.append(root / "usr" / "bin" / "bash.exe")
    for candidate in candidates:
        try:
            if candidate.exists():
                return str(candidate)
        except OSError:
            continue

    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return None


def _decode(data: bytes) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=10,
            )
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _cap(text: str, head: int = 5000, tail: int = 1500) -> tuple[str, bool]:
    if len(text) <= head + tail:
        return text, False
    return text[:head] + f"\n...（省略 {len(text) - head - tail} 字符）...\n" + text[-tail:], True


class BashTool(Tool):
    name = "terminalbash"
    description = "执行 shell 命令，返回退出码与输出"
    params = {"command": "", "timeout": "60", "cwd": "."}

    def _argv(self, command: str, ctx: ToolContext) -> list[str]:
        shell = str(getattr(ctx.config, "shell", "auto") or "auto").lower()
        if shell == "auto":
            if os.name == "nt":
                shell = "bash" if find_bash() else "cmd"
            else:
                shell = "bash"
        if shell == "cmd":
            return ["cmd", "/c", command]
        if shell == "powershell":
            return ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
        bash = find_bash()
        if not bash:
            raise ToolError("未找到 Git Bash；请安装 Git for Windows 或设置 shell=cmd")
        return [bash, "-c", command]

    def run(self, args, ctx: ToolContext) -> ToolResult:
        command = self.arg_str(args, "command").strip()
        if not command:
            raise ToolError("command 不能为空")
        timeout = max(1, min(600, self.arg_int(args, "timeout", 60)))
        cwd = ctx.resolve(self.arg_str(args, "cwd", "."))
        if not cwd.is_dir():
            raise ToolError(f"cwd 不是目录: {ctx.rel(cwd)}")

        if getattr(ctx.config, "confirm_dangerous", True) and not getattr(ctx.config, "yolo", False):
            low = command.lower()
            if any(re.search(pattern, low) for pattern in _DANGEROUS):
                approved = ctx.ask(f"命令可能破坏数据，是否执行？\n  {command[:300]}")
                if not approved:
                    return ToolResult(False, "error: 用户拒绝执行该命令（危险操作）", display="已拒绝")

        env = os.environ.copy()
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "GIT_PAGER": "cat", "PAGER": "cat"})

        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                self._argv(command, ctx),
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"找不到 shell: {exc}") from exc

        box: dict[str, bytes] = {}

        def _reader() -> None:
            try:
                out, err = proc.communicate()
                box["out"], box["err"] = out or b"", err or b""
            except Exception:
                box.setdefault("out", b"")
                box.setdefault("err", b"")

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()

        timed_out = False
        cancelled = False
        while thread.is_alive():
            if ctx.cancelled():
                cancelled = True
                _kill_tree(proc)
                break
            if time.monotonic() - started > timeout:
                timed_out = True
                _kill_tree(proc)
                break
            time.sleep(0.05)
        thread.join(timeout=3)

        elapsed = time.monotonic() - started
        stdout = _ANSI_RE.sub("", _decode(box.get("out", b""))).strip()
        stderr = _ANSI_RE.sub("", _decode(box.get("err", b""))).strip()
        exit_code = proc.returncode if proc.returncode is not None else -1

        if cancelled:
            return ToolResult(False, "error: 命令被用户中断", display="已中断")
        if timed_out:
            body = f"exit=-1 timeout={timeout}s\n{stdout}"
            if stderr:
                body += f"\n[stderr]\n{stderr}"
            return ToolResult(False, body[:6000] + "\n（超时终止，可调大 timeout 或改用后台方式）", display=f"超时 {timeout}s")

        parts = [f"exit={exit_code} time={elapsed:.2f}s"]
        if stdout:
            capped, _ = _cap(stdout)
            parts.append(capped)
        if stderr:
            capped, _ = _cap(stderr, 3000, 1000)
            parts.append("[stderr]\n" + capped)
        if not stdout and not stderr:
            parts.append("(无输出)")
        content = "\n".join(parts)
        summary = f"exit={exit_code} · {elapsed:.2f}s"
        return ToolResult(exit_code == 0, content, display=summary, meta={"exit": exit_code})
