"""Filesystem tools.  Output is intentionally compact: line numbers, no fluff."""

from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import Path

from .base import Tool, ToolContext, ToolError, ToolResult

_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    ".idea", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".elagent",
}
_MAX_FILE_BYTES = 2 * 1024 * 1024


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" in data[:2048]:
        raise ToolError(f"{path.name} 看起来是二进制文件")
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".elagent.tmp")
    tmp.write_text(content, encoding="utf-8", newline="")
    os.replace(tmp, path)


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"


class ReadTool(Tool):
    name = "read"
    description = "读文本文件，返回带行号内容"
    params = {"path": "", "offset": "1", "limit": "200"}

    def run(self, args, ctx: ToolContext) -> ToolResult:
        path = ctx.resolve(self.arg_str(args, "path"), must_exist=True)
        if path.is_dir():
            raise ToolError(
                f"{ctx.rel(path)} 是目录，read 只能读文件；下一步用 ls path=\"{ctx.rel(path)}\"，"
                "不要再次 read 该路径"
            )
        if path.stat().st_size > _MAX_FILE_BYTES:
            raise ToolError(f"文件过大 ({_fmt_size(path.stat().st_size)})，请用 grep 或分段读取")
        text = read_text(path)
        lines = text.splitlines()
        offset = max(1, self.arg_int(args, "offset", 1))
        limit = max(1, min(1000, self.arg_int(args, "limit", 200)))
        chunk = lines[offset - 1 : offset - 1 + limit]
        if not chunk:
            return ToolResult(True, f"{ctx.rel(path)} 共 {len(lines)} 行，offset={offset} 无内容")
        body = "\n".join(f"{offset + i:>5}|{line}" for i, line in enumerate(chunk))
        tail = ""
        end = offset - 1 + len(chunk)
        if end < len(lines):
            tail = f"\n...（共 {len(lines)} 行，剩余 {len(lines) - end} 行，用 offset={end + 1} 继续）"
        return ToolResult(
            True,
            f"{ctx.rel(path)} ({len(lines)} 行)\n{body}{tail}",
            display=f"{ctx.rel(path)} · {len(chunk)} 行",
        )


class WriteTool(Tool):
    name = "write"
    description = "新建或整体覆盖文件，自动建目录"
    params = {"path": "", "content": ""}

    def run(self, args, ctx: ToolContext) -> ToolResult:
        raw = args.get("content", "")
        if isinstance(raw, (dict, list)):
            raw = json.dumps(raw, ensure_ascii=False, indent=2)
        content = str(raw)
        if "content" not in args:
            raise ToolError("缺少 content 参数")
        path = ctx.resolve(self.arg_str(args, "path"))
        if path.is_dir():
            raise ToolError(f"{ctx.rel(path)} 是目录")
        existed = path.exists()
        old_lines = 0
        if existed:
            try:
                old_lines = len(read_text(path).splitlines())
            except Exception:
                old_lines = 0
        write_text(path, content)
        new_lines = len(content.splitlines())
        note = f"覆盖原文件（原 {old_lines} 行）" if existed else "新建文件"
        return ToolResult(
            True,
            f"已写入 {ctx.rel(path)}：{new_lines} 行，{len(content.encode('utf-8'))} 字节，{note}",
            display=f"{ctx.rel(path)} · {new_lines} 行",
            meta={"changed": ctx.rel(path)},
        )


class EditTool(Tool):
    name = "edit"
    description = "精确替换文件片段，old 必须在文件中唯一"
    params = {"path": "", "old": "", "new": "", "replace_all": "false"}

    def run(self, args, ctx: ToolContext) -> ToolResult:
        path = ctx.resolve(self.arg_str(args, "path"), must_exist=True)
        old = self.arg_str(args, "old")
        new = self.arg_str(args, "new")
        if not old:
            raise ToolError("old 不能为空；如需整体重写请用 write")
        text = read_text(path)
        count = text.count(old)
        if count == 0:
            raise ToolError(f"在 {ctx.rel(path)} 中未找到 old 片段；请先 read 确认原文（注意缩进与空格）")
        replace_all = self.arg_bool(args, "replace_all", False)
        if count > 1 and not replace_all:
            raise ToolError(f"old 在 {ctx.rel(path)} 中匹配 {count} 处，请提供更长的唯一片段或设 replace_all=true")
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        write_text(path, updated)
        line_no = text[: text.index(old)].count("\n") + 1
        context_lines = updated.splitlines()[max(0, line_no - 2) : line_no + 2]
        preview = "\n".join(f"{max(1, line_no - 1) + i:>5}|{line}" for i, line in enumerate(context_lines))
        return ToolResult(
            True,
            f"已修改 {ctx.rel(path)}：替换 {count if replace_all else 1} 处，第 {line_no} 行\n{preview}",
            display=f"{ctx.rel(path)} · 第 {line_no} 行",
            meta={"changed": ctx.rel(path)},
        )


class LsTool(Tool):
    name = "ls"
    description = "列出目录内容（目录加 /，附大小）"
    params = {"path": ".", "limit": "200"}

    def run(self, args, ctx: ToolContext) -> ToolResult:
        path = ctx.resolve(self.arg_str(args, "path", "."), must_exist=True)
        if path.is_file():
            return ToolResult(True, f"{ctx.rel(path)} ({_fmt_size(path.stat().st_size)})")
        limit = max(1, min(500, self.arg_int(args, "limit", 200)))
        entries = []
        try:
            children = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError as exc:
            raise ToolError(f"无法读取目录: {exc}") from exc
        for child in children:
            if child.name in _SKIP_DIRS:
                continue
            if child.is_dir():
                entries.append(f"{child.name}/")
            else:
                try:
                    entries.append(f"{child.name} ({_fmt_size(child.stat().st_size)})")
                except OSError:
                    entries.append(child.name)
            if len(entries) >= limit:
                entries.append("...（已达上限）")
                break
        if not entries:
            return ToolResult(True, f"{ctx.rel(path)} 为空目录", display="空目录")
        return ToolResult(
            True,
            f"{ctx.rel(path)}/\n" + "\n".join(entries),
            display=f"{ctx.rel(path)} · {len(entries)} 项",
        )


class GlobTool(Tool):
    name = "glob"
    description = "按通配符查找文件，如 **/*.py"
    params = {"pattern": "", "path": ".", "limit": "100"}

    def run(self, args, ctx: ToolContext) -> ToolResult:
        pattern = self.arg_str(args, "pattern")
        if not pattern:
            raise ToolError("pattern 不能为空")
        root = ctx.resolve(self.arg_str(args, "path", "."), must_exist=True)
        limit = max(1, min(500, self.arg_int(args, "limit", 100)))
        matches = []
        try:
            for candidate in root.glob(pattern):
                if any(part in _SKIP_DIRS for part in candidate.parts):
                    continue
                if candidate.is_file():
                    matches.append(candidate)
                if len(matches) >= limit * 3:
                    break
        except (ValueError, OSError) as exc:
            raise ToolError(f"glob 失败: {exc}") from exc
        matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        matches = matches[:limit]
        if not matches:
            return ToolResult(True, f"未匹配到文件: {pattern}（root={ctx.rel(root)}）")
        body = "\n".join(ctx.rel(p) for p in matches)
        return ToolResult(True, f"匹配 {len(matches)} 个文件:\n{body}", display=f"{len(matches)} 个文件")


class GrepTool(Tool):
    name = "grep"
    description = "按正则搜索文件内容，返回 文件:行号:内容"
    params = {"pattern": "", "path": ".", "glob": "*", "limit": "50", "ignore_case": "false"}

    def run(self, args, ctx: ToolContext) -> ToolResult:
        pattern = self.arg_str(args, "pattern")
        if not pattern:
            raise ToolError("pattern 不能为空")
        flags = re.IGNORECASE if self.arg_bool(args, "ignore_case", False) else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            raise ToolError(f"正则无效: {exc}") from exc
        root = ctx.resolve(self.arg_str(args, "path", "."), must_exist=True)
        name_glob = self.arg_str(args, "glob", "*") or "*"
        limit = max(1, min(200, self.arg_int(args, "limit", 50)))
        hits: list[str] = []
        files = [root] if root.is_file() else self._walk(root)
        for file in files:
            if not fnmatch.fnmatch(file.name, name_glob):
                continue
            try:
                if file.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = read_text(file)
            except (OSError, ToolError, UnicodeDecodeError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{ctx.rel(file)}:{number}: {line.strip()[:200]}")
                    if len(hits) >= limit:
                        break
            if len(hits) >= limit:
                break
        if not hits:
            return ToolResult(True, f"未找到匹配: {pattern}")
        return ToolResult(True, f"匹配 {len(hits)} 处:\n" + "\n".join(hits), display=f"{len(hits)} 处匹配")

    @staticmethod
    def _walk(root: Path):
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for name in files:
                yield Path(current) / name
