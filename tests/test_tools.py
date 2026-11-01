from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from elagent.config import Config
from elagent.tools import ToolContext, build_registry
from elagent.tools.base import ToolError
from elagent.tools.shell import find_bash


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._env = patch.dict(os.environ, {"ELAGENT_HOME": str(self.root / "home")})
        self._env.start()
        self.cfg = Config(workspace=str(self.root), shell="bash" if find_bash() else "cmd")
        self.ctx = ToolContext(self.root, self.cfg, "test-session")
        self.tools = build_registry()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def run_tool(self, name, **args):
        return self.tools.execute(name, args, self.ctx)

    def test_write_read(self):
        result = self.run_tool("write", path="src/a.py", content="print('hi')\n")
        self.assertTrue(result.ok, result.content)
        read = self.run_tool("read", path="src/a.py")
        self.assertIn("1|print('hi')", read.content)

    def test_edit_unique(self):
        self.run_tool("write", path="a.txt", content="alpha\nbeta\ngamma\n")
        result = self.run_tool("edit", path="a.txt", old="beta", new="BETA")
        self.assertTrue(result.ok, result.content)
        self.assertIn("BETA", (self.root / "a.txt").read_text(encoding="utf-8"))

    def test_edit_ambiguous_is_error(self):
        self.run_tool("write", path="a.txt", content="x\nx\n")
        result = self.run_tool("edit", path="a.txt", old="x", new="y")
        self.assertFalse(result.ok)
        self.assertIn("2 处", result.content)

    def test_edit_missing_fragment(self):
        self.run_tool("write", path="a.txt", content="abc")
        result = self.run_tool("edit", path="a.txt", old="zzz", new="y")
        self.assertFalse(result.ok)
        self.assertIn("未找到", result.content)

    def test_path_escape_blocked(self):
        with self.assertRaises(ToolError):
            self.ctx.resolve("../outside.txt")
        result = self.run_tool("read", path="../outside.txt")
        self.assertFalse(result.ok)
        self.assertIn("越界", result.content)

    def test_glob_and_grep(self):
        self.run_tool("write", path="pkg/a.py", content="TODO: fix\n")
        self.run_tool("write", path="pkg/b.txt", content="TODO: doc\n")
        glob = self.run_tool("glob", pattern="**/*.py")
        self.assertIn("pkg/a.py", glob.content)
        grep = self.run_tool("grep", pattern="TODO", glob="*.py")
        self.assertIn("pkg/a.py:1", grep.content)
        self.assertNotIn("b.txt", grep.content)

    def test_ls_skips_noise(self):
        (self.root / "node_modules").mkdir()
        self.run_tool("write", path="main.py", content="x=1")
        result = self.run_tool("ls", path=".")
        self.assertIn("main.py", result.content)
        self.assertNotIn("node_modules", result.content)

    def test_todo_roundtrip(self):
        result = self.run_tool("todo", action="set", items=["读代码", "改 bug"])
        self.assertIn("[ ] 1. 读代码", result.content)
        result = self.run_tool("todo", action="done", id=1)
        self.assertIn("[x] 1. 读代码", result.content)

    def test_time_tool(self):
        from datetime import datetime

        result = self.run_tool("time")
        self.assertTrue(result.ok, result.content)
        self.assertIn(str(datetime.now().year), result.content)
        self.assertIn("时区", result.content)

    def test_ask_tool_with_ui(self):
        self.ctx.ask_user = lambda question: "蓝色"
        result = self.run_tool("ask", question="喜欢什么颜色？")
        self.assertTrue(result.ok, result.content)
        self.assertIn("蓝色", result.content)

    def test_ask_tool_without_ui(self):
        result = self.run_tool("ask", question="喜欢什么颜色？")
        self.assertFalse(result.ok)
        self.assertIn("非交互", result.content)

    def test_subagent_tool_uses_runner(self):
        calls: list[tuple[str, int]] = []

        def runner(task: str, steps: int):
            calls.append((task, steps))
            return True, "子代理结果", ["a.py"]

        self.ctx.subagent = runner
        result = self.run_tool("subagent", task="独立子任务", max_steps=3)
        self.assertTrue(result.ok, result.content)
        self.assertIn("子代理结果", result.content)
        self.assertEqual(calls, [("独立子任务", 3)])
        self.assertEqual(result.meta["changed_files"], ["a.py"])

    def test_workflow_tool_uses_runner(self):
        seen: list[list[str]] = []

        def runner(steps: list[str]):
            seen.append(list(steps))
            return True, "；".join(steps), ["b.py"]

        self.ctx.workflow = runner
        result = self.run_tool("workflow", steps=["第一步", "第二步"])
        self.assertTrue(result.ok, result.content)
        self.assertEqual(seen, [["第一步", "第二步"]])
        self.assertIn("第一步", result.content)
        self.assertEqual(result.meta["changed_files"], ["b.py"])

    def test_missing_path_suggests_close_name(self):
        (self.root / ".hidden").mkdir()
        result = self.run_tool("ls", path="hidden")
        self.assertFalse(result.ok)
        self.assertIn(".hidden", result.content)
        self.assertIn("是否指", result.content)

    @unittest.skipUnless(find_bash(), "Git Bash not available")
    def test_shell_success_and_failure(self):
        ok = self.run_tool("terminalbash", command="echo hello")
        self.assertTrue(ok.ok, ok.content)
        self.assertIn("hello", ok.content)
        bad = self.run_tool("terminalbash", command="exit 3")
        self.assertFalse(bad.ok)
        self.assertIn("exit=3", bad.content)

    @unittest.skipUnless(find_bash(), "Git Bash not available")
    def test_shell_timeout(self):
        result = self.run_tool("terminalbash", command="sleep 5", timeout=1)
        self.assertFalse(result.ok)
        self.assertIn("超时", result.content)


if __name__ == "__main__":
    unittest.main()
