from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.text import Text

from elagent.ui.transcript import (
    Block,
    Transcript,
    final_block,
    render_ansi,
    slice_lines,
    tool_end_block,
    tool_start_block,
    user_block,
    welcome_block,
)

try:
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    HAVE_PTK_SESSION = True
except Exception:  # pragma: no cover
    HAVE_PTK_SESSION = False


def _display_width(text: str) -> int:
    from prompt_toolkit.utils import get_cwidth

    return get_cwidth(text)


class RenderTest(unittest.TestCase):
    def test_ascii_lines_fit_width(self):
        lines = render_ansi(Text("x" * 300), 40)
        self.assertTrue(len(lines) >= 8)
        for line in lines:
            width = sum(_display_width(text) for _, text in line)
            self.assertLessEqual(width, 40)

    def test_cjk_lines_fit_width(self):
        lines = render_ansi(Text("中文字符" * 100), 20)
        self.assertTrue(len(lines) >= 20)
        for line in lines:
            width = sum(_display_width(text) for _, text in line)
            self.assertLessEqual(width, 20)

    def test_panel_keeps_borders(self):
        lines = render_ansi(final_block("# 标题\n内容"), 60)
        joined = "".join(text for line in lines for _, text in line)
        self.assertIn("╭", joined)
        self.assertIn("标题", joined)

    def test_block_caches_by_width(self):
        block = Block(Text("hello"))
        first = block.lines(80)
        self.assertIs(first, block.lines(80))
        self.assertIsNot(first, block.lines(40))


class SliceTest(unittest.TestCase):
    def test_sticks_to_bottom(self):
        lines = [str(i) for i in range(50)]
        chunk, scroll = slice_lines(lines, 10, 0)
        self.assertEqual(len(chunk), 10)
        self.assertEqual(chunk[-1], "49")

    def test_scrolled_keeps_position(self):
        lines = [str(i) for i in range(50)]
        chunk, _ = slice_lines(lines, 10, 5)
        self.assertEqual(chunk[-1], "44")

    def test_clamps_overflow(self):
        lines = [str(i) for i in range(5)]
        chunk, scroll = slice_lines(lines, 10, 100)
        self.assertEqual(len(chunk), 5)
        self.assertEqual(scroll, 0)

    def test_empty(self):
        chunk, scroll = slice_lines([], 10, 0)
        self.assertEqual(chunk, [])
        self.assertEqual(scroll, 0)


class TranscriptTest(unittest.TestCase):
    def test_live_tail_replaced_on_add(self):
        transcript = Transcript()
        transcript.set_live("streaming ...")
        with_live = transcript.lines(80)
        self.assertIn("streaming", "".join(t for line in with_live for _, t in line))
        transcript.add(user_block("done"))
        after = transcript.lines(80)
        joined = "".join(text for line in after for _, text in line)
        self.assertNotIn("streaming", joined)
        self.assertIn("done", joined)

    def test_clear(self):
        transcript = Transcript()
        transcript.add(Text("x"))
        transcript.clear()
        self.assertEqual(transcript.lines(80), [])


class BuildersTest(unittest.TestCase):
    def test_tool_blocks_render(self):
        start = render_ansi(tool_start_block("read", "a.py"), 60)
        end = render_ansi(tool_end_block(True, "120 行", 3), 60)
        joined_start = "".join(text for line in start for _, text in line)
        joined_end = "".join(text for line in end for _, text in line)
        self.assertIn("read", joined_start)
        self.assertIn("a.py", joined_start)
        self.assertIn("✓", joined_end)

    def test_welcome_block(self):
        lines = render_ansi(welcome_block("m", "C:/x", "s", False, ""), 80)
        joined = "".join(text for line in lines for _, text in line)
        self.assertIn("ELAgent", joined)
        self.assertIn("C:/x", joined)


@unittest.skipUnless(HAVE_PTK_SESSION, "prompt_toolkit app session unavailable")
class FullAppSmokeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {"ELAGENT_HOME": str(Path(self._tmp.name) / "home")})
        self._env.start()
        self.cfg_workspace = Path(self._tmp.name) / "ws"
        self.cfg_workspace.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def _make_app(self):
        from elagent.config import Config
        from elagent.ui.fullscreen import FullApp

        cfg = Config(workspace=str(self.cfg_workspace), base_url="http://127.0.0.1:9/v1")
        return FullApp(cfg)

    def _joined(self, app, width=80, height=12):
        return "".join(text for _, text in app.visible_fragments(width, height))

    def test_build_and_render_headless(self):
        app = self._make_app()
        app.write("hello")
        app.write_renderable(user_block("任务"))
        fragments = app.visible_fragments(80, 10)
        self.assertTrue(fragments)
        joined = "".join(text for _, text in fragments)
        self.assertIn("hello", joined)
        self.assertIn("任务", joined)

    def test_thinking_expanded_by_default(self):
        from elagent.events import Event

        app = self._make_app()
        self.assertTrue(app.show_thinking)
        app._handle_event(Event("reasoning", {"text": "先分析目录结构再动手"}))
        self.assertIn("思考", self._joined(app))
        app._handle_event(Event("turn_end", {"text": "", "reasoning": "结论是先读文件"}))
        joined = self._joined(app)
        self.assertIn("先读文件", joined)
        app.show_thinking = False
        app._handle_event(Event("reasoning", {"text": "不应该出现这段"}))
        self.assertNotIn("不应该出现这段", self._joined(app))

    def test_memory_save_requires_finished_task(self):
        app = self._make_app()
        app.record_memory()
        joined = "".join(text for line in app.transcript.lines(80) for _, text in line)
        self.assertIn("还没有已完成的任务", joined)
        # a finished task arms /memory save
        app._task_done(ok=True, final="已创建文件并验证", task="创建文件")
        self.assertEqual(app.last_task, "创建文件")

    def test_ask_user_bridges_input(self):
        app = self._make_app()
        result: dict[str, str | None] = {}

        def worker() -> None:
            result["answer"] = app.ask_user("喜欢什么颜色？")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        time.sleep(0.2)

        class FakeBuffer:
            text = "蓝色"

        app._accept(FakeBuffer())
        thread.join(timeout=2)
        self.assertEqual(result.get("answer"), "蓝色")
        self.assertIsNone(app.pending_question)

    def test_each_launch_starts_new_session(self):
        first = self._make_app()
        first.session.title = "keep"
        first.store.save(first.session)
        second = self._make_app()
        self.assertNotEqual(first.session.id, second.session.id)
        self.assertFalse(second.resumed)

    def test_resume_flag_loads_latest_session(self):
        from elagent.config import Config
        from elagent.ui.fullscreen import FullApp

        first = self._make_app()
        first.session.title = "old"
        first.store.save(first.session)
        cfg = Config(workspace=str(self.cfg_workspace), base_url="http://127.0.0.1:9/v1")
        resumed = FullApp(cfg, resume=True)
        self.assertEqual(resumed.session.id, first.session.id)
        self.assertTrue(resumed.resumed)

    def test_accept_dispatches_command(self):
        app = self._make_app()

        class FakeBuffer:
            text = "/help"

        # commands clear the input buffer (return False) and render output
        self.assertFalse(app._accept(FakeBuffer()))
        joined = "".join(text for line in app.transcript.lines(80) for _, text in line)
        self.assertIn("/goal", joined)

    def test_run_and_quit_via_keys(self):
        app = self._make_app()

        def feeder(pipe):
            time.sleep(0.8)
            pipe.send_text("/help\r")
            time.sleep(0.8)
            pipe.send_text("/quit\r")

        from elagent.config import Config  # noqa: F401

        with create_pipe_input() as pipe:
            thread = threading.Thread(target=feeder, args=(pipe,), daemon=True)
            thread.start()
            with create_app_session(input=pipe, output=DummyOutput()):
                app.run()
            thread.join(timeout=3)
        self.assertTrue(app.quit)
        joined = "".join(text for line in app.transcript.lines(80) for _, text in line)
        self.assertIn("/goal", joined)


if __name__ == "__main__":
    unittest.main()
