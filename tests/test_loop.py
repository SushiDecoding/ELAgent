from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from elagent.agent import GoalRunner, MainAgent
from elagent.config import Config
from elagent.context import ContextManager
from elagent.events import EventBus
from elagent.llm import LLMResponse
from elagent.session import SessionStore
from elagent.tools import build_registry

from mock_llm import MockLLM


class LoopTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._env = patch.dict(os.environ, {"ELAGENT_HOME": str(self.root / "home")})
        self._env.start()
        self.cfg = Config(
            workspace=str(self.root),
            max_steps=6,
            max_rounds=-1,
            compact_threshold=100000,
            tool_output_limit=4000,
        )
        self.bus = EventBus()
        self.events: list[tuple[str, dict]] = []
        self.bus.subscribe(lambda e: self.events.append((e.kind, e.data)))
        self.tools = build_registry()
        self.store = SessionStore(Path(self.root / "home"))
        self.session = self.store.new(str(self.root), "mock")

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def agent(self, llm):
        return MainAgent(self.cfg, llm, self.tools, self.bus, self.session)

    # ------------------------------------------------------------------
    def test_tool_then_final(self):
        llm = MockLLM(
            [
                '{"tool":"write","args":{"path":"hello.py","content":"print(1)\\n"}}',
                '{"final":"已创建 hello.py"}',
            ]
        )
        result = self.agent(llm).run("创建 hello.py")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.tool_calls, 1)
        self.assertTrue((self.root / "hello.py").exists())
        self.assertIn("hello.py", result.changed_files)
        self.assertIn("OBS[write] ok", llm.calls[1][-1]["content"])

    def test_invalid_then_recover(self):
        llm = MockLLM(
            [
                '{"tool": }',
                '{"tool":"ls","args":{"path":"."}}',
                '{"final":"完成"}',
            ]
        )
        result = self.agent(llm).run("看看目录")
        self.assertTrue(result.ok)
        self.assertEqual(result.tool_calls, 1)
        self.assertTrue(any(kind == "notice" for kind, _ in self.events))

    def test_unknown_tool_is_recoverable(self):
        llm = MockLLM(
            [
                '{"tool":"shell","args":{"command":"ls"}}',
                '{"final":"改用 ls 完成"}',
            ]
        )
        result = self.agent(llm).run("列出目录")
        self.assertTrue(result.ok)
        nudge = llm.calls[1][-1]["content"]
        self.assertIn("无法解析", nudge)
        self.assertIn("terminalbash", nudge)

    def test_max_invalid_degrades(self):
        self.cfg.max_invalid_actions = 2
        llm = MockLLM(['{"tool": }', '{"tool": }', '{"final":"无法继续"}'])
        result = self.agent(llm).run("做点什么")
        self.assertFalse(result.ok)
        self.assertEqual(result.final, "无法继续")

    def test_step_limit_synthesises_final(self):
        self.cfg.max_steps = 2
        llm = MockLLM(
            [
                '{"tool":"ls","args":{}}',
                '{"tool":"ls","args":{}}',
                "仍然无法解析的输出 {",
            ]
        )
        result = self.agent(llm).run("无限循环测试")
        self.assertFalse(result.ok)
        self.assertIn("达到本轮步数上限", result.final)

    def test_goal_retry_then_pass(self):
        llm = MockLLM(
            [
                '{"final":"v1 完成"}',
                '{"status":0,"reason":"缺少运行验证"}',
                '{"final":"v2 已补测试"}',
                '{"status":1}',
            ]
        )
        runner = GoalRunner(self.cfg, llm, self.tools, self.bus, self.session, self.store)
        result = runner.run("实现功能")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.reasons, ["缺少运行验证"])
        second_round_prompt = llm.calls[2][-1]["content"]
        self.assertIn("缺少运行验证", second_round_prompt)

    def test_goal_max_rounds(self):
        self.cfg.max_rounds = 1
        llm = MockLLM(['{"final":"v1"}', '{"status":0,"reason":"还不行"}'])
        runner = GoalRunner(self.cfg, llm, self.tools, self.bus, self.session, self.store)
        result = runner.run("任务")
        self.assertFalse(result.ok)
        self.assertEqual(result.rounds, 1)
        self.assertEqual(result.error, "达到最大轮数")

    def test_goal_judge_garbage_aborts(self):
        self.cfg.max_rounds = -1
        self.cfg.judge_retries = 1
        llm = MockLLM(
            [
                '{"final":"v1"}', "说不好",
                '{"final":"v2"}', "还是说不好",
                '{"final":"v3"}', "依然说不好",
            ]
        )
        runner = GoalRunner(self.cfg, llm, self.tools, self.bus, self.session, self.store)
        result = runner.run("任务")
        self.assertFalse(result.ok)
        self.assertIn("非法 JSON", result.error)
        self.assertEqual(result.rounds, 3)

    def test_history_gets_task_appended(self):
        llm = MockLLM(['{"final":"好的"}'])
        history: list[dict] = []
        result = self.agent(llm).run("第一个任务", messages=history)
        self.assertTrue(result.ok)
        sent = [m["content"] for m in llm.calls[0]]
        self.assertTrue(any("第一个任务" in text for text in sent))
        self.assertEqual(history[-1]["role"], "assistant")

        llm2 = MockLLM(['{"final":"第二个完成"}'])
        self.agent(llm2).run("第二个任务", messages=history)
        sent2 = [m["content"] for m in llm2.calls[0]]
        self.assertTrue(any("第二个任务" in text for text in sent2))
        self.assertTrue(any("第一个任务" in text for text in sent2))

    def test_truncated_final_is_rejected(self):
        agent = self.agent(MockLLM([]))
        action = agent._parse(LLMResponse(text='{"final":"半截结论"}', truncated=True), ["ls"])
        self.assertEqual(action.kind, "invalid")

    def test_truncated_tool_call_is_rejected(self):
        agent = self.agent(MockLLM([]))
        action = agent._parse(
            LLMResponse(text='{"tool":"write","args":{"path":"a.py","content":"x"}}', truncated=True),
            ["write"],
        )
        self.assertEqual(action.kind, "invalid")

    def test_goal_skips_judge_when_no_final(self):
        class Broken(MockLLM):
            def chat(self, messages, **kwargs):
                self.calls.append(list(messages))
                raise RuntimeError("upstream down")

        self.cfg.max_rounds = 1
        llm = Broken([])
        runner = GoalRunner(self.cfg, llm, self.tools, self.bus, self.session, self.store)
        result = runner.run("任务")
        self.assertFalse(result.ok)
        self.assertEqual(len(llm.calls), 1)  # judge must not be consulted

    def test_subagent_tool_runs_child(self):
        self.cfg.max_steps = 6
        llm = MockLLM(
            [
                '{"tool":"subagent","args":{"task":"列出文件"}}',
                '{"final":"子代理完成"}',
                '{"final":"父完成"}',
            ]
        )
        result = self.agent(llm).run("用子代理做事")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.final, "父完成")
        self.assertIn("子代理完成", llm.calls[2][-1]["content"])

    def test_subagent_changed_files_propagate(self):
        self.cfg.max_steps = 6
        llm = MockLLM(
            [
                '{"tool":"subagent","args":{"task":"创建 a.py"}}',
                '{"tool":"write","args":{"path":"a.py","content":"x=1"}}',
                '{"final":"子代理完成"}',
                '{"final":"父完成"}',
            ]
        )
        result = self.agent(llm).run("让子代理创建文件")
        self.assertTrue(result.ok, result.error)
        self.assertIn("a.py", result.changed_files)

    def test_workflow_tool_runs_steps(self):
        self.cfg.max_steps = 6
        llm = MockLLM(
            [
                '{"tool":"workflow","args":{"steps":["步骤一","步骤二"]}}',
                '{"final":"一完成"}',
                '{"final":"二完成"}',
                '{"final":"父完成"}',
            ]
        )
        result = self.agent(llm).run("用工作流做事")
        self.assertTrue(result.ok, result.error)
        self.assertIn("一完成", llm.calls[3][-1]["content"])
        self.assertIn("二完成", llm.calls[3][-1]["content"])

    def test_repeat_action_stops_round(self):
        self.cfg.max_steps = 10
        llm = MockLLM(
            [
                '{"tool":"ls","args":{"path":"."}}',
                '{"tool":"ls","args":{"path":"."}}',
                '{"tool":"ls","args":{"path":"."}}',
                '{"final":"卡在重复动作"}',
            ]
        )
        result = self.agent(llm).run("看看目录")
        self.assertFalse(result.ok)
        self.assertIn("重复动作", result.final)
        self.assertLessEqual(result.steps, 3)
        self.assertTrue(any(kind == "notice" for kind, _ in self.events))

    def test_repeat_resets_after_file_change(self):
        self.cfg.max_steps = 6
        llm = MockLLM(
            [
                '{"tool":"ls","args":{"path":"."}}',
                '{"tool":"write","args":{"path":"a.txt","content":"x"}}',
                '{"tool":"ls","args":{"path":"."}}',
                '{"final":"完成"}',
            ]
        )
        result = self.agent(llm).run("写文件后确认")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.tool_calls, 3)

    # ------------------------------------------------------------------
    def test_compaction_reduces_context(self):
        messages = [{"role": "system", "content": "sys"}]
        for index in range(20):
            role = "assistant" if index % 2 == 0 else "user"
            messages.append({"role": role, "content": f"message {index} " + "x" * 200})
        cfg = Config(keep_recent_messages=4, compact_threshold=10)
        llm = MockLLM(["摘要：" + "关键进展" * 10])
        ctx = ContextManager(messages, cfg, llm, self.bus)
        before = ctx.tokens()
        self.assertTrue(ctx.compact())
        self.assertLess(ctx.tokens(), before)
        self.assertIn("【进度摘要】", messages[1]["content"])
        self.assertEqual(messages[0]["role"], "system")

    def test_compaction_falls_back_when_llm_fails(self):
        class Broken(MockLLM):
            def chat(self, messages, **kwargs):
                raise RuntimeError("down")

        messages = [{"role": "system", "content": "sys"}]
        for index in range(20):
            messages.append({"role": "user", "content": f"m{index}"})
        cfg = Config(keep_recent_messages=4)
        ctx = ContextManager(messages, cfg, Broken([]), self.bus)
        self.assertTrue(ctx.compact(force=True))
        self.assertIn("自动压缩", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
