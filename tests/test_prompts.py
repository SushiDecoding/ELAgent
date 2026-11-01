from __future__ import annotations

import unittest

from elagent.prompts import budget_report, loop_nudge, system_prompt, task_prompt


class PromptTest(unittest.TestCase):
    def test_system_prompt_has_routing_and_anti_loop_rules(self):
        text = system_prompt("C:/work", "read(path)")
        self.assertIn("【先判断类型】", text)
        self.assertIn("对话", text)
        self.assertIn("禁止原样重试", text)
        self.assertIn("信息足够就 final", text)
        self.assertIn('{"final"', text)

    def test_task_prompt_is_type_aware(self):
        text = task_prompt("你好")
        self.assertIn("对话", text)
        self.assertIn("工程任务", text)
        # The old unconditional "start with a tool" line caused chat loops.
        self.assertNotIn("空目录不是停止的理由", text)

    def test_loop_nudge_names_the_action(self):
        text = loop_nudge("ls", '{"path": "./agents-anywhere"}')
        self.assertIn("ls", text)
        self.assertIn("禁止原样重试", text)

    def test_prompt_budget(self):
        self.assertLessEqual(max(budget_report().values()), 1000)


if __name__ == "__main__":
    unittest.main()
