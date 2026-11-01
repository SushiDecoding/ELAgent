from __future__ import annotations

import unittest

from elagent.protocol import parse_action, parse_verdict, repair_json, scan_objects

TOOLS = {"read", "write", "edit", "ls", "glob", "grep", "terminalbash", "todo"}


class ActionParsingTest(unittest.TestCase):
    def test_canonical(self):
        action = parse_action('{"tool":"read","args":{"path":"a.py"}}', TOOLS)
        self.assertEqual(action.kind, "tool")
        self.assertEqual(action.tool, "read")
        self.assertEqual(action.args["path"], "a.py")

    def test_fenced(self):
        text = '```json\n{"tool": "ls", "args": {"path": "."}}\n```'
        action = parse_action(text, TOOLS)
        self.assertEqual((action.kind, action.tool), ("tool", "ls"))

    def test_python_literals(self):
        action = parse_action("{'tool': 'write', 'args': {'path': 'a.txt', 'content': 'hi'}}", TOOLS)
        self.assertEqual(action.kind, "tool")
        self.assertEqual(action.args["content"], "hi")

    def test_trailing_comma_and_prose(self):
        text = '我先看看目录。\n{"tool":"ls","args":{"path":".",},}\n然后继续。'
        action = parse_action(text, TOOLS)
        self.assertEqual(action.tool, "ls")

    def test_missing_closing_brace(self):
        action = parse_action('{"tool":"read","args":{"path":"a.py"', TOOLS)
        self.assertEqual(action.kind, "tool")
        self.assertEqual(action.tool, "read")

    def test_openai_style(self):
        text = '{"tool_calls":[{"function":{"name":"grep","arguments":"{\\"pattern\\":\\"TODO\\"}"}}]}'
        action = parse_action(text, TOOLS)
        self.assertEqual(action.tool, "grep")
        self.assertEqual(action.args["pattern"], "TODO")

    def test_function_call_style(self):
        action = parse_action('read(path="a.py", limit=10)', TOOLS)
        self.assertEqual(action.kind, "tool")
        self.assertEqual(action.args["path"], "a.py")

    def test_shorthand(self):
        action = parse_action('{"glob": {"pattern": "**/*.py"}}', TOOLS)
        self.assertEqual(action.tool, "glob")

    def test_final(self):
        action = parse_action('{"final":"已完成，修改了 a.py"}', TOOLS)
        self.assertEqual(action.kind, "final")
        self.assertIn("a.py", action.text)

    def test_final_object(self):
        action = parse_action('{"final":{"时间":"20:28","用量":"0 tok"}}', TOOLS)
        self.assertEqual(action.kind, "final")
        self.assertIn("20:28", action.text)

    def test_plain_text_final(self):
        action = parse_action("这个任务不需要改代码，项目结构已经符合要求。", TOOLS)
        self.assertEqual(action.kind, "final")
        self.assertEqual(action.reason, "plain_text")

    def test_unknown_tool(self):
        action = parse_action('{"tool":"shell","args":{"command":"ls"}}', TOOLS)
        self.assertEqual(action.kind, "invalid")
        self.assertIn("unknown_tool", action.reason)

    def test_tool_call_xml(self):
        text = (
            "<tool_call>ls <arg_key>path</arg_key> <arg_value>.</arg_value>"
            "<arg_key>limit</arg_key><arg_value>100</arg_value></tool_call>"
        )
        action = parse_action(text, TOOLS)
        self.assertEqual((action.kind, action.tool), ("tool", "ls"))
        self.assertEqual(action.args["path"], ".")
        self.assertEqual(action.args["limit"], 100)

    def test_tool_call_xml_function_style(self):
        text = '<tool_call>terminalbash(command="ls -la", timeout=10)</tool_call>'
        action = parse_action(text, TOOLS)
        self.assertEqual((action.kind, action.tool), ("tool", "terminalbash"))
        self.assertEqual(action.args["command"], "ls -la")
        self.assertEqual(action.args["timeout"], 10)

    def test_tool_call_xml_json_body(self):
        action = parse_action('<tool_call>read {"path": "a.py"}</tool_call>', TOOLS)
        self.assertEqual((action.kind, action.tool), ("tool", "read"))
        self.assertEqual(action.args["path"], "a.py")

    def test_unparsed_tool_attempt_is_not_final(self):
        text = "<tool_call>nope <arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
        action = parse_action(text, TOOLS)
        self.assertEqual(action.kind, "invalid")

    def test_broken_json(self):
        action = parse_action('{"tool": }', TOOLS)
        self.assertEqual(action.kind, "invalid")

    def test_multiple_actions_keeps_first_and_counts(self):
        text = (
            '{"tool":"write","args":{"path":"a.py","content":"x"}}\n'
            '{"tool":"terminalbash","args":{"command":"python a.py"}}'
        )
        action = parse_action(text, TOOLS)
        self.assertEqual((action.kind, action.tool), ("tool", "write"))
        self.assertEqual(action.extra, 1)

    def test_multiple_objects_takes_first_valid(self):
        text = '{"foo": 1}\n{"tool":"ls","args":{}}'
        action = parse_action(text, TOOLS)
        self.assertEqual(action.tool, "ls")

    def test_newlines_inside_string(self):
        text = '{"tool":"write","args":{"path":"a.txt","content":"line1\nline2"}}'
        action = parse_action(text, TOOLS)
        self.assertEqual(action.kind, "tool")
        self.assertEqual(action.args["content"], "line1\nline2")


class RepairTest(unittest.TestCase):
    def test_scan_ignores_braces_in_strings(self):
        found = scan_objects('x {"a":"} not end"} y')
        self.assertEqual(len(found), 1)

    def test_repair_balances(self):
        self.assertEqual(repair_json('{"a": [1, 2'), '{"a": [1, 2]}')


class VerdictTest(unittest.TestCase):
    def test_pass(self):
        verdict = parse_verdict('{"status":1}')
        self.assertEqual(verdict.status, 1)

    def test_fail(self):
        verdict = parse_verdict('{"status":0,"reason":"缺少边界处理"}')
        self.assertEqual(verdict.status, 0)
        self.assertIn("边界", verdict.reason)

    def test_fenced_and_text(self):
        verdict = parse_verdict('评审结果：\n```json\n{"status": 0, "reason": "未运行测试"}\n```')
        self.assertEqual(verdict.status, 0)
        self.assertIn("测试", verdict.reason)

    def test_string_status(self):
        self.assertEqual(parse_verdict('{"status":"pass"}').status, 1)
        self.assertEqual(parse_verdict('{"status":"fail"}').status, 0)

    def test_chinese(self):
        self.assertEqual(parse_verdict("结论：不通过，存在缺陷").status, 0)

    def test_garbage(self):
        verdict = parse_verdict("我觉得还行吧")
        self.assertEqual(verdict.status, -1)
        self.assertTrue(verdict.parse_error)


if __name__ == "__main__":
    unittest.main()
