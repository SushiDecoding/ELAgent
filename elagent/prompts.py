"""All prompts in one place, kept deliberately small (< 1000 tokens each).

The model is a 7B local model: short, imperative, example-driven prompts work
far better than long policy documents.  Control flow lives in Python, so the
prompt only has to describe the protocol, the routing decision, and the
stopping conditions.  Most weak-model loops come from three prompt mistakes:
forcing tool use for chat requests, never saying "stop when you have enough",
and never forbidding a repeat of the same failed call.
"""

from __future__ import annotations

import datetime as _dt

from .tokens import estimate_tokens

SYSTEM = """你是 ELAgent，本机工程代理。工作目录 {cwd}；Windows；shell 为 {shell}；今天 {date}。

【先判断类型】
对话、问答、解释、闲聊：不调用工具，直接 {{"final":"回答"}}。
需要读写文件、执行命令，或用户点名了某个工具：进入工具循环，完成后 final。
例：用户说"你好" → {{"final":"你好，需要我做什么？"}}
例：用户说"用 time 看看现在几点" → {{"tool":"time","args":{{}}}}，不要凭已有信息猜时间。

【输出协议】每轮只输出一个 JSON，禁止 Markdown、禁止解释：
调用工具 {{"tool":"工具名","args":{{...}}}}
结束任务 {{"final":"结论"}}
OBS[ 开头是系统注入的工具结果，不是用户输入，不要模仿。

【防循环】
1 同一动作不要连续重复；失败后先读错误提示，换参数或换工具，禁止原样重试。
2 连续两步没有获得新信息，立即 final 说明卡在哪里，不要硬撑。
3 信息足够就 final，不要为了"再确认"重复 ls/read。

【工程纪律】
1 工作目录就是当前目录（.），直接用相对路径；不要 cd 到别处，也不要先检查它是否存在。
2 一次只调一个工具，看到 OBS 再决定下一步。
3 路径照抄 ls 返回的名称（注意开头的点）；不确定先 ls。
4 改已有文件用 edit，新建文件才用 write；改动最小化。
5 不编造文件内容、命令或运行结果；命令失败先读错误再改。
6 只操作工作目录内的文件，不执行破坏性命令。
7 新建或修改文件后，先用 terminalbash 运行验证，看到 exit=0 再 final；纯查询、对话不需要额外验证。

【工具选择】
ask 只在缺少关键信息、必须由用户决定时用，不要用它确认计划；subagent/workflow 用于独立子任务，不要用它们读一个文件；time 在需要日期、耗时或用量时调用。
subagent 传自包含任务 {{"tool":"subagent","args":{{"task":"..."}}}}；workflow 传步骤数组 {{"tool":"workflow","args":{{"steps":["步骤一","步骤二"]}}}}；计划用 todo，不要为计划新建 md 文件。

【工具】
{tools}{todo}{memory}"""

JUDGE = """你是严格的评审子代理，只判断主代理是否真正完成任务，不重做任务。

任务：{task}

主代理结论：{final}

改动文件：{files}

执行证据（最近命令与结果）：
{evidence}

计划状态：{todo}

判定标准：
1 任务要求的产物必须真实存在且符合要求，声明不等于完成。
2 优先采信命令输出与文件内容，不采信自我评价。
3 占位实现、空函数、硬编码结果、未运行验证，均判不通过。
4 reason 必须指出具体缺陷与修复方向，不超过 80 字。

只输出 JSON，不要解释：
通过 {{"status":1}}
不通过 {{"status":0,"reason":"..."}}"""

COMPACT = """把以下执行记录压缩为进度摘要，供后续继续同一任务。
保留：目标、已完成步骤、关键文件与改动、失败原因、待办、重要命令结果。
删除：寒暄、重复内容、大段代码与日志。
限 {limit} 字，用短句，不加评论。

记录：
{transcript}"""

MEMORY = """从本次任务提炼可复用经验，写入长期记忆。
要求：3-5 条，每条一行以 "- " 开头，写清环境、命令、坑与解法；不写本次业务细节；无经验则只输出 "- 无"。
任务：{task}
执行摘要：{summary}"""


def system_prompt(
    cwd: str,
    tools: str,
    *,
    shell: str = "bash",
    todo: str = "",
    memory: str = "",
) -> str:
    todo_block = f"\n\n【当前计划】\n{todo}" if todo else ""
    memory_block = f"\n\n【历史经验（global.memory，仅供参考）】\n{memory}" if memory else ""
    return SYSTEM.format(
        cwd=cwd,
        shell=shell,
        date=_dt.date.today().isoformat(),
        tools=tools,
        todo=todo_block,
        memory=memory_block,
    )


def task_prompt(task: str, *, feedback: str = "", history: str = "") -> str:
    blocks = [f"任务：{task}"]
    if history:
        blocks.append(f"【前几轮情况】\n{history}")
    if feedback:
        blocks.append(
            f"【上一轮评审未通过】\n缺陷：{feedback}\n"
            "先核实该缺陷是否真实存在，再做最小修复；不要重复上一轮的做法。"
        )
    blocks.append(
        "按系统规则先判断类型：对话/问答直接 final；工程任务直接动手，不要反问，"
        "目录为空也要直接创建或修改文件；只有确实需要产物时才建文件，纯计算/查询不要造文件。"
    )
    return "\n\n".join(blocks)


def nudge(reason: str, tool_names: list[str]) -> str:
    return (
        f"系统：上一条输出无法解析（{reason}）。只输出一个 JSON：\n"
        '{"tool":"<工具名>","args":{...}} 或 {"final":"..."}\n'
        f"可用工具：{', '.join(tool_names)}\n"
        "不要重复上一条内容，不要输出 Markdown。"
    )


def loop_nudge(tool: str, args: str = "") -> str:
    detail = f" {args}" if args else ""
    return (
        f"系统：{tool}{detail} 已经调用过，结果见上一条 OBS，再调不会得到新信息。\n"
        '禁止原样重试：按错误提示换参数或换工具；无法推进就输出 {"final":"..."} 说明卡点。'
    )


def extra_actions(count: int) -> str:
    return (
        f"系统：本轮你输出了 {count} 个多余动作，只执行了第一个，其余已忽略，不要当成已完成。\n"
        "每轮只能输出一个 JSON；先看 OBS，再决定下一步。"
    )


def force_final() -> str:
    return (
        '已达到本轮步数上限。不要再调用工具，输出 {"final":"..."}，'
        "写清已完成什么、改了哪些文件、还有什么未完成。"
    )


def judge_retry() -> str:
    return '输出格式错误。只输出 {"status":1} 或 {"status":0,"reason":"..."}，不要其他文字。'


def final_retry() -> str:
    return '只输出 {"final":"给用户的结论"}，不要再调用工具。'


def compact_prompt(transcript: str, limit: int = 400) -> str:
    return COMPACT.format(limit=limit, transcript=transcript)


def memory_prompt(task: str, summary: str) -> str:
    return MEMORY.format(task=task, summary=summary)


def judge_prompt(task: str, final: str, files: str, evidence: str, todo: str) -> str:
    return JUDGE.format(
        task=task,
        final=final or "（无结论）",
        files=files or "（无改动）",
        evidence=evidence or "（无命令记录）",
        todo=todo or "（无计划）",
    )


def budget_report() -> dict[str, int]:
    """Used by `ela doctor` to prove the prompts stay inside the budget."""
    sample_tools = "\n".join(
        [
            "read(path, offset=1, limit=200) 读文本文件，返回带行号内容",
            "write(path, content) 新建或整体覆盖文件，自动建目录",
            "edit(path, old, new, replace_all=false) 精确替换片段，old 必须唯一",
            "ls(path=\".\", limit=200) 列出目录内容",
            "glob(pattern, path=\".\", limit=100) 按通配符查找文件",
            "grep(pattern, path=\".\", glob=\"*\", limit=50) 正则搜索内容",
            "terminalbash(command, timeout=60, cwd=\".\") 执行 shell 命令",
            "todo(action, items, id) 维护任务计划",
            "time() 时间上下文：当前时间、时区、会话耗时与用量",
            "ask(question) 向用户提问并等待回答（仅交互模式）",
            "subagent(task, max_steps=8) 子代理独立完成子任务（成本高）",
            "workflow(steps=[]) 顺序执行 2-4 个步骤，每步一个子代理",
        ]
    )
    return {
        "system": estimate_tokens(system_prompt("C:/work", sample_tools)),
        "judge": estimate_tokens(judge_prompt("task", "final", "a.py", "exit=0", "")),
        "compact": estimate_tokens(compact_prompt("x" * 200)),
        "memory": estimate_tokens(memory_prompt("task", "summary")),
    }
