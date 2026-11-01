# ELAgent

面向本地小模型（蚂蚁百灵 **Ling-3.0-tiny**）的长程任务 Agent 框架。

设计前提很明确：**7B 级本地模型不可靠**。它会漏掉任务、编造工具结果、输出坏 JSON、把推理预算烧光。
因此 ELAgent 把循环、裁剪、解析、重试、轮数控制、评审判定全部写在 Python 里，模型只负责"下一步做什么"。

```
ela                          # 进入全屏 TUI
ela -p "任务"                # 标准模式执行一次
ela -p "任务" --goal         # 目标模式：主 Agent + 评审子 Agent 循环
ela --goal --rounds 5 -p "…" # 限制最多 5 轮
```

---

## 0. 全屏界面

默认进入 Pi 风格的全屏 TUI（备用屏幕缓冲区，退出后还原终端）：

```
┌ ELAgent 0.1.0 · 模型 · 工作目录 · 会话 ┐   ← 固定头部
│                                        │
│            对话区（可滚动）             │   ← PageUp/PageDown、Ctrl+Home/End、滚轮
│                                        │
└────────────────────────────────────────┘
 标准 │ ⠋ 生成中 1.2s · 240 tok │ ctx 3% │ … ← 状态栏（含流式计数与思考量）
 目标 › █                                  ← 单行输入（/ 命令补全）
```

- 模型在后台线程运行，界面永不冻结；`Esc`/`Ctrl+C` 中断当前任务。
- 流式 token 实时计入状态栏；**思考内容默认展开**（流式淡显 + 回合后保留），`/thinking` 收起。
- 工具调用逐条渲染，`/verbose` 展开完整输出。
- 危险命令确认使用底部模态条（Y/N/Esc），不弹系统对话框。
- 无控制台环境（管道、CI）自动降级为行式界面，功能不变。
- **每次启动默认新建会话**；需要旧会话时用 `ela --resume`，或在 TUI 里 `/resume`。
---

## 1. 参考架构分析

### Pi agent（TUI 与指令系统）

| 设计点 | Pi 的做法 | ELAgent 的取舍 |
|---|---|---|
| 定位 | 极简 harness，能力靠扩展 | 同样保持核心小，工具与 UI 解耦 |
| 会话 | JSONL 树，可 `/tree` `/fork` `/resume` | 采用 JSON 单文件 + `/resume`，不做树分支（小模型用不上） |
| 上下文 | 接近上限时自动 compaction，可自定义 | Python 侧按 token 估算触发，失败时机械摘要兜底 |
| 命令 | `/model` `/new` `/compact` `/export` `/status` | 保留同类命令，另加 `/goal` `/rounds` `/memory` |
| 输入 | `!` 执行 shell、`@` 引用文件、`Ctrl+O` 展开工具输出 | 实现 `!` 与 `@`，用 `/verbose` 替代快捷键 |
| 工具输出 | 折叠展示，失败可见 | 单行工具调用 + 结果摘要，`/verbose` 展开 |
| 明确不做 | 无子 Agent、无 plan mode | **反其道**：子 Agent 是本框架的核心（评审） |

### dsh-ops（工具实现）

| 设计点 | dsh-ops 的做法 | ELAgent 的取舍 |
|---|---|---|
| Token 成本 | 隐藏冗余工具、渐进式披露 | 工具描述压缩成一行签名，8 个工具约 200 token |
| 返回格式 | 紧凑、带行号、错误可执行 | `read` 返回 `行号|内容`，错误附修复提示 |
| 平台 | Windows 用 `ops_bash` 替换 PowerShell | `terminalbash` 自动探测 bash/cmd/powershell |
| 渐进加载 | `tools_addition` 按需加载 | 保留精简常驻工具集，未做动态加载（规模不需要） |
| 安全 | 低侵入、可卸载 | 路径越界拦截 + 危险命令确认 |

### 结论

Pi 提供"人机交互外壳"，dsh-ops 提供"省 token 的工具面"，ELAgent 补上第三块：**面向弱模型的确定性控制面**。

---

## 2. 项目结构

```
ELAgent/
├─ pyproject.toml            # 打包与 ela 入口点
├─ README.md
├─ ela.cmd                   # 免安装启动垫片
├─ install.ps1               # 安装 + PATH 配置
├─ elagent/
│  ├─ cli.py                 # 参数解析、一次性执行、doctor
│  ├─ config.py              # 默认值 <- config.json <- 环境变量 <- CLI
│  ├─ llm.py                 # OpenAI 兼容客户端：流式、重试、预算升级
│  ├─ protocol.py            # 动作/评审 JSON 的容错解析（核心）
│  ├─ prompts.py             # 全部提示词，单条 < 1000 token
│  ├─ context.py             # token 估算、自动压缩、机械兜底
│  ├─ session.py             # 会话持久化、原子写、Markdown 导出
│  ├─ memory.py              # global.memory 读写、相关度召回、经验提炼
│  ├─ agent.py               # 主 Agent / 评审子 Agent / 目标循环
│  ├─ events.py              # 事件总线（核心不依赖 UI）
│  ├─ tokens.py              # 中英混排 token 估算
│  ├─ tools/
│  │  ├─ base.py             # Tool / ToolResult / ToolContext / Registry
│  │  ├─ fs.py               # read write edit ls glob grep
│  │  ├─ shell.py            # terminalbash（可取消、可超时、杀进程树）
│  │  ├─ todo.py             # 计划持久化（长程任务的外部记忆）
│  │  └─ extra.py            # time / ask / subagent / workflow
│  └─ ui/
│     ├─ fullscreen.py       # 全屏 TUI（Pi 风格，后台线程 + 事件总线）
│  ├─ transcript.py    # 对话区模型：Rich 渲染、宽度缓存、滚动切片
│  ├─ tui.py           # 行式界面（无控制台时的降级实现）
│     ├─ commands.py         # / 命令系统
│     └─ render.py           # rich 渲染、流式状态、工具块
└─ tests/
   ├─ mock_llm.py            # 脚本化模型，离线可测
   ├─ test_protocol.py       # 24 种畸形 JSON 场景
   ├─ test_tools.py          # 文件/路径/超时/待办
   └─ test_loop.py           # 标准循环、目标循环、压缩、恢复
```

运行时数据：

```
~/.elagent/
├─ config.json               # 可手工编辑
├─ global.memory             # 长期经验（Markdown，可编辑）
├─ history                   # 输入历史
├─ state.json                # 每个工作目录最近会话
└─ sessions/
   ├─ 20261030-131201-9308.json       # 会话全文
   └─ 20261030-131201-9308.todo.json  # 该会话的计划
```

---

## 3. 核心设计

### 3.1 单动作协议

每轮模型只能输出一个 JSON：

```json
{"tool":"read","args":{"path":"a.py"}}
{"final":"已完成，修改了 a.py，验证命令 exit=0"}
```

Python 侧强制：一次一个工具、看到 `OBS[...]` 才能继续、非法输出触发提醒并重试。
工具结果以 `OBS[read] ok` 注入，提示词明确告知"这是系统注入，不要模仿"。

提示词先做**类型分流**：对话/问答直接 `final`，工程任务才进工具循环；并明确"信息足够就停、同一动作禁止原样重试"。Python 侧再兜一层：同一工具 + 同一参数第 3 次出现即强制收尾。模型一轮输出多个动作时只执行第一个，并在 OBS 后明确告知忽略了几个，避免"第二条验证命令被静默丢弃"。

### 3.2 JSON 容错（`protocol.py`）

模型实际会输出这些变体，全部可解析：

```
```json {...} ```                 围栏
{'tool': 'read', ...}            Python 字面量
{"tool":"read","args":{...},}    尾逗号
{"tool":"read","args":{...}      截断（自动补括号）
read(path="a.py")                函数调用风格
{"read": {"path": "a.py"}}       简写
{"tool_calls":[{...}]}           OpenAI 原生结构
说明文字 + JSON + 说明文字        散文包裹
```

顺序：直接 `json` → `ast.literal_eval` → 结构修复（智能引号、注释、裸键、尾逗号、换行转义、括号配平）→ 正则提取字段。
仍失败则返回 `invalid` 并附原因，由主循环注入"只输出 JSON"的纠正提示；连续 3 次失败强制收尾，不会死循环。

评审结果同样容错：`{"status":1}` / `"status":"pass"` / `status: 1` / 中文"不通过" 都能识别；
无法判定时返回 `-1`，最多重试 3 次后中止，避免把"评审坏了"误判为"任务通过"。

### 3.3 两种模式

**标准模式**：一次任务，ReAct 循环直到 `final` 或步数上限（默认 24 步）。

**目标模式**（`/goal`）：

```
round = 0
while max_rounds < 0 or round < max_rounds:
    round += 1
    result  = MainAgent.run(task, feedback=上一轮 reason, history=历史缺陷)
    verdict = JudgeAgent.evaluate(task, result, 证据)
    if verdict.status == 1: 成功
    if verdict.status == 0: 带着 reason 进入下一轮
    if verdict.status == -1: 连续 3 次则中止
```

关键控制点全部在 Python：

- 主 Agent 没有产出 `final` → **不调用评审**，直接判 0（避免"空结果被判通过"）。
- 同一工具 + 同一参数第 3 次出现 → 判定**重复动作**，注入纠正提示并强制收尾（文件改动后计数清零，避免误伤"改完再跑一遍测试"）。
- 连续 2 轮模型请求失败 → 中止，不空转。
- `max_rounds` 默认 `-1`（无限），达到上限返回"未完成"。
- 每轮使用**全新上下文**，只注入任务 + 最近 8 条缺陷摘要，防止无限轮次撑爆窗口。
- 评审拿到的是证据而非自述：改动文件（行数/字符数/是否存在）、最近 6 条命令及退出码、计划状态。

### 3.4 上下文压缩

- 估算：CJK ≈ 1 token/字，拉丁 ≈ 1 token/3.6 字符（偏保守）。
- 触发：超过 `compact_threshold`（默认 12000）或 `已用 + max_tokens > context_limit`。
- 动作：保留 system + 最近 8 条，中间部分交给模型压缩成 ≤400 字进度摘要。
- 兜底：压缩请求失败时改用机械摘要（消息数、用过的工具、最近请求），**永不因压缩失败中断任务**。
- 原始消息仍在会话 JSON 中，可导出查看。

### 3.5 会话与恢复

- 每次工具调用后不写盘，每个任务结束、每轮评审后原子写（临时文件 + `os.replace`）。
- `ela` 每次启动**新建会话**（不自动恢复）；`ela --resume` 或 TUI `/resume` 手动恢复；`/new` 随时新建。
- `/export` 导出 Markdown（含轮次、缺陷、全文）。
- 会话记录同时用于 `--resume` 与跨天续跑。

### 3.6 global.memory

写入：**不自动记录**。任务完成后用 `/memory save` 手动触发——用独立提示词让模型提炼 3–5 条经验 → Python 清洗、去重、截断 → 追加到 `~/.elagent/global.memory`。

读取：任务开始时按关键词重合度（中文 bigram + 英文词）召回最多 3 条注入 system prompt（上限 700 token），`/memory off` 关闭参考、`/memory on` 开启。一次性模式自动召回。

文件是纯 Markdown，可手工编辑：

```markdown
## 2026-10-30 13:12 · 创建 calc.py 并测试
- Windows 下用 python 而不是 python3，后者可能指向 Store 别名
- 断言脚本要显式 from calc import add, div，否则 ModuleNotFoundError
```

### 3.7 时间、提问、子代理与工作流

| 能力 | 行为 | 边界 |
|---|---|---|
| `time` | 返回本地时间/时区，并从会话文件读创建时间、消息数、累计用量 | 只读，不修改任何状态 |
| `ask` | 全屏 TUI 用输入行回答；行式界面用 prompt；`-p` 非交互直接返回错误 | 绝不阻塞批处理；模型被提示“只在缺少关键信息时用” |
| `subagent` | 全新上下文执行一个自包含子任务，返回最终结论；用量记入父会话 | 默认深度 1（子代理不能再开子代理）、最多 8 步；事件静默，避免污染父对话 |
| `workflow` | 把 2–4 个步骤顺序交给子代理，前序结论截断后注入下一步，返回逐步汇总 | 串行执行；任一步失败立即停止；总步数受 `workflow_max_steps` 限制 |

子代理复制的工具集会自动去掉 `ask`、`subagent`、`workflow`、`todo`：既防递归，也避免子代理改父代理的计划。子代理与父代理共用同一个模型服务，因此是串行且成本叠加的。

---

## 4. 工具

| 工具 | 参数 | 说明 |
|---|---|---|
| `read` | `path, offset=1, limit=200` | 带行号；二进制/超大文件拦截；UTF-8/GB18030 自动回退 |
| `write` | `path, content` | 原子写、自动建目录、报告覆盖前后行数 |
| `edit` | `path, old, new, replace_all=false` | 要求唯一匹配，多处匹配报错并提示加长片段 |
| `ls` | `path=".", limit=200` | 跳过 `.git` `node_modules` `__pycache__` 等 |
| `glob` | `pattern, path=".", limit=100` | 按修改时间排序 |
| `grep` | `pattern, path=".", glob="*", limit=50` | 正则，返回 `文件:行号:内容` |
| `terminalbash` | `command, timeout=60, cwd="."` | 自动选 bash/cmd/powershell；可取消；超时杀进程树；输出头尾截断 |
| `todo` | `action, items, id` | 计划写入会话目录，并注入 system prompt |
| `time` | 无 | 时间上下文：本地时间/时区 + 会话耗时与用量 |
| `ask` | `question` | 交互模式向用户提问并等待回答；非交互模式立即返回错误，不会挂起 |
| `subagent` | `task, max_steps=8` | 独立上下文的子代理，返回结论；默认最多一层，工具集去掉 ask/subagent/workflow/todo |
| `workflow` | `steps`（2–4 个字符串） | 顺序执行，每步一个子代理，前序结果截断注入下一步；任一步失败即停止 |

安全边界：路径默认锁定工作目录；`rm -rf`、`format`、`git reset --hard` 等需确认（`--yolo` 跳过，非交互模式默认拒绝）。

扩展新工具只需继承 `Tool` 并在 `build_registry()` 注册，无需改 Agent 或 UI。

---

## 5. 命令

| 命令 | 作用 |
|---|---|
| `/goal [任务]` | 开目标模式；`/goal off` 关闭 |
| `/rounds N` | 最大轮数，`-1` 无限 |
| `/new` `/resume` `/sessions` | 会话管理 |
| `/compact` | 立即压缩 |
| `/memory save` | 记录上次任务的经验到 global.memory |
| `/memory on\|off\|show` | 参考开关与查看（不自动写入） |
| `/model [name]` | 查看/切换模型 |
| `/status` `/tools` | 状态与工具清单 |
| `/verbose` `/thinking` | 工具详情 / 思考过程 |
| `/export [path]` | 导出 Markdown |
| `!命令` | 直接执行 shell 并注入上下文 |
| `@路径` | 附加文件内容到任务 |
| `Ctrl+C` / `Ctrl+D` | 中断 / 退出 |

---

## 6. 配置

`~/.elagent/config.json`（不存在则用默认值）：

```json
{
  "base_url": "http://127.0.0.1:8138/v1",
  "api_key": "sk-local",
  "model": "auto",
  "send_generation_params": false,
  "temperature": 0.2,
  "top_p": 0.9,
  "max_tokens": 2048,
  "max_stream_chars": 8000,
  "context_limit": 32768,
  "compact_threshold": 12000,
  "keep_recent_messages": 8,
  "max_steps": 24,
  "max_rounds": -1,
  "max_invalid_actions": 3,
  "max_repeat_actions": 3,
  "subagent_max_steps": 8,
  "subagent_max_depth": 1,
  "workflow_max_steps": 4,
  "judge_retries": 3,
  "judge_max_tokens": 700,
  "shell": "auto",
  "confirm_dangerous": true,
  "memory_enabled": true,
  "memory_auto_recall": true,
  "memory_on_complete": false
}
```

默认 `send_generation_params=false`：**不向服务端发送 `temperature` / `top_p` / `max_tokens`**，这三项完全由启动 llama.cpp 时的参数决定，ELAgent 也不会在运行中修改它们（例如自动翻倍预算）。只有显式设为 `true` 时才会用本文件的参数覆盖服务端。`max_tokens` 在本地仅作为上下文预算的预留量。

`max_stream_chars`（默认 8000 字符，0 为不限制）是**本地**流式读取上限：达到就断开这次请求，且不接受被截断的 `final` 或工具调用。它不改变任何服务端参数，只是防止服务端未设 `--n-predict` 时客户端被失控推理拖住；建议同时在 llama.cpp 侧限制生成长度。

环境变量：`ELAGENT_BASE_URL`、`ELAGENT_MODEL`、`ELAGENT_API_KEY`、`ELAGENT_HOME`、`ELAGENT_MAX_ROUNDS`。

---

## 7. 针对本地模型的兼容性修复

实测 Ling-3.0-tiny（llama.cpp，131k ctx）时发现并已在代码中处理：

1. **keep-alive 404**：同一连接上 `GET /v1/models` 后的首个 `POST /v1/chat/completions` 返回 404。
   处理：禁用连接复用（`max_keepalive_connections=0`），并把 404 纳入重试。
2. **推理占满预算**：`reasoning_content` 吃掉全部输出预算，`content` 为空。
   处理：不修改服务端参数、**不再自动翻倍 `max_tokens`**；空响应做有限次原参数重试，评审场景允许从 reasoning 兜底解析。
3. **空响应**：200 但无内容。处理：有限次原参数重试，最终返回明确错误而不是崩溃。
4. **bash 解析到 WSL**：Windows 上 `shutil.which("bash")` 可能命中 `System32\bash.exe`（WSL 启动器，慢、可能被策略拦截、文件系统视图不同）。
   处理：优先探测 Git for Windows 的 `bin/bash.exe`，找不到才回退 `cmd`。
5. **无控制台环境**：prompt_toolkit 在管道输入或非 Windows 控制台下会抛 `NoConsoleScreenBufferError`。
   处理：自动降级为逐行读取，Agent 功能不受影响。
6. **生成失控**：服务端未设 `--n-predict` 时，推理可能连续生成数分钟。
   处理：客户端在 `max_stream_chars`（默认 8000 字符）处断开且不重试；截断的输出不接受 `final` 或工具调用，交给普通纠错流程。

---

## 8. 提示词预算

`ela --doctor` 实测：

```
system 926 · judge 217 · compact 149 · memory 81   （上限 1000）
```

控制手段：规则用短句、工具只给一行签名、示例只给必要的一个、压缩/记忆提示词独立且极短。

---

## 9. 测试

```bash
python -m unittest discover -s tests -t tests -v
```

92 项，覆盖：畸形 JSON、工具路径越界与近似名提示、编辑多处匹配、shell 超时、评审判定、目标循环重试、评审非法输出中止、上下文压缩与兜底、多轮历史拼接、全屏渲染与滚动、思考展开、手动记忆、提示词类型分流与预算、重复动作收尾、生成参数不下发、截断输出拒绝、XML/函数式工具调用、多动作只执行第一个、默认新建会话与 `--resume`、time/ask/subagent/workflow、对象形式的 final、子代理改动文件回传。

`ela --doctor` 检查端点、模型、原生上下文、提示词预算、工具与会话目录。

---

## 10. 限制

- 工具调用是**文本协议**，不是原生 function calling；对支持原生 tools 的服务端也兼容（会优先读取 `tool_calls`）。
- 不做沙箱，`terminalbash` 拥有当前用户权限；不可信任务请在容器/虚拟机中运行。
- `subagent`/`workflow` 共用同一个本地模型服务，串行执行、成本叠加；建议只在任务确实可拆分时使用。
- 会话树不支持分支（Pi 的 `/tree`），只做线性恢复。
- 评审质量受模型能力限制；它降低而非消除误判，关键任务仍需人工复核。
