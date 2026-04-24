# Phase 7: 高级实战项目 - 系统级编码助理 v2 (Advanced System Agents)

**前置依赖**: `11_coding_assistant_v1`

## 12_coding_assistant_v2

**目标**: 实现一个闭环的高阶系统级 Agent。它不仅能写代码，还能在操作系统终端帮你执行代码（例如跑单测），并在看到报错后，自主反思、回去修改原代码，直到测试通过。

**学习与安全核心**:
- **终端执行工具 (Terminal Execution Tool)**: 利用 `subprocess` 让 LLM 拥有运行 `python` 或 `pytest` 的能力。需要引入安全沙箱概念，严厉禁止 `rm -rf` 等危险指令。
- **“闭环”与自动迭代**: `读代码 -> 写代码 -> 跑单测运行 -> 观察报错日志 -> 继续写代码` 的强化学习闭环。
- **Human-in-the-loop (人类在环)**: 终极安全防线。在 Agent 准备真正执行终端命令或者覆盖源码之前，必须让程序暂停，在终端弹出一个框等待管理员（你）的 `Yes/No` 审阅许可！

**技术栈**:
- Python (3.10+)
- `subprocess` 模块
- LangGraph (`interrupt_before` 中断节点机制)
