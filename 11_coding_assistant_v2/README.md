# Phase 7: 高级实战项目 (Advanced Projects)

## 11_coding_assistant_v2

**目标**: 探索构建具备一定自治能力，甚至进行自我纠错的系统级 Agent（进阶版）。

**学习内容**:
- 为 Agent 提供执行终端命令的能力的 Tool（例如执行 `pip install`，`pytest`）。
- 建立闭环工作流：写代码 -> 跑单测 -> 获取报错 -> 分析报错 -> 修复代码 -> 再次执行。
- 总结 AI Agent 架构设计的挑战以及如何避免无限循环错误。

**技术栈**:
- Python
- LangGraph
- Subprocess / Terminal Tooling
