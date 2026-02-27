# Phase 7: 高级实战项目 (Advanced Projects: System Agents)

**前置依赖**: Phase 2, Phase 5

## 12_coding_assistant_v2 (原 11_coding_assistant_v2)

**目标**: 闭环高阶系统级 Agent 开发，与系统的安全性思考。

**学习内容**:
- 为 Agent 提供执行终端命令的能力的 Tool（例如执行 `pip install`，`pytest`）。
- 建立闭环工作流：写代码 -> 跑单测 -> 获取报错 -> 分析报错 -> 修复代码 -> 再次执行。
- **🚨【核心：避免无限循环错误】🚨**
  - **背景**：当 Agent 获取到错误并自主编写修复脚本时，如果始终抓不到痛点，非常容易陷入反复报错、反复修复、不断消耗大量 Token 甚至打满服务器计算资源的【死循环（Infinite Recovery Loop）】。
  - **重点解决方案**: 
    1. 为 Agent 运行图引入 `max_steps` 或递归深度控制机制以强行兜底停止。
    2. 使用 Human-in-the-Loop 请求人类反馈再执行。
    3. 设计 fallback 系统以防工具异常。

**技术栈**:
- Python (3.10+)
- LangGraph 
- Subprocess / Terminal Tooling
