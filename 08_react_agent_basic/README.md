# Phase 5: 规划与复杂任务执行 (Planning & Execution)

**前置依赖**: Phase 1, Phase 2

## 08_react_agent_basic

**目标**: 扫盲级 ReAct 模式实践，学习拆解 Thought / Action / Observation 护城河。

**学习内容**:
- 通过极简单的任务（如问答或调取两个极简工具），直观地感受和追踪 Agent 执行流：
  - `Thought`: Agent 如何分析当前的已知状态。
  - `Action`: Agent 如何决定要调用哪个 Tool 参数是什么。
  - `Observation`: Agent 收到 Tool 返回结果后重新进入思考循环。

**技术栈**:
- Python (3.10+)
- LangChain Agents (`langchain >= 0.2.0`)
