# Phase 5: 规划与复杂任务执行 (Planning & Execution)

**前置依赖**: Phase 1, Phase 2

## 09_research_agent (原 08_research_agent)

**目标**: 理解并应用经典的 ReAct 模式，让 Agent 具备思考、观察并执行长线任务的复合应用能力。

**学习内容**:
- 基于 LangGraph 或原生的长流程编排构建一个能主动搜集信息的研究 Agent。
- Agent 在缺乏信息时会自主调用搜索引擎进行查找。
- 最终能将搜集到的资料汇总并输出结构化长篇报告。

**技术栈**:
- Python (3.10+)
- LangGraph / LangChain Agents (`langchain >= 0.2.0`)
- Search Tools (Tavily / DuckDuckGo 等)
