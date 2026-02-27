# Phase 8: 评估与追踪 (Evaluation & Tracing)

**前置依赖**: 无严格约束，可在跑通任意前六个 Phase 后切入

## 13_agent_evaluation

**目标**: 解决“如何知道我的 Agent 做得好不好”的工程盲区，完成 Agent 工程的闭环。

**学习内容**:
- 引入 LangSmith 记录 Agent 的每个工具调用链、思考链条及 Token 使用占比。
- 编写专门的评估脚本与测试集。
- 从工程实用角度，量化考察 Agent：输出的内容精准度、Task 完成率、工具触发正确率。
- 配置 Feedback 和自定义评估模型（LLM-as-a-judge）。

**技术栈**:
- Python (3.10+)
- LangSmith / LangGraph
