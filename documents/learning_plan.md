# AI Agent Development Learning Plan

这是一个从基础理论到高级复杂实战的 AI Agent 开发学习计划。我们将按照阶段式进阶，并在当前项目根目录下为每个阶段创建一个或多个对应的实战工具文件夹。

## 技术栈确认
- **编程语言**: Python (推荐 3.10+)
- **核心框架**: 
  - `langchain >= 0.2.0` (注：API 变动频繁，本项目主要基于 0.2.x 版本生态)
  - `langgraph >= 0.1.0` (用于复杂流程及多智能体编排)
- **大模型支持**: 兼容 OpenAI / Anthropic 等（按需配置）

---

## 学习路线与实战目录规划

以下是具体的进阶学习阶段与实战项目规划：

### Phase 1: 基础理论与大模型交互 (Basics & API)
*   **前置**: 无
*   **目录**: `01_basic_api/`
    *   **目标**: 学习如何调用大模型的 API（如 OpenAI, Anthropic），引入 LangChain 基础概念。
    *   **内容**: 简单的文本生成、Prompt Template 构造、Output Parser 基础，以及流式响应（Streaming）体验。

### Phase 2: 工具调用与外部交互 (Tool Calling / Function Calling)
*   **前置**: 无，或基本了解 Python
*   **目录**: `02_calculator_agent/`
    *   **目标**: 理解大模型如何与外部世界互动。
    *   **内容**: 使用 LangChain Tools/Toolkits 构建一个能识别数学计算意图，并准确调用本地加减乘除 Python 函数的 Agent。
*   **目录**: `03_weather_agent/`
    *   **目标**: 接入真实的外部网络 API。
    *   **内容**: 编写一个能够通过调用公开天气 API 来解答用户天气相关问题的 Agent。

### Phase 3: 上下文与记忆机制 (Memory & Context)
*   **前置**: 无
*   **目录**: `04_memory_chatbot_v1/`
    *   **目标**: 让 Agent 拥有“短期记忆”。
    *   **内容**: 运用基于消息历史列表的模块构建上下文，实现基本的多轮对话机器人。
*   **目录**: `05_memory_chatbot_v2/`
    *   **目标**: 让 Agent 拥有“长期记忆”。
    *   **内容**: 进阶版本，将聊天记录持久化到本地文件或 SQLite 中，探索摘要记忆策略。

### Phase 4: 检索增强生成 (RAG - Retrieval-Augmented Generation)
*   **前置**: Phase 1
*   **目录**: `06_simple_rag/`
    *   **目标**: 解决大语言模型“不知道私有数据”的问题。
    *   **内容**: 实现一个基础的文档问答系统。结合简单向量表示并在内存中进行 QA 检索。
*   **目录**: `07_advanced_rag/`
    *   **目标**: 应对长文档与复杂查询。
    *   **内容**: 引入向量数据库（如 Chroma），加入文档分块（Text Splitters）、多路召回与重排序（Reranking）等高级策略。

### Phase 5: 规划与复杂任务执行 (Planning & Execution)
*   **前置**: Phase 1, Phase 2
*   **目录**: `08_react_agent_basic/`
    *   **目标**: 扫盲级 ReAct 模式实践。
    *   **内容**: 通过极简的任务，体验并理解 ReAct (Reason + Act) 模式下的 Thought / Action / Observation 循环，弄清 Agent 是如何做出决策的。
*   **目录**: `09_research_agent/`
    *   **目标**: 掌握复杂信息收集与任务拆分能力。
    *   **内容**: 基于 LangGraph 或原生的长流程编排，构建一个能自主拆解长线研究任务，主动使用搜索引擎查找资料并总结报告的复合型 Agent。

### Phase 6: 多智能体协同 (Multi-Agent Systems)
*   **前置**: Phase 5 (需掌握基本的 LangGraph)
*   **目录**: `10_dev_team_agents/`
    *   **目标**: 探索多个专注不同职责的 Agent 如何协同工作。
    *   **内容**: 使用 LangGraph 模拟一个微型开发团队，包含“产品经理”、“程序员”、“测试员” 节点，通过图状态轮转交替完成代码交付。

### Phase 7: 高级实战项目 (Advanced Projects: System Agents)
*   **前置**: Phase 2, Phase 5
*   **目录**: `11_coding_assistant_v1/`
    *   **目标**: 构建专注本地任务的专属 Agent（基础版）。
    *   **内容**: 扩展系统级工具，根据指令要求读取本地 Python 文件，修改代码并安全覆写回本地。
*   **目录**: `12_coding_assistant_v2/`
    *   **目标**: 闭环高阶系统级 Agent 开发，与安全性思考。
    *   **内容**: 赋予终端命令执行能力（如跑单测 `pytest`），并根据报错信息迭代修复 Bug。
    *   **核心安全要点 - 避免无限循环**: 本项目将重点剖析和解决大模型在代码自主修复过程中最容易引发的“死循环”与过度消耗 Token 的工程灾难，学习实施最大迭代步数（Max Steps）和人类中断介入（Human-in-the-loop）的安全兜底设计。

### Phase 8: 评估与追踪 (Evaluation & Tracing) 
*   **前置**: 可以在任意 Phase 跑通后进行
*   **目录**: `13_agent_evaluation/`
    *   **目标**: 解决“如何知道我的 Agent 做得好不好”的盲区。
    *   **内容**: 引入 LangSmith 或自定义追踪脚本记录 Agent 的中间思考链条，编写测试用例，从工程实用角度对 Agent 的输出准确率、工具调用成功率进行量化评估。
