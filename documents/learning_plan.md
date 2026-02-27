# AI Agent Development Learning Plan

这是一个从基础理论到高级复杂实战的 AI Agent 开发学习计划。我们将按照阶段式进阶，并在当前项目根目录下为每个阶段创建一个或多个对应的实战工具文件夹。部分项目会分为 v1, v2 等不同复杂度的迭代版本。

## 技术栈确认
- **编程语言**: Python 
- **核心框架**: LangChain
- **大模型支持**: 兼容 OpenAI / Anthropic 等（按需配置）

---

## 学习路线与实战目录规划

以下是具体的进阶学习阶段与实战项目规划：

### Phase 1: 基础理论与大模型交互 (Basics & API)
*   **目录**: `01_basic_api/`
    *   **目标**: 学习如何调用大模型的 API（如 OpenAI, Anthropic），引入 LangChain 基础概念。
    *   **内容**: 简单的文本生成、Prompt Template 构造、Output Parser 基础，以及流式响应（Streaming）体验。

### Phase 2: 工具调用与外部交互 (Tool Calling / Function Calling)
*   **目录**: `02_calculator_agent/`
    *   **目标**: 理解大模型如何与外部世界互动。
    *   **内容**: 使用 LangChain Tools/Toolkits 构建一个能识别数学计算意图，并准确调用本地加减乘除 Python 函数的 Agent。
*   **目录**: `03_weather_agent/`
    *   **目标**: 接入真实的外部网络 API。
    *   **内容**: 编写一个能够通过调用公开天气 API 来解答用户天气相关问题的 Agent。

### Phase 3: 上下文与记忆机制 (Memory & Context)
*   **目录**: `04_memory_chatbot_v1/`
    *   **目标**: 让 Agent 拥有“短期记忆”。
    *   **内容**: 运用 LangChain 的 ConversationBufferMemory 等内存模块，实现基本的多轮对话机器人。
*   **目录**: `05_memory_chatbot_v2/`
    *   **目标**: 让 Agent 拥有“长期记忆”。
    *   **内容**: 进阶版本，将聊天记录持久化到本地文件或 SQLite 中，探索摘要记忆策略（ConversationSummaryMemory）。

### Phase 4: 检索增强生成 (RAG - Retrieval-Augmented Generation)
*   **目录**: `06_simple_rag/`
    *   **目标**: 解决大语言模型“不知道私有数据”的问题。
    *   **内容**: 实现一个基础的文档问答系统。结合 LangChain Document Loaders, 简单的向量表示并在内存中进行 QA 检索。
*   **目录**: `07_advanced_rag/`
    *   **目标**: 应对长文档与复杂查询。
    *   **内容**: 引入向量数据库（如 Chroma），加入文档分块（Text Splitters）、LangChain Retriever、多路召回与重排序（Reranking）等高级策略。

### Phase 5: 规划与复杂任务执行 (Planning & Execution)
*   **目录**: `08_research_agent/`
    *   **目标**: 学习经典的 ReAct (Reason + Act) 模式，让 Agent 学会“思考-行动-观察”。
    *   **内容**: 基于 LangChain AgentExecutor，构建一个能自主拆解长线复杂研究任务，主动使用搜索引擎查找资料，总结并最终输出长篇报告的 Agent。

### Phase 6: 多智能体协同 (Multi-Agent Systems)
*   **目录**: `09_dev_team_agents/`
    *   **目标**: 探索多个专注不同职责的 Agent 如何协同工作。
    *   **内容**: 使用 LangGraph，模拟一个微型开发团队。包含一个“产品经理”提出需求，“程序员”负责写代码，“测试员”负责进行单测和代码走查，最终协作输出可用代码。

### Phase 7: 高级实战项目 (Advanced Projects)
*   **目录**: `10_coding_assistant_v1/`
    *   **目标**: 构建专注于特定本地领域任务的专属 Agent。
    *   **内容**: 基础的自动编码助手，能根据指令读取本地 Python 代码文件，按要求修改代码并覆写回本地。
*   **目录**: `11_coding_assistant_v2/`
    *   **目标**: 系统级 Agent 开发。
    *   **内容**: 进阶版助手，具备扫描整个工作区、自主运行终端命令行（如安装依赖 `pip install`、执行单测 `pytest`）、根据报错信息进行闭环迭代修复 Bug 的高级能力。
