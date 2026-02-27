# Phase 3: 上下文与记忆机制 (Memory & Context)

## 04_memory_chatbot_v1

**目标**: 让 Agent 拥有处理多轮对话的“短期记忆”。

**学习内容**:
- 理解大模型本身无状态（Stateless）的特性。
- 运用 LangChain 的 `ConversationBufferMemory` 或基于消息历史列表模块构建上下文。
- 实现一个基本的、能记住前几轮对话的聊天机器人。

**技术栈**:
- Python
- LangChain Memory / RunnableWithMessageHistory
