# Phase 3: 上下文与记忆机制 (Memory & Context)

## 05_memory_chatbot_v2

**目标**: 让 Agent 拥有“长期记忆”与更复杂的记忆策略。

**学习内容**:
- 将聊天记录持久化存储到本地文件或 SQLite 数据库中。
- 探索超出大模型上下文窗口的记忆压缩策略，如 `ConversationSummaryMemory`。 
- 让 Agent 能够在重启后或者面对超长对话时依然能够保持正确的上下文记忆。

**技术栈**:
- Python
- SQLite
- LangChain Memory
