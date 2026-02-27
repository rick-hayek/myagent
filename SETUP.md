# AI Agent 开发学习环境配置指南 (Setup Guide)

本项目从基础 API 调用到复杂的多智能体编排，循序渐进地带您实践 Agent 开发。在开始之前，我们需要配置一个统一且干净的 Python 环境。

## 1. 环境准备 (Python 3.10+)

建议使用虚拟环境工具（如 `venv`, `conda`, 或 `poetry`）来隔离依赖。

使用 `venv` 的示例：
```bash
# 在项目根目录下创建虚拟环境
python3 -m venv venv

# 激活虚拟环境 (macOS/Linux)
source venv/bin/activate
```

## 2. 安装核心依赖

您可以根据当前学习的 Phase 逐步安装依赖，也可以现在就安装第一阶段的核心包：

```bash
# 安装 LangChain 核心及 OpenAI 官方库
pip install -U langchain langchain-openai langchain-community 
```

*(注：后续阶段如果涉猎 LangGraph, ChromaDB, Tavily 等组件时，对应目录下的文档会提示安装命令。)*

## 3. 环境变量配置 (.env)

为了避免将敏感的 API Key 提交到代码仓库中，我们使用 `.env` 文件来管理环境变量。

1. 在项目根目录复制 `.env.example` 为 `.env`：
   ```bash
   cp .env.example .env
   ```
2. 编辑 `.env` 文件，填入您申请的各个平台的 API Key。
3. （重要）本项目开发时，请确保使用 `python-dotenv` 库在代码入口加载这些变量：
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

## 4. 准备就绪
完成以上配置后，您可以直接进入 `01_basic_api` 目录，开始您的第一行代码编写了！
