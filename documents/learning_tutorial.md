# AI Agent 实战开发备忘录 (Step-by-Step Tutorial)

此文档用于记录我们在开发过程中的每一步实际操作，作为详尽的动手指南。

## 准备工作：环境初始化

我们在正式开始编码前，首先需要搭建一个隔离的 Python 环境并安装必需的库。

### 步骤 1：配置虚拟环境与核心依赖

**目标**：创建一个独立干净的开发环境，避免与 Mac 系统的全局 Python 库产生冲突，并按需安装第一阶段需要的库。

**执行命令**：
在项目根目录 `/Users/rick/src/myagent` 下的终端中执行：

```bash
# 1. 创建名为 venv 的虚拟环境
python3 -m venv venv

# 2. 激活虚拟环境 (你会看到命令行前缀多出了 (venv))
source venv/bin/activate

# 3. 安装 LangChain 核心、Google 官方纯 SDK、LangChain Google 集成、以及环境变量读取库
pip install -U langchain google-genai langchain-google-genai python-dotenv
```

### 步骤 2：使用纯 SDK 请求大模型

**目标**：不依赖高级框架，直接使用 Google 官方自带的 Python SDK `google-genai` 进行底层调用。这有助于我们在未来使用 LangChain 遇到报错时，排查是否是底层接口或 Key 的问题。

**代码实现** (`01_basic_api/1_ai_sdk_call.py`)：

```python
import os
from dotenv import load_dotenv
from google import genai

# 1. 自动寻找并加载环境中的 .env 文件
load_dotenv()

print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")

# 2. 实例化官方无头 SDK 客户端
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print(f"User input: Explain how AI works in a few words")

# 3. 调用底层 API 生成内容
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

# 4. 打印纯文本结果
print(f"AI response: {response.text}")
```

### 步骤 3：使用 LangChain 调用大模型

**目标**：引入 LangChain 框架，体验它是如何将各种底层的大模型接口（如 Google, OpenAI）抽象包装成统一格式的。

**代码实现** (`01_basic_api/2_langchain_basic.py`)：

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 实例化 LangChain 的大模型包装类
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")
print("通过LangChain调用Google Gemini API")
print("input: Explain how AI works in a few words")

# 调用 invoke 方法向模型提出问题
response = llm.invoke("Explain how AI works in a few words")

# 解析并打印回答内容。注意 response 是一个 AIMessage 对象。
print("\n [LangChain 返回的完整对象类型]:", type(response))
print("\n [AI 生成的文本内容]:", response.content)
```

**核心观察**：无论底层是哪个模型，LangChain 返回的始终是标准化的 `AIMessage` 对象。

### 步骤 4：引入提示词模板 (Prompt Templates)

**目标**：学习使用 `ChatPromptTemplate`，将用户角色（System/User）和动态变量（{user_input}）结合起来，体会大模型应用中不可或缺的 Prompt 工程化。

**代码实现** (`01_basic_api/3_prompt_template.py`)：
```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. 创建一个 LLM 实例
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 2. 创建一个带有 system 设定和 user 动态变量的聊天提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的英语翻译官...只输出翻译结果，不要带任何多余解释。"),
    ("user", "给我翻译这句话：{user_input}"),
])

# 3. 利用 LCEL 管道符将模板和大模型组合拼接
chain = prompt | llm

user_text = input("请输入你想翻译的中文互联网黑话: ")


# 4. 执行这套组合逻辑，传入变量字典
response = chain.invoke({"user_input": user_text})

print("\n[最终翻译结果]:")
print(response.content)
```

**核心观察**：
`chain = prompt | llm` 是 LangChain 最灵魂的用法，称为 **LCEL** (LangChain Expression Language)。它代表将左侧的处理结果“流式”传递给右侧。

### 步骤 5：引入输出解析器 (Output Parsers)

**目标**：掌握如何使用 `StrOutputParser` 获取纯文本，以及使用 `JsonOutputParser` 结合 Pydantic 获取结构化的 Python 字典数据，为后续 Agent 的稳定运行打下基础。

**代码实现** (`01_basic_api/4_output_parser.py`)：
```python
import pydantic
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("=============== 测试一：字符串解析器 ===============")
prompt1 = ChatPromptTemplate.from_template("请用一句话夸奖这款产品: {product}")
chain1 = prompt1 | llm | StrOutputParser()

input_product = input("请输入产品名称: ")
text_response = chain1.invoke({"product": input_product})
print("[纯文本结果]:", text_response)

print("\n=============== 测试二：JSON 解析器 ===============")
class Joke(BaseModel):
    setup: str = Field(description="笑话的铺垫部分")
    punchline: str = Field(description="笑话的包袱/笑点部分")

json_parser = JsonOutputParser(pydantic_model=Joke)
prompt2 = ChatPromptTemplate.from_template("讲一个关于{topic}的冷笑话。\n{format_instructions}")
chain2 = prompt2 | llm | json_parser

json_response = chain2.invoke({
    "topic": "程序员",
    "format_instructions": json_parser.get_format_instructions()
})
print("[字典解析结果]:", json_response)


print("\n=============== 测试三：现代结构化输出 ===============")
# 1. 现代解法：在 API 底层强制 LLM 绑定 Pydantic 数据结构 (彻底取代 JsonOutputParser)
# 只要模型不支持这个结构，它就会在生成前直接抛错，避免"幻觉"格式
structured_llm = llm.with_structured_output(Joke)

# 2. 提示词变得极度纯净，无需再硬塞 format_instructions 占位符
prompt3 = ChatPromptTemplate.from_template("讲一个关于{topic}的冷笑话。")

# 3. 组装新管道: Prompt -> 强约束的 LLM
chain3 = prompt3 | structured_llm

# 4. 执行生成
structured_response = chain3.invoke({"topic": "产品经理"})

# 返回的直接是被验证过 100% 符合 Joke 结构的 Pydantic 类的实例对象
print("[结构化输出强约束返回的数据]:", structured_response)
print("[结构化输出数据类型]:", type(structured_response))
# 像操作普通对象一样，直接用对象点属性名的方式取值
print("[坚固提取 setup]:", structured_response.setup)
print("[坚固提取 punchline]:", structured_response.punchline)
```

**核心观察 1（传统解析）**：`JsonOutputParser` 会把 Pydantic 类的结构说明转义成一段很长的提示词喂给大模型，拿到长文本后会自动尝试反序列化成 Python 原生的字典。但这种基于 Prompt 的方式在弱模型下可能会遭到“指令无视”导致返回 None。

**核心观察 2（现代约束）**：`.with_structured_output()` 是当今 Agent 开发中最先进、成功率最高的结构化提取方式。它是 API 底层特性，直接返回 Pydantic 对象（通过 `.` 访问属性），从而彻底消除了传统提示词方案的不稳定性。

### 步骤 6：体验打字机效果的流式输出 (Streaming)

**目标**：掌握大模型生成长文本时的最佳用户体验实践——流式输出，让客户端实现像 ChatGPT 一样的打字机打印效果。

**代码实现** (`01_basic_api/5_streaming.py`)：
```python
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = ChatPromptTemplate.from_template("请帮我写一首关于{topic}的长诗，至少包含四段。")
chain = prompt | llm | StrOutputParser()

topic_input = input("请输入你想写诗的主题: ")
print("\n[AI 正在努力作诗中，请欣赏终端里的打字机效果]...\n")

# 【核心差异】不使用 invoke()，改用 stream() 方法获取流式迭代器
for chunk in chain.stream({"topic": topic_input}):
    # print 默认会在每个字符后加换行，我们改成加空字符串，并强制立刻刷新输出流(flush=True)
    print(chunk, end="", flush=True)

print("\n\n[诗歌创作完毕！]")
```

**核心观察**：
`invoke()` 是一次性返回完整结果（会有很长的白屏等待期），而 `stream()` 是一有新内容就立刻返回。配合 `print` 的 `flush=True` 强制缓冲刷新，就能在终端完美复刻商业级 AI 应用的流式体验。

---

### Phase 1 核心知识点总结：解析器 vs 结构化输出的底层机制区别

在体验了大模型提取结构化数据（JSON）的过程中，我们使用了两种截然不同的方案。理解它们的差异是掌握现代 Agent 稳定性的关键：

#### 1. `JsonOutputParser` 的底层机制
它做的事情其实很"笨"，分两步：
1. **指令拼接**：调用 `json_parser.get_format_instructions()` 会自动将你定义的 Pydantic 数据类转换成一段纯文本的严格格式说明（比如要求包含某些特定的字段名和类型）。这段文字被硬塞进 Prompt（提示词）末尾发给模型，本质上就是在"求"模型按格式输出。
2. **事后验证**：拿到模型返回的普通自然语言文本后，Parser 尝试用 `json.loads()` 等手段事后解析它。

**问题根源**：模型返回的是自然语言文本，Parser 只是在事后尝试解析。模型如果是强大的（如 GPT-4），可能愿意乖乖按格式来；如果模型较弱或者被前置任务绕晕了，就会忽略末尾的格式要求乱输出文本。一旦格式不对，Parser 就会报错或提取出 `None`，因为它对模型生成过程**完全没有控制权**。

#### 2. `with_structured_output` 的底层机制
这是现代 Agent 开发的标准解法，它走的是完全不同的路径，利用的是模型的 **Function Calling（函数调用）/ 强结构化特性**。
具体来说：
1. 你定义的 Pydantic 结构会被直接转换成严谨的 JSON Schema。
2. 这个 Schema 是通过 **API 参数级别**（而不是丢进 Prompt 文本里）硬性传给大模型的。
3. 模型在逐字生成（Token 生成机制）的底层阶段就被强行约束：它只能输出符合这个 Schema 规则的 Token，一旦预测偏离就会被 API 底层拦截重绘。
4. 最终返回给你的甚至都不再是纯文本，而是已经被 LangChain 直接反序列化好的原生 **Pydantic 对象实例**。

**关键区别**：约束发生在**生成过程中**，而不是生成之后。模型根本没有任何机会"乱输出"废话。

**一个绝佳的类比**：
- `JsonOutputParser` 像是你告诉一个人：*"请在这张白纸上画一个表格然后把答案填进去"*。你拿到白纸后，自己去猜画成什么样的表格、以及数据怎么填。
- `with_structured_output` 像是你直接递给他一张**已经印好死框框的 Excel 填空表**，他只能且必须在指定的格子里写内容，想写到格子外面都不可能。

---

> **恭喜！你已通关 Phase 1：大模型核心三剑客（API、Prompt、OutputParser）。**
> 下一步，我们将进入真正让大模型变成 Agent (智能体) 的核心阶段：**Tool Calling (工具调用/函数调用)**。

## Phase 2: 工具调用与外部交互 (Tool Calling)

大模型本身只是一个“缸中之脑”，它没有外部实时信息，算数也经常因为幻觉算错。为了让它能解决现实问题，我们需要给它装上“手”和“脚”：也就是提供给它一系列严格规范的 Python 函数。

### 步骤 1：给大模型打造第一批工具

**目标**：理解 LangChain 是如何把普通的 Python 函数包装成大模型能看懂的硬核功能说明书（JSON Schema）的。

**代码实现** (`02_calculator_agent/1_basic_tools.py`)：

```python
from langchain_core.tools import tool

# @tool 装饰器是 LangChain 里极其核心的魔法
@tool
def multiply(a: int, b: int) -> int:
    """这是一个乘法器。当你需要计算两个数字相乘时，请调用此工具。
    Args:
        a: 第一个被乘数
        b: 第二个乘数
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """这是一个加法器。请用来计算两个数字相加的和。"""
    return a + b

tools = [multiply, add]

print("工具在系统中的名称:", multiply.name)
print("大模型看到的工具描述说明书:", multiply.description)
print("\n[底层给大模型看的数据结构 (Schema)]:\n", multiply.args_schema.model_json_schema())

result = multiply.invoke({"a": 3, "b": 4})
print("\n[本地测试向工具传入参数调用: 3 * 4 =]:", result)
```

**核心观察**：
`@tool` 装饰器会自动提取 Python 的**类型注解 (Type Hints)** 和文档说明 (**Docstrings**)，翻译成大模型底层的强结构签名。当你写 Agent 时，你的注释和类型是**写给 AI 读的说明书**，决定了它是否能正确触发和传参。

---

### 步骤 2：将工具绑定 (Bind) 给大模型

**目标**：观察大模型在获得了工具的“说明书”之后，当遇到无法自己回答的问题（如大数相乘）时，它是如何改变行为模式的。

**代码实现** (`02_calculator_agent/2_bind_tools_to_llm.py`)：
```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """这是一个乘法器。当你需要计算两个数字相乘时，请调用此工具。"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """这是一个加法器。请用来计算两个数字相加的和。"""
    return a + b

tools = [multiply, add]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 【核心魔法】将工具说明书通过 API 底层告知大模型
llm_with_tools = llm.bind_tools(tools)

query = "计算123乘以456等于多少？然后再把25加上13算算等于多少？"
response = llm_with_tools.invoke(query)

print("\n[大模型返回的文本内容 response.content]:\n", response.content)
print("\n[大模型返回的工具调用 response.tool_calls]:\n", response.tool_calls)
```

#### ⚠️ 极其关键的核心认知破局

> **问：代码跑完了，可是大模型为什么没有告诉我 123 乘以 456 的结果？！**

如果你有这个疑问，恭喜你，你已经接触到了 AI Agent 最最核心的本质规律：**大模型本身是不具备执行代码的能力的。它只是一个“大脑”。**

在上面的输出中，你会发现 `response.content` (自然语言回复) 是**完全空白**的！
而 `response.tool_calls` 里却多了一串清晰的数据：`{'name': 'multiply', 'args': {'a': 123, 'b': 456}}`。

这说明什么？这代表大模型变聪明了。它在推演时发现了两件事：
1. 自己直接算乘法会胡说八道；
2. 刚才人类发过来的工具列表里，恰好有一个叫 `multiply` 的工具看起来能解决这个问题。

于是，大模型做出了一个 **Tool Call (工具调用)** 的决策，它的潜台词是：
_"嗨人类！我闭嘴不回答了。我已经帮你把你问题里的数字提取出来了，我命令你去调用那个叫做 `multiply` 的 Python 函数，参数我都给你配好 a=123, b=456 了。**你把函数跑完之后，再把结果告诉我！**"_

这就是大模型工具调用的真相：**大模型只负责“规划意图”和“提取参数”，具体跑代码、拿结果的粗活，得咱们外部的 Python 程序自己去跑！**

#### 拓展观察：为什么多步计算只返回了一个工具调用？

细心的你可能发现了，用户问的是 `“先算乘法，然后再算加法”`，但大模型只下发了一个 `multiply` 指令。这是因为：
1. **模型的顺序思维**：模型读懂了“先...然后再...”的逻辑。它知道必须先拿到第一步的乘积结果，才能进行第二步评估。所以它暂时“扣住”了加法任务。
2. **缺乏执行循环 (Execution Loop)**：在我们的脚本中，向模型请求调用后就结束退出了。在真实的 Agent 中，我们要拿着第一步的结果**再次**去访问大模型，大模型才会下发第二步的加法工具调用。

#### 拓展观察：并行工具调用 (Parallel Tool Calling)

如果我们在 Prompt 里**打破先后顺序约束**，要求模型“同时立刻”做两件事：
`query2 = "请同时立刻帮我做两件事：计算123*456，以及计算25+13"`

你会惊艳地发现，大模型的 `response.tool_calls` 里同时出现了**两个**独立的指令：一个 `multiply`，一个 `add`！这就是现代旗舰大模型必备的**并行调用能力**，它极大地提高了处理复杂任务的效率！

---

### 步骤 3：实现人机互动的 Tool Execution Loop（执行循环）

在真实场景中，大模型在遇到需要工具的问题时只会返回 `tool_calls` 指令，它没有真正执行代码的能力（即无实体）。我们需要编写代码来接管这些指令，在本地把这些挂载的 Python 函数真正跑起来，然后再把结果作为 `ToolMessage` 喂回给大模型。

**代码实现** (`02_calculator_agent/3_agent_execution_loop.py`)：

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """这是一个乘法器。当你需要计算两个数字相乘时，请调用此工具。"""
    print(f"\n[🔧 本地工具 multiply 执行中... 检测到参数 a={a}, b={b}]")
    return a * b

@tool
def add(a: int, b: int) -> int:
    """这是一个加法器。请用来计算两个数字相加的和。"""
    print(f"\n[🔧 本地工具 add 执行中... 检测到参数 a={a}, b={b}")
    return a + b

tools = [multiply, add]
tools_by_name = {t.name: t for t in tools}

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

# 1. 模拟第一次提问：包含我们希望模型做多步计算的问题
messages = [
    ("user", "先计算123乘以456等于多少？算出结果后，把它加上25。")
]
print(f"[用户原始问题]: {messages[0][1]}\n")

# 2. 第一次调用：大模型决定发号施令，给出第一步操作
print("[第一回合：大模型思考中...]")
response_message = llm_with_tools.invoke(messages)
messages.append(response_message) # 把大模型的指令存进聊天记录里

# 3. 检查大模型是不是派发了任务 (tool_calls)
if response_message.tool_calls:
    print(f"[大模型决定不直接回答，而是分配了{len(response_message.tool_calls)}个本地工具执行任务]")
    
    # 4. 遍历大模型分配给我们的工具调用请求
    for tool_call in response_message.tool_calls:
        print(f"👉 决定调用工具: {tool_call['name']}, 参数准备: {tool_call['args']}")
        tool_name = tool_call['name']
        tool_args = tool_call['args']

        # 从我们本地注册的工具字典里找到对应的原生 Python 函数
        selected_tool = tools_by_name[tool_name]

        # 5. 【执行器】真正的动手跑代码发生在这里！
        tool_result = selected_tool.invoke(tool_args)
        print(f"[本地代码执行完毕！计算结果直接得出了: {tool_result}]")

        # 6. 【最关键的一步】：必须把计算结果包装成 ToolMessage，喂回给大模型！
        # 这个动作相当于跟大模型汇报：“首长，工具我替你跑完了，结果是这个，请过目！”
        from langchain_core.messages import ToolMessage
        messages.append(ToolMessage(
            tool_call_id=tool_call['id'], 
            name=tool_name, 
            content=str(tool_result) 
        ))
        
    # 7. 第二次调用：大模型根据拿到的工具执行结果继续思考
    print("\n[第二回合：大模型查收刚才的计算结果报告，并总结最终发言...]")
    final_response = llm_with_tools.invoke(messages)
    print("\n🎉【大模型的最终人类自然语言回复】:\n", final_response.content)

else:
    print("[大模型觉得这个问题不需要使用工具，直接输出自然语言回答]:", response_message.content)
```

**核心报错与知识点观察**：
运行这段代码，当你要求“先算乘法，再加25”时，经过两轮通信后 `final_response.content` 依然是空的，而是输出了一个全新的 `tool_call: {'name': 'add'}`！
这是因为我们代码里的 `if tool_calls:` 判断是**一次性**的。大模型拿到乘法结果后，发现题没做完，于是它下发了第二步指令（加法），但我们的代码在这时候**由于没有 while 循环，直接执行结束了**。
这也完美引出了**真正的 AI Agent 执行引擎**需要一个专门托管这个复杂死循环的容器！

---

### 步骤 4：引入真正的执行引擎 (AgentExecutor)

**目标**：解决人工硬核编写 `while` 循环去解析、调用、回传 ToolMessage 的繁琐工作。利用 LangChain 内置的 Agent 架构（`create_agent`）实现全自动的“推理-行动”循环。

**代码实现** (`02_calculator_agent/4_langgraph_agent.py`)：

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
# 引入 LangChain 预置的 Agent 循环引擎
from langchain.agents import create_agent

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """这是一个乘法器。当你需要计算两个数字相乘时，请调用此工具。"""
    print(f"\n[🚀 框架后台自动运行工具 multiply: a={a}, b={b}]")
    return a * b

@tool
def add(a: int, b: int) -> int:
    """这是一个加法器。请用来计算两个数字相加的和。"""
    print(f"\n[🚀 框架后台自动运行工具 add: a={a}, b={b}]")
    return a + b

tools = [multiply, add]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 1. 见证奇迹的时刻：把大模型和工具列表打包交给主管引擎
# create_agent 在后台帮我们写好了无比完备的 while 循环和 tool_message 拼接！
agent_executor = create_agent(llm, tools)

# 2. 我们只需要像面对一个普通模型那样，给它扔一句话就行了
query = "先计算123乘以456等于多少？算出结果后，把它加上25。"
print(f"[用户原始问题]: {query}\n")

# 3. 触发主管引擎运作
print("正在引擎内全自动多轮激战中，请观察打印日志...")
response_state = agent_executor.invoke({
    "messages": [("user", query)]
})

# 4. 直接获取它循环到底、直到完全算完后得出的最终结论！
# 框架返回的 state 中包含了所有的流转信息，最后一条 message 就是彻底完工的自然语言回复
final_response = response_state["messages"][-1]
print("\n🎉【大模型的最终人类自然语言回复】:\n", final_response.content)
```

**核心观察与知识点总结**：
1. **自动死循环 (The ReAct Loop)**：你可以在终端清楚地看到，不需要写任何 `if tool_calls` 或者 `for` 循环，引擎自动连续触发了 `multiply` 和 `add` 两个本地工具。它在后台默默地完成了我们上一步手动做的所有拼装脏活累活。
2. **Agent 引擎的更新**：在旧版 LangChain 中使用的 `AgentExecutor` 已经被统一整合进了现在的 `langchain.agents.create_agent`。这是目前最标准、最易用的用法。
3. **Gemini 的结构化返回 (List of Dicts)**：在使用原生 API 特性或特定的 Agent 循环时，Gemini 大模型返回的最终 `content` 可能不再是一个简单的纯字符串，而是由不同内容块组成的列表（List of Dictionaries）。比如 `[{'type': 'text', 'text': '计算结果是...'}]`。这也是自然语言，只不过是带有类型的富文本块。如果想提取纯字符串，可以在代码里直接索引 `final_response.content[0]['text']` 取出纯文本。

#### 深度进阶：关于工具调用的 3 个终极拷问

在完成 Agent 基础框架后，我们必须理清以下 3 个极其重要的底层机制：

1. **为什么 `ToolMessage` 必须传 `tool_call_id` 字段？如果去掉会发生什么？**
   - **底层机制**：`tool_call_id` 是大模型用来“对账”的唯一凭证。大模型可能会一次性下发多个工具调用指令（比如 ID=001 是买苹果，ID=002 是买香蕉）。当你把结果送回去时，如果不带 ID，模型根本不知道你送回来的是苹果的价格还是香蕉的价格。

     **代码演示说明**：
     ```python
     # 如果模型并发时发了两个完全同名工具的调用 (tool_calls)
     [
         {'name': 'multiply', 'args': {'a': 123, 'b': 456}, 'id': 'id-001'},
         {'name': 'multiply', 'args': {'a': 7, 'b': 8}, 'id': 'id-002'},
     ]

     # 错误示例：喂回结果时遗漏 tool_call_id
     messages.append(ToolMessage(
         # tool_call_id=tool_call['id'], 🚨 致命错误：这行被注释掉了
         name=tool_name, 
         content=str(tool_result) 
     ))
     ```
     > **灵魂拷问**：上面两个调用都叫 `multiply`，当你抽出时间算出 `56` 这个结果打算送回去交差时，模型怎么知道 `56` 是 `id-001` 的答案还是 `id-002` 的答案？
     > 
     > 这就是 `tool_call_id` 存在的根本意义。它是**这一次具体执行请求的唯一流水号**。而 `name` 只是工具的一个类别名。两者分工明确：`name` 说明你用了哪个工具，`id` 证明你回答的是哪道题。

   - **后果**：如果去掉，或者填错 ID，大模型所在的 API 服务端会直接抛出 **HTTP 400 Bad Request** 报错，告诉你消息格式损坏或对不上账，直接拒绝回答。这就是严格的协议级约束。

2. **带依赖关系的并行调用 (Parallel Tool Calling) 会同时执行吗？**
   - **场景**：用户问 `“请计算 123*456+25 的结果”`。模型会同时调用乘法和加法吗？
   - **机制**：**绝对不会！**因为加法的其中一个参数来自于乘法的结果，这两者在数学逻辑上存在强因果/时序依赖。大模型在“规划(Reasoning)”时非常聪明，它会自动退化为**顺序调用**：先发命令只算乘法 -> 收到人类发回乘法结果 -> 再次发命令算加法。
   - **只有当任务相互独立时**（例如第一题算乘法，第二题算另一个水果的数量），大模型才会真正并发返回两个 `tool_call`。

3. **如果用户需要的工具我们根本没有提供（比如没提供减法），模型会怎么办？**
   - **场景**：只给了 `add` 和 `multiply`，但用户问 `25 - 13 等于几`？
   - **大模型的应对策略（通常会混用）**：
     - **策略 A（变通 Hacker）**：高级模型（如 Gemini 2.5 Flash / GPT-4）拥有极强的联想能力，它会尝试自己变通，神奇地下发一条指令 `add(a=25, b=-13)` 来强行使用你现有的加法器实现减法！
     - **策略 B（脑内直出）**：如果问题极度简单（比如 25-13），它评估出“靠自己的词向量就能算出 12，且没有对应工具”，它会跳过 `tool_calls`，直接在自然语言 `response.content` 里回答“等于 12”。
     - **策略 C（诚实道歉）**：如果问题非常复杂（比如计算复杂的微积分）且没有工具可以变通，它会在自然语言里回复：“抱歉，你没有给我相关的微积分计算工具，我无法保证心算的准确性。”

---

## Phase 3: 工具与开放网络 API 交互 (Weather Agent)

在掌握了本地计算器 Agent 之后，我们需要让 Agent 具备真正的“联网”超能力。本阶段我们调用了真实的第三方公开 API（OpenWeatherMap）。

### 步骤 1：构建具有优雅错误处理的网络 API 工具

**目标**：演示如何编写一个极度健壮的、带有网络调用防御机制（防超时、防 404 挂起、智能语种转换）的外部 API 工具。

**代码实现** (`03_weather_agent/1_weather_tool.py`)：
```python
import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
openweather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")

@tool
def get_weather(city: str) -> str:
    """这是一个天气查询工具。当你需要查询指定城市的天气时，请调用此工具。
    Args:
        city: 想要查询的天气城市名称。⚠️警告：无论用户是用中文还是其他语言提问，你都必须在此处将城市名称翻译为纯英文全拼（如用户问"上海"，这里必须传入"Shanghai"），否则第三方天气接口将返回 404 错误。
    """
    # 坑点 1：缺少 API Key 的优雅处理
    if not openweather_api_key:
        return "【系统报错返回】由于系统后台没有配置 OPENWEATHERMAP_API_KEY，工具当前不可用。请向用户致歉。"
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={openweather_api_key}&units=metric&lang=zh_cn"

    try:
        # 坑点 2：调用外部网络请求时，务必加 timeout 网络超时时间！防止 Agent 死循环卡死
        response = requests.get(url, timeout=5)
        
        # 坑点 3：处理 404 等 HTTP 报错，绝不抛出崩溃，而是以人类自然语言转告大模型
        if response.status_code == 404:
            return f"【查询失败】[code: 404]找不到气象库中名为 '{city}' 的城市。请让用户核对拼写或尝试英文全拼。"
        elif response.status_code == 401:
            return "【查询失败】[code: 401] API Key 鉴权失败。请致歉。"
            
        response.raise_for_status()
        
        data = response.json()
        temp = data["main"]["temp"]
        weather_desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        
        return f"{city} 的当前实况天气：{weather_desc}，实时气温：{temp}℃，湿度：{humidity}%。你可以基于这些数据自由组合发散回复。"
        
    except requests.exceptions.Timeout:
        return "【网络超时返回】请求第三方天气服务耗时过长。请告诉用户网络开小差了。"
    except Exception as e:
        return f"【未知系统错误】调用天气 API 接口时出现代码级崩溃，详情: {str(e)}。"
```

### 步骤 2：组装气象 Agent 与底层的动态输出格式解析

**目标**：将网络 API 工具注册进大模型，并处理大模型在多模态与纯文本场景下的动态 JSON 结构解析。

**代码实现** (`03_weather_agent/2_weather_agent.py`)：
```python
import os
import importlib
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

# 因为文件名以数字开头，必须用 importlib 动态绕过 Python 导入限制
weather_module = importlib.import_module("1_weather_tool")
get_weather = weather_module.get_weather

load_dotenv()
tools = [get_weather]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
agent_executor = create_agent(llm, tools)

query = "我明天打算去北京出差，请帮我查一下那里现在的天气情况，并根据情况给我一些穿衣建议。"
print(f"[用户原始问题]: {query}\n")

print("正在引擎内全自动多轮激战中，请观察打印日志...")
response_state = agent_executor.invoke({
    "messages": [("user", query)]
})

# 优雅处理：应对大模型“薛定谔的返回值”类型推断
final_response = response_state["messages"][-1]
content = final_response.content

if isinstance(content, list):
    # 如果它是复杂列表，遍历寻找 type 为 text 的文本块抽取出来
    final_text = "".join(block["text"] for block in content if block.get("type") == "text")
else:
    # 如果 LangChain 帮我们折叠成了纯字符串，那就直接用！
    final_text = content

print("\n🎉【大模型的最终人类自然语言回复】:\n", final_text)
```

**核心观察：大模型返回体 `content` 的类型变频 (List of Dicts vs. String)**

当你连续调用气象 Agent 时，有时打印 `final_response.content` 是一长串类似 JSON 的 `List`（包含 `signature` 等复杂结构），有时却直接是一段干净的 `String`（纯自然语言文本）。

> **这是为什么？**
> 1. **大模型的动态返回体**：像 Gemini 这类原生多模态模型，标准的响应体本就是一个多内容块列表（List of Blocks）。当它的底层安全检查器生成了数字签名（Signature），或者包含引用元数据时，必须用结构化数据发送。
> 2. **LangChain 框架的自动折叠 (Auto-folding)**：LangChain 作为中间件，非常聪明但也容易让人迷惑：
>    - 如果模型返回了复杂的附加物体（如 Signature），LangChain 为了保全数据，会原封不动把 `content` 保持为 `List[Dict]`。
>    - 如果模型返回得极其干净，纯粹只为了说话，LangChain 会自作主张把它**自动折叠**成一个普通的 `String`。

**💡 工业级的高阶解法**：
如上方代码所示，绝对不能在代码里写死 `content[0]['text']`。必须使用 `isinstance(content, list)` 做类型防御，这是大模型客户端开发的必修课！
---

## 🏆 Phase 2 全体总结与核心收尾

我们通过 `02_calculator_agent` (纯本地计算工具) 和 `03_weather_agent` (网络 API 工具) 完整地打通了 Phase 2 的学习链路。在迈向更高阶的智能体协作（Phase 5 的 ReAct）之前，我们必须在大脑中建立以下三个不可动摇的底层认知：

### 1. Tool Calling (工具调用) 的本质究竟是什么？
**大模型本身不能也没有权限执行任何一行代码！**
工具调用的本质，是**一份“外包合同”**。
当你使用 `.bind_tools()` 将功能暴露给大模型时，你其实只是把这些 Python 函数的“说明书”（Schema，比如需要几个参数、是什么类型、有什么限制）发给了大模型。
大模型遇到问题时，通过阅读说明书，决定“哦！我做不了，但我算出参数了，包工头（程序）你去帮我跑一下这个叫 `multiply` 的活儿，跑完把结果告诉我”。
**执行代码的行为永远发生在你的本地环境，大模型只负责“规划意图”和“填参提取”。**

### 2. 本地工具 vs 网络 API 工具的防坑差异
- **本地工具（如加减乘除）**：相对安全、确定性高。主要防范参数类型报错（比如模型硬塞个字符串给要求 int 的加法器）。
- **网络 API 工具（如天气查询）**：极其脆弱、随时会挂。
  - **请求超时 (Timeout)** 必须显式设置，否则整个程序的死循环会被外网卡死。
  - **API 错误状态码 (404/401)** 绝不能作为 Exception 抛出导致程序崩溃，必须包装成友好的**人类自然语言**（如“找不到这个城市”）发回给大模型。
  - **利用 Prompt 进行参数约束**：OpenWeatherMap 对中文城市名支持极差（报 404）。最佳实践是直接在 `@tool` 的 `docstring` (说明书) 里写死一句警告：“务必将源地名翻译为纯英文再传入”。因为大模型阅读这段提示词后，拥有极强的中译英能力，这等于白嫖了一层翻译器防崩机制。

### 3. Agent 执行循环 (Execution Loop) 的终极模式
一个合格的 Agent 绝对不是 `if tool_calls:` 触发一次就结束的单线程脚本。
它是一个**生生不息的 `while` 死循环 (ReAct Pattern)**：

`获取问题` -> `大模型思考决策` -> `下发指令 (tool_calls)` -> `本地执行代码` -> `把结果装入 ToolMessage 喂回给大模型` -> `大模型重新思考` -> **(循环直至大模型认为不需要再调用工具，直接给出含有了 final_response 的最终回答)**

在工业界，我们不会手写这个复杂且容错率低的死循环，而是直接丢给 `langchain.agents.create_agent` (或底层更强大的 `LangGraph`) 这种图架构引擎去自动化托管。这就构成了现代 AI Agent 最核心的心智模型！

---
## Phase 3: 上下文与记忆机制 (Memory & Context)

大模型在物理底层是 100% 绝对无状态 (Stateless) 的。每一次 API 请求对它来说都是一次全新的宇宙重启。维持记忆的责任永远在客户端（也就是我们的代码）手里。本阶段我们将探索如何赋予 Agent 长短期的记忆力。

### 步骤 1：体验大模型的“金鱼记忆” (Stateless LLM)

**目标**：直观感受不带任何上下文外挂的裸模型是如何丢失历史对话记忆的。

**代码实现** (`04_memory_chatbot_v1/1_stateless_llm.py`)：
```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("🤖 聊天机器人已启动 (输入 'quit' 退出)")
print("-" * 50)

while True:
    user_input = input("\n你: ")
    if user_input.lower() == "quit":
        break
    
    # 直接丢给裸模型，没有任何历史记录上下文
    response = llm.invoke(user_input)
    print(f"AI: {response.content}\n")
```

**执行命令**：
```bash
python3 04_memory_chatbot_v1/1_stateless_llm.py
```

**核心观察 / 核心知识点总结**：
当你第一句话说“我叫XX”，第二句问“我叫什么”时，大模型会直接回答不知道。这说明底层 API 不会为你保存任何会话状态。这也引出了第一个破局思路：人为把前面的聊天记录全部发过去。

### 步骤 2：纯手工打造“人工短期记忆” (Manual Memory)

**目标**：通过在本地 Python 列表里维护历史记录队列，手动在每次提问时将“全部上下文”发给模型，实现人工记忆。

**代码实现** (`04_memory_chatbot_v1/2_manual_memory.py`)：
```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 1. 在客户端维护一个列表，专门用来存储聊天历史
chat_history = []

print("🤖 聊天机器人已启动 (输入 'quit' 退出)")
print("-" * 50)

while True:
    user_input = input("\n你: ")
    if user_input.lower() == "quit":
        break
    
    # 2. 用户说的话，追加到历史记录里
    chat_history.append(HumanMessage(content=user_input))
    
    # 3. 把累积了所有上下文的完整的 chat_history 列表一股脑发送给模型！
    response = llm.invoke(chat_history)

    # 4. 把模型的回应也追加到历史记录里，形成闭环！
    chat_history.append(AIMessage(content=response.content))

    print(f"AI: {response.content}\n")
    print(f"  [幕后揭秘：当前发送给模型的聊天记录条数：{len(chat_history)} 条]")
```

**执行命令**：
```bash
python3 04_memory_chatbot_v1/2_manual_memory.py
```

**核心观察 / 核心知识点总结**：
- **记忆的本质**：我们在每一轮对话中，送给大模型的永远是 `[Msg1, Msg2, Msg3...]` 的全量记录。模型依然是无定式 (stateless) 的，它只是通过阅读你刚刚传过去的所有历史大纲，做出了符合上下文情境的延续性生成。
- **字典列表 vs Message 对象**：直接传原生的 JSON 字典 `{"role": "user", "content": "..."}` 和传 LangChain 的 `HumanMessage` 对象产生了完全一致的行为。这是因为 `HumanMessage(BaseMessage)` 在底层 `type` 已经被严格定死为 "human"，并在 LangChain 发起底层 API 握手时被转换回了原生字典。之所以坚持使用对象而不是原始字典，是因为 Message 对象能够承载多模态数据（图片）、额外的 kwargs（如示例提示），还可以安全地搭载 ToolMessage 等复杂的系统级结构体。

### 步骤 3：引入自动化记忆胶囊 (RunnableWithMessageHistory)

**目标**：解决手工管理历史记录列表（`chat_history.append(...)`）的繁琐问题，使用 LangChain 的流水线封装胶囊实现记忆管理的自动化。

**代码实现** (`04_memory_chatbot_v1/3_auto_memory.py`)：
```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

store = {}

def get_session_memory(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 最激动人心的一步：把 "没记忆的 llm" 封装进 "记忆胶囊"
llm_with_memory = RunnableWithMessageHistory(
    llm, 
    get_session_memory 
)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
        
    response = llm_with_memory.invoke(
        user_input, 
        config={"configurable": {"session_id": "user_rick"}}
    )
    print(f"AI: {response.content}\n")
```

**执行命令**：
```bash
python3 04_memory_chatbot_v1/3_auto_memory.py
```

**核心观察 / 核心知识点总结**：
当你抱怨“代码变简单了，但我怎么感觉我只是在学库的用法，大模型底层的行为全被遮蔽了？”时，你触碰到了架构师必须要面对的**“抽象陷阱”**。

无论框架（LangChain, AutoGen, LangGraph）的语法糖多么花哨，它们对记忆的底层处理逻辑有且仅有一种：**在发起 API 请求的那一刻前，将你的新问题与缓存中的旧历史拼接成一个巨型的主题数组（Message List），然后整体发送**。

**终极破局解法：全局透视法**
为了不被框架绑架，我们可以随时开启“上帝视角”。在任何 LangChain 脚本的顶部（`load_dotenv()` 之前）添加以下两行代码：
```python
from langchain_core.globals import set_debug
set_debug(True)
```
控制台会立刻向你展示最赤裸的真相：所有被掩盖的拼装动作、注入的 System Prompt、工具说明书的构造，以及最终通过 HTTP 发送给大模型的浩如烟海的 JSON Payload 会一字不差地打印出来。
通过 `Debug = True` 模式，我们可以做到**享受框架带来的提效语法糖，但灵魂深处依然死死拿捏住大模型的第一性原理**。
**实例代码与 Debug 视角对比**

你在终端的交互看起来是连续的：
```text
You: Hello，I'm Rick
AI: Hello Rick! Nice to meet you. I'm an AI assistant. How can I help you today?

You: what is my name?
AI: Your name is Rick! You told me that earlier.
```

但在开启 `set_debug(True)` 后的底层视角下，你会看到这一切的虚幻：

**第一回合：**
当你说出 "Hello, I'm Rick" 时，LangChain 发送的 payload 是干净的：
```json
{
  "prompts": [
    "Human: Hello，I'm Rick"
  ]
}
```

**第二回合（核心奥秘）：**
当你说出 "what is my name?" 时，LangChain 触发了 `RunnableWithMessageHistory`。它去内存字典里把之前的对话翻了出来，**生硬地拼装在一起**，最终发给大模型的 payload 变成了这样一个包含所有历史的“巨型 prompt”：

```json
{
  "prompts": [
    "Human: Hello，I'm Rick\nAI: Hello Rick! Nice to meet you.\n\nI'm an AI assistant. How can I help you today?\nHuman: what is my name?"
  ]
}
```

通过这几行日志，你可以清晰地得出结论：**大模型根本没有任何所谓的“记忆”。它只是像做阅读理解一样，在每一回合里重新阅读了由 LangChain 偷偷塞给它的“包含了迄今为止所有聊天记录的一篇长文”而已。**

### 步骤 4：打造可持久化的长期记忆 (Persistent Long-Term Memory)

**目标**：解决内存字典（`InMemoryChatMessageHistory`）在进程被杀死后记忆清空的问题，将用户的聊天记录永久存入本地 SQLite 数据库中。

**代码实现** (`05_memory_chatbot_v2/1_sqlite_memory.py`)：
```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
# 引入连接硬盘数据库的全新记忆力类
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.globals import set_debug

set_debug(True)
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def get_session_memory(session_id: str):
    # 直接返回一个连接到本地 SQLite 数据库的历史记录对象
    # 如果库或表不存在，它会自动建库建表
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db" # 数据库存放在本地当前目录
    )

llm_with_memory = RunnableWithMessageHistory(
    llm,
    get_session_history=get_session_memory
)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
        
    response = llm_with_memory.invoke(
        user_input, 
        config={"configurable": {"session_id": "user_rick"}}
    )
    print(f"AI: {response.content}\n")
```

**执行命令**：
```bash
python3 05_memory_chatbot_v2/1_sqlite_memory.py
```

**核心观察 / 核心知识点总结**：
如果我们在第一次对话后直接杀死进程，再次启动程序时，Agent 依然能准确叫出我们的名字。
它的工作流是这样的：
1. **启动时**：不再创建空的 `store = {}`。
2. **提问时**：底层的 `SQLChatMessageHistory` 截获 `session_id`（如 "user_rick"），**向本地 SQLite 数据库发起 `SELECT` 查询**，捞出这个用户的过往记录。
3. **调用 LLM**：将新问题和从数据库捞出的老记录拼装（如步骤 3 所述的 payload 模式）发给 API。
4. **保存记录**：大模型返回后，再次向数据库发起 `INSERT` 操作，把这轮对话存盘。

由此可见，AI 真正变成了一个带有状态的“后端服务器”应用。它的记忆外挂正式从“内存级”进化到了“工业硬盘数据库级”！

### ⚠️ 工业级隐患：失控的上下文（The Context Window Trap）

现在，你拥有了一个即使关电脑拔电源也不会失忆的机器人。
**但是，深渊正在凝视着你！**
假设你和这个机器人就这样日复一日地聊了整整一年，`chat_history.db` 里的记录达到了 10 万条。
明天你再问它“今天天气如何”的时候，LangChain 的 `RunnableWithMessageHistory` 会做什么？

根据我们在“Debug 真相”里学到的第一性原理：**它会把你以前聊过的所有 10 万条历史记录，和今天的新问题拼在一起，合成一台多达几十万 Token 的“超级巨无霸发送机”，一次性全砸给大模型 API！**

这会导致三种极其惨烈的工业事故：
1. **超出 Context Window**：API 拒绝响应，直接崩溃报错（比如 `Token Limit Exceeded`）。
2. **天价账单**：大模型 API 是按 Token 计费的。你每次问一句哪怕只有两个字的新问题，它都会带着前面那十万句废话去请求，你点一下回车键可能就要烧掉几十块钱人民币。
3. **注意力迷失 (Lost in the Middle)**：业界公认的 LLM 缺陷，当喂给大模型的上下文太长时，它只记得开头和结尾，中间提取信息的准确率会出现断崖式下跌。

这也就是我们 `05_memory_chatbot_v2` 后半部分的终极挑战：
> 如何既能保住硬盘里的长线历史，又能在发送给大模型 API 时进行截断（Truncation）或摘要浓缩（Summarization）？

### 步骤 5：挽救天价账单 - 历史聊天截断术 (Message Truncation)

**目标**：在发送给大模型之前，通过切断过于久远的历史消息，保持 Payload 大小永远稳定在一个安全、便宜的区间内。



**代码实现** (`05_memory_chatbot_v2/2_truncate_memory.py`)：
```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.globals import set_debug

set_debug(True)
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 声明一个记忆修剪器 (Trimmer)
trimmer = trim_messages(
    max_tokens=40,   # 为了方便测试，设置得很短
    strategy="last", # 从后往前数，保留最新消息
    token_counter=llm, # 把计算每个字占多少 token 的工作交给这门模型自己
    include_system=True, # 永远保留 System Prompt 
)

# 使用管道符把 修剪器 和 llm 拼在一起
chain_with_trimming = trimmer | llm

def get_session_memory(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db" 
    )

llm_with_memory = RunnableWithMessageHistory(
    chain_with_trimming,
    get_session_history=get_session_memory
)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
        
    response = llm_with_memory.invoke(
        user_input, 
        config={"configurable": {"session_id": "user_rick"}}
    )
    print(f"AI: {response.content}\n")
```

**执行命令**：
```bash
python3 05_memory_chatbot_v2/2_truncate_memory.py
```

**核心知识点深挖：Token 到底是怎么计算的？ (`max_tokens=40`)**

在前面的代码中，有一个极其隐蔽但关键的参数：`token_counter=llm`。

1. **什么是 Token？**
Token 并不是字数，也不是字母数。它是大模型底层识别文本的“最小分词单位”（Subword）。
粗略估计下：
- 1 个 Token ≈ 0.75 个英文单词（例如 "hamburger" 可能被切成 "ham", "bur", "ger" 3个 token）
- 1 个 Token ≈ 0.5 个中文字符（中文分词比英文碎得多，所以中文请求往往更费 token 钱）

2. **为什么计算这么复杂？**
每一家大模型公司（OpenAI, Google, Anthropic）底层的分词器（Tokenizer）算法都是**完全不一样**的！
同样一句话 "你好，Rick"，在 OpenAI 家算出来可能是 5 个 token，在 Google 家算出来可能是 8 个 token。

3. **`token_counter=llm` 的妙用**
正是因为这个原因，LangChain 的 `trim_messages` 不敢自己去瞎算字数。我们必须把 `llm` 对象通过 `token_counter` 传给它。
在底下，LangChain 会调用这个特定的 `GoogleGenerativeAI` 的官方 Python SDK 提供的方法，去精准计算这条记录在 Google 眼里究竟沾了多少个 Token。
这样截断出来的大小，才是对这家 API 公司最精确、最安全的防超载边界。


### 步骤 6：化繁为简 —— 记忆摘要压缩术 (Message Summarization)

**目标**：解决截断（Truncation）带来的“一旦截掉就彻底遗忘”的问题。我们在长历史记录发送给 LLM 前，先唤醒一次 LLM 将以前的烂账压缩成一句干练的“前情提要”摘要。

**代码实现** (`05_memory_chatbot_v2/3_summary_memory.py`)：
```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.globals import set_debug

set_debug(True)
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 手动写一个记忆摘要拦截器
def summarize_messages(messages):
    print(f"\n[摘要器] 当前拦截到 {len(messages)} 条历史记录。")
    if len(messages) <= 3:
        return messages
    
    print("[摘要器] 触发阈值！正在调用大模型将旧对话压缩成摘要...")
    old_messages = messages[:-2]
    recent_messages = messages[-2:]
    
    # 将老消息合成一段 Prompt 专门用作总结
    summary_prompt = "请用一句简短的话总结以下对话的核心内容，以第三人称描述：\n"
    for msg in old_messages:
        summary_prompt += f"{msg.type}: {msg.content}\n"
        
    # 偷偷调用一次大模型获取压缩摘要
    summary_response = llm.invoke(summary_prompt)
    print(f"[摘要器] 压缩完成 -> {summary_response.content}")
    
    # 伪装成一条干练的 SystemMessage 系统提示音塞进去
    summary_msg = SystemMessage(content=f"【前情提要（记忆摘要）】：{summary_response.content}")
    return [summary_msg] + recent_messages

# 把手写的拦截器和 LLM 拼成一条处理管道
chain_with_summary = summarize_messages | llm

def get_session_memory(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history_summary.db" 
    )

llm_with_memory = RunnableWithMessageHistory(
    chain_with_summary,
    get_session_history=get_session_memory
)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
        
    response = llm_with_memory.invoke(
        user_input, 
        config={"configurable": {"session_id": "user_rick"}}
    )
    print(f"AI: {response.content}\n")
```

**执行命令**：
```bash
python3 05_memory_chatbot_v2/3_summary_memory.py
```

**核心观察 / 核心知识点总结**：
- **Truncation vs Summarization 的取舍**：截断会彻底丢失长线信息（如你第一轮给出的名字，第100轮时被裁掉），而摘要相当于让大模型预处理提取要素，既省了 Token 钱，又保住了关键事实。
- **动态滚动摘要 (Rolling Summarization)**：当我们做摘要时，如果每次都把成千上万条“未经摘要的原生 old_messages”拿去重新算一遍摘要，依然会面临 Token 爆炸的问题。**真正的解法是“增量摘要”（或滚动摘要）**。即：如果历史记录中已经存在上一轮的 `SystemMessage(摘要)`，我们只需要让大模型做这样一道题：`"这是旧的摘要：" + 旧摘要内容 + "，这是刚才发生的两句新对话：" + 两条新消息 + "。请据此生成一份融合后的最新摘要。"` 这样不管聊多少年，做摘要时的算力开销永远是一个极小、极稳定的常量值。这就是高端 AI 产品节约成本的终极秘诀。

---

## 🏆 Phase 3 全体总结与核心沉淀

在 Phase 3 中，我们彻底攻克了 AI Agent 开发中最为关键和最易踩坑的“记忆力”（Memory）管理。这也是所有对话式产品（如 ChatGPT, Claude）后端的基石。

1. **核心第一性原理：Stateless（无状态）**
   - 大模型 API 在物理层面是 **百分百金鱼记忆**。每一次请求都是全新的开始。
   - 所有所谓的“记忆”，其本质都是：**在每一次请求时，由客户端代码将【过去的所有聊天记录】无情地拼合在【当前的新问题】之上，构造成一个极其庞大的 Payload 一同发送给大模型。**
   
2. **短期记忆到长期记忆的跃迁**
   - **`InMemoryChatMessageHistory`**：把对话存在 Python 字典里，一旦电脑死机或进程重启，所有聊天数据灰飞烟灭。
   - **`SQLChatMessageHistory`**：将记录持久化到本地 SQLite 数据库中。当客户端重启后，新的一对一发起对话时（带上具体的 `session_id`），程序会自动去数据库 `SELECT` 过往的烂账，拼装并发送，实现了跨越生死的真正长期记忆。
   
3. **上下文爆炸与工业级防御术：截断与摘要**
   - 伴随着日积月累的持久化记忆，必然会遭遇 **Context Window Trap（上下文溢出）** 引发的报错崩溃、天价 Token 账单，以及不可避免的 Lost in the Middle（注意力迷失）问题。
   - **拦截方案 1：Truncation (切断)** (`trim_messages`)。简单直接地只保留最近的 N 条对话或者最后 M 个 Token。适合短剧场对话，不适合记长笔记。值得注意的是，Token 计算由于不同平台分词器（Tokenizer）算法差异，必须注入绑定特定的 `llm` 以做到百发百中精确切割。
   - **拦截方案 2：Summarization (增量摘要压缩)**。用一个手写的 Python 函数做中间人。通过抽出之前的烂账对话，让底层大模型额外“打一次黑工”生成一层简短的【前情提要】，然后替换掉成堆的历史再传输下去。在最新一代架构中，使用滚动重叠算法，可以保障发送和做摘要的计算成本一直恒定在低水位。这也是工业界终极推荐的王道打法。

---


# Phase 4: 检索增强生成 (RAG - Retrieval-Augmented Generation)

## 步骤 1：第一局 - 内存级基础 RAG 问答体验

**目标**：打破大模型自身的知识盲区和时间限制，赋予它读取我们本地私有文档的能力。在这个极简示例中，我们将演示完整的 RAG 黄金三步：Indexing (索引/向量化) -> Retrieval (检索提取) -> Generation (拼接生成)。

**代码实现** (`06_simple_rag/1_basic_rag.py`)：
```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_debug

set_debug(True)

# 1. Indexing (载入与向量化入库)
loader = TextLoader("06_simple_rag/my_secret_data.txt", encoding="utf-8")
docs = loader.load()

# 将长文档切碎
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = text_splitter.split_documents(docs)

# 调用 Embedding 模型将文本块转为高维向量，存入内存级别的向量数据库
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

# 2. Retrieval (检索相关碎片)
query = "Rick用大葱换了什么交通工具？它的开机密码是多少？"
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retrieved_docs = retriever.invoke(query)

# 合并相关碎片
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

# 3. Generation (组装 Prompt 让 LLM 结合上下文作答)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
rag_prompt = PromptTemplate.from_template("""
请严格根据下方的【参考资料】来回答用户的问题。
【参考资料开始】
{context}
【参考资料结束】

用户问题：{question}
""")

final_chain = rag_prompt | llm
response = final_chain.invoke({"context": context_text, "question": query})
print(response.content)
```

**执行命令**：
```bash
python3 06_simple_rag/1_basic_rag.py
```

**核心观察 / 核心知识点深挖**：

1. **底层欺骗术**：大模型本质上并没有真的像人类一样“通读并领悟”了这本小说或说明书。它依然是一个无状态的翻译机。我们只是通过数学手段（Embedding向量距离计算），在数据库里算出了和用户问题最相关的几个段落（Chunk），然后硬生生把它塞到了发给 LLM 的 Payload（上下文 Context）里。LLM 只是根据我们硬塞给它的参考资料做了一次小学级别的“阅读理解”罢了。
2. **文本分割器的心机 (`RecursiveCharacterTextSplitter`)**：
   - 参数 `chunk_size=50` 的计算逻辑是 **“字符串的字符数量”** (即 Python 里的 `len(string)`)，**不是 Token 数量，也不是文件大小 Byte**。不管是中文还是英文字母，1 个字符长度就是 1。
   - `chunk_overlap=10` 指的是两个碎片的边缘重叠区。为了防止一句话正好在中间被一刀切断（导致句意断裂，向量计算大减分），我们需要让相邻的两块碎肉有互相重叠交织的地方。

3. **中文切词陷阱 (The Chinese Text Splitter Trap)**：
   - `RecursiveCharacterTextSplitter` 名字里的 `Recursive` (递归) 意味着它是按顺序尝试切断文本的。LangChain 默认的切刀顺序是：`["\n\n", "\n", " ", ""]`。
   - **为什么会出现单独的 "Rick" 碎片？** 因为这是按英语习惯设计的。英语里词与词之间有空格，系统会先用换行切，再用空格切，尽量保证切出来的每一小块都是完整的单词（不把单词拦腰切断）。但在中文里，几百个字的句子可能连一个空格都没有！此时系统把这几百字当成了一个“超级长的英语单词”，但在你的文本中 `"Rick 乘坐银河..."` 这里刚好有一个空格。系统极其高兴地切下了这一刀，把 `"Rick"` 单独剥离了出来。而后面那段长达 100 多个中文字的半截残党，因为没有空格，系统被迫触发最后一站：`""`（逐字符硬砍）。所以你会看到极其诡异的切分。
   - **解法**：做中文 RAG 时，**必须改写切刀（separators 参数）**，加入中文标点符号！
   ```python
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=50, 
       chunk_overlap=10,
       separators=[
           "\n\n",
           "\n",
           "。", # 先按中文句号切
           "！",
           "？",
           "，", # 再按中文逗号切
           " ",  # 英文空格
           ""    # 最坏情况：死磕字符
       ]
   )
   ```
   加入了中文标点后，文本就会极其优雅地按照意群被切割，大幅提升后续的向量搜索精准度。

## 步骤 2：第二局 - 接入工业级 ChromaDB 与 MMR 检索抗重叠

**目标**：抛弃玩具级别的内存数据库，解决每次启动都要重新按字符切断、重新调 Embedding 模型算几十万次向量的性能惨剧。引入真实磁盘向量库，并引入高级检索策略解决连环提问。

**核心知识点深挖**：
1. **持久化向量数据库 (Persistent Vector Store)**：
   - 使用 `Chroma.from_documents(..., persist_directory="./my_local_vectordb")`。这辈子只需花钱、花算力运行一次。它把所有句子的超高维坐标永久写死在硬盘里。
   - 这实现了真正的**“索引/检索解耦”**。推理服务启动时，只需 `Chroma(persist_directory=...)` 挂载硬盘就能立刻开喷，0 延迟启动。
2. **多核连环提问惨剧与 MMR (Maximal Marginal Relevance) 破局**：
   - 痛点：面对用户四连问（“换了什么？密码多少？外星人结局？小明送了啥？”），如果只使用普通相似度搜索并限制 `k=2`（只取最像的两条），注定会遗漏关键信息，导致 LLM 胡编乱造。
   - 但如果我们简单粗暴把 `k=20`，如果其中有 15 条讲的全是同一件事的废话，就会浪费海量 Token 钱，还会干扰大模型。
   - **MMR 的神之一手**：`search_type="mmr"`。它的数学逻辑是：先贪婪地抓取一大批（比如 `fetch_k=10`）相关记录，然后在内部进行去重惩罚。如果 A 和 B 意思差不多，它会把 B 踢掉，强迫自己必须找一些不仅跟问题相关，而且互相之间**提供不同信息维度（不重复）**的句子。这几乎是全行业提升检索召回率（Recall）的保底绝技。

---

## 🏆 Phase 4 全体总结与核心沉淀

RAG 的精髓从来不是训练大模型，而是**“如何在一座超级图书馆里，精准地把那一张藏着答案的纸条递给一个没有任何记忆的聪明天才”**。

你掌握了：
1. `RecursiveCharacterTextSplitter` 第一性原理：哪怕一段几百字的文本没有任何空格，也不该被盲切。必须给 `separators` 加上 `["。", "！", "？", "，"]` 这几个中文标点灵魂。
2. 彻底跑通了 Indexing / Retrieval / Generation 三层漏斗。
3. 学会了用 ChromaDB 这个真刀真枪的本地服务把大资产持久化。
4. 用 MMR 策略打败了单维度复读机，搞定了超级复杂的并发四连问提问。

