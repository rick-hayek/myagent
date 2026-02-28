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

**运行命令**：
```bash
python3 01_basic_api/1_ai_sdk_call.py
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

**运行命令**：
```bash
python3 01_basic_api/2_langchain_basic.py
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
llm = ChatGoogleGenerativeAI(model="gemini-3-flash")

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
print("\n[底层给大模型看的数据结构 (Schema)]:\n", multiply.args_schema.schema())

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

---
