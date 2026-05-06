import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.globals import set_debug
from pydantic import BaseModel, Field

set_debug(True)
load_dotenv()

gemini_model = os.getenv("GEMINI_API_MODEL")

# 1. 强制规范裁判的打分表单结构（使用 Pydantic）
class GradeResult(BaseModel):
    score: int = Field(description="给AI的代码打分，范围0-100")
    reason: str = Field(description="给出扣分或满分的详细理由，字数100字以内")

# 2. 召唤更严格的裁判模型
# ⚠️ 进阶知识点：做评估的裁判模型，必须强制降温 (temperature=0.0)，确保打分绝对稳定可复现！
llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.0)

# 把裁判变成了必须按表单结构填写的“打分机器”
structured_judge = llm.with_structured_output(GradeResult)

# 3. 编写裁判的批改准则
def evaluate_code(task: str, output: str):
    eval_prompt = (
        f"你是一名极度严格的资深技术总监，你需要对员工产出的代码进行评分。\n\n"
        f"【原始任务】\n{task}\n\n"
        f"【员工产出】\n{output}\n\n"
        f"【评分标准 (Rubric)】\n"
        f"请基于以下尺度打分（0-100分）：\n"
        f"- 90-100：算法逻辑完全正确，考虑了边界条件，且完全遵守题目限制。\n"
        f"- 70-89：逻辑基本跑通，但存在性能问题、可读性问题，或微小的边界 Bug。\n"
        f"- 0-69：存在严重逻辑漏洞，或耍小聪明（如直接调用 Python 内置 API 代替手写算法）。\n\n"
        f"请审查代码，严格遵照上述评分标准给出你的分数和评语。"
    )

    result = structured_judge.invoke(eval_prompt)
    return result

if __name__ == "__main__":
    task = "用Python写一个快速排序算法"
    # 这里伪造一个表现拉垮的 Agent 原产出
    bad_code = "def quick_sort(arr): return sorted(arr) # 谁说这不是快排呢？"
    
    print("👩‍⚖️ 裁判正在审阅中...")
    report = evaluate_code(task, bad_code)

    print(f"\n[最终得分]: {report.score}")
    print(f"[裁判毒舌]: {report.reason}")