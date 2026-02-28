---
description: Automatically update learning_tutorial.md to document theoretical concepts and code
---

# Update Tutorial Workflow

Always automatically apply this workflow when making progress on a project step, resolving a conceptual misunderstanding, modifying code related to the learning path, or summarizing key concepts. 

### Trigger Conditions
- When the user successfully runs a new piece of code and asks to move on.
- When the user asks for a conceptual summary or asks a deep "Why" question about mechanisms (like JSON Parser vs Structured Output).
- When transitioning to a new step or phase in the `implementation_plan.md` or `task.md`.

### Steps
1. Use the `view_file` tool to open and review `/Users/rick/src/myagent/documents/learning_tutorial.md`.
2. Construct a new section or update an existing section corresponding to the latest discussion and code.
3. Ensure the structure follows the established pedagogical conventions in this document:
   - **目标**: Clear objective of the specific step
   - **代码实现**: The core code snippet (within `python` blocks) and its filepath (`filename.py`)
   - **执行命令**: The exact Bash commands to execute the script
   - **核心观察 / 核心知识点总结**: The philosophical/theoretical mechanisms, developer takeaways, or summaries of bug reports/misunderstandings encountered and fixed with the user. **This is the most important part of the learning framework**.
4. Use the `replace_file_content` or `multi_replace_file_content` tool to append/inject this newly written content seamlessly into the `/Users/rick/src/myagent/documents/learning_tutorial.md` file.
5. Notify the user once the tutorial has been successfully updated and clearly state that the knowledge has been immortalized.
