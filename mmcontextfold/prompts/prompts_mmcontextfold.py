"""
prompts_mmcontextfold.py

MM-ContextFold prompts: MAIN-BRANCH architecture with convergence-focused design.

MAIN: Research strategist with global budget awareness.
BRANCH: Task executor with budget / evidence / anti-loop / source-quality rules.
"""

import re
import json


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a research agent that answers questions by planning and executing "
    "focused subtasks. You operate in two modes: MAIN (planner) or BRANCH (executor)."
)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN Prompts
# ═══════════════════════════════════════════════════════════════════════════

MAIN_USER_PROMPT = """**MODE: MAIN** — You are the research strategist.

## Workflow
1. Analyze the question and image descriptions
2. Identify what information is still missing
3. Create a focused subtask to gather it
4. Review subtask results, then create another subtask or answer
5. Keep a global round budget in mind; when budget is low, request a synthesis-focused subtask or provide the best calibrated answer.

## Creating a Subtask
Output ONE subtask per response. The executor (BRANCH) will inherit your full context and can use web search, page reading, and reverse image search.

<subtask>
{"description": "3-5 word summary", "prompt": "Clear task prompt with objectives. State what to search for, what specific information to extract, and what to return.", "assigned_images": ["<image_0>"]}
</subtask>

**`assigned_images` guide:**
- Only assign images that the subtask **directly needs** (e.g., for reverse image search, reading text/numbers, identifying visual details).
- For pure text-based research (e.g., searching a name or fact), use an empty list `[]` — no images needed.
- Each image is only loaded inside the BRANCH that receives it, keeping context focused.
- **When you assign images, write the `prompt` to explicitly instruct the executor to use `lens_search` first** (e.g., "Use lens_search on <image_0> to identify ..., then search for ...").

## Answering
When evidence is sufficient:
<answer>your final answer</answer>

## Image Context
{image_context}

## Question
{question}
"""


MAIN_CONTINUE = """## Subtask [{description}] completed

{return_message}

---

Review the findings above. Then take **exactly one** action:

**Option A — Create another subtask:**
<subtask>
{"description": "...", "prompt": "...", "assigned_images": [...]}
</subtask>

**Option B — Provide your final answer:**
<answer>your final answer</answer>
"""


MAIN_FORCE_ANSWER = (
    "Maximum research rounds reached. You MUST provide your best answer now "
    "based on all evidence gathered.\n<answer>your final answer</answer>"
)


# ═══════════════════════════════════════════════════════════════════════════
# BRANCH Tool Definitions
# ═══════════════════════════════════════════════════════════════════════════

_TOOL_SEARCH = {
    "name": "search",
    "description": "Perform a Google web search. Use when you need factual information, news, dates, or names.",
    "parameters": {
        "query": "string - The search query",
        "goal": "string - What specific information you want to find"
    }
}

_TOOL_VISIT = {
    "name": "visit",
    "description": "Visit a URL to extract detailed content. Use when search snippets are insufficient.",
    "parameters": {
        "url": "string - The full URL to visit",
        "goal": "string - What information you're looking for on this page"
    }
}

_TOOL_LENS_SEARCH = {
    "name": "lens_search",
    "description": "Perform Google Lens reverse image search. Use to identify where an image comes from, or find visually similar images.",
    "parameters": {
        "image_id": "string - Image ID to search, e.g. \"<image_0>\"",
        "goal": "string - What you want to find about this image"
    }
}

_TOOL_RETURN = {
    "name": "return",
    "description": "Report findings back to MAIN. Call this when the task is complete.",
    "parameters": {
        "message": "string - Concise findings with key evidence and sources"
    }
}

_BRANCH_TOOLS = [_TOOL_SEARCH, _TOOL_VISIT, _TOOL_LENS_SEARCH, _TOOL_RETURN]


def _format_branch_tools() -> str:
    return "\n".join(json.dumps(t, indent=2) for t in _BRANCH_TOOLS)


BRANCH_PROMPT = f"""**MODE: BRANCH** — Execute the assigned research task.

## Available Tools
<tools>
{_format_branch_tools()}
</tools>

## Response Format
Think step-by-step, then call **one tool** per response:
<tool_call>
{{"name": "tool_name", "parameters": {{"key": "value"}}}}
</tool_call>

## Rules
1. Focus exclusively on the assigned task.
2. You may call multiple tools in sequence (one per response).
3. When done, you **must** call `return` to report findings back to MAIN.
4. Budget: finish within at most 5 tool calls; if uncertainty remains, call `return` with the best-supported conclusion and explicit uncertainty.
5. Evidence policy: before `return`, try to validate the key claim with at least two independent signals (e.g., lens_search + search, or search + visit).
6. Anti-loop: do not repeat near-duplicate queries more than twice; if results are noisy, pivot strategy, then conclude.
7. Source quality: prefer official or domain-relevant sources over forums/aggregators when evidence conflicts.

## Task
{{task_prompt}}
"""


# ═══════════════════════════════════════════════════════════════════════════
# Init Perception
# ═══════════════════════════════════════════════════════════════════════════

INIT_PERCEPTION_PROMPT = """Briefly describe this image for a research planner who cannot see it.

1. What the image shows (subject, type of content)
2. Any visible text or numbers
3. What type of source it appears to be from

Output as JSON:
```json
{"summary": "...", "source_hint": "..."}
```"""


# ═══════════════════════════════════════════════════════════════════════════
# Parsers
# ═══════════════════════════════════════════════════════════════════════════

def extract_subtask(text: str) -> dict | None:
    if not text:
        return None
    match = re.search(r'<subtask>\s*(.*?)\s*</subtask>', text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        if not isinstance(data, dict) or not data.get("prompt"):
            return None
        return data
    except json.JSONDecodeError:
        return None


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_tool_call(text: str) -> dict | None:
    if not text:
        return None
    matches = list(re.finditer(
        r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL
    ))
    if not matches:
        return None
    try:
        return json.loads(matches[-1].group(1))
    except json.JSONDecodeError:
        return None
