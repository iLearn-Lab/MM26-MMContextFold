"""
pipeline_mmcontextfold.py

MM-ContextFold Pipeline — MAIN-BRANCH architecture with convergence-focused design.

Key design:
- MAIN prompt: global budget awareness, research strategist
- BRANCH prompt: budget / evidence / anti-loop / source-quality rules
- Visual-Augmented Context Folding for multimodal information retrieval
"""

import os
import re
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

from pipelines import register_pipeline
from pipelines.base_pipeline import BasePipeline, PipelineConfig, SampleLogger, log, set_logger
from utils.search_backbone_model import ChatClient_OpenRouter
from utils.search_tools import SearchTools
from configs.api_keys import OPENROUTER_API_KEY

from prompts.prompts_mmcontextfold import (
    SYSTEM_PROMPT,
    MAIN_USER_PROMPT,
    MAIN_CONTINUE,
    MAIN_FORCE_ANSWER,
    BRANCH_PROMPT,
    INIT_PERCEPTION_PROMPT,
    extract_subtask,
    extract_answer,
    extract_tool_call,
)


@dataclass
class MMContextFoldConfig(PipelineConfig):
    max_branches: int = 5
    max_branch_turns: int = 10


class MMContextFoldAgent:
    """
    MM-ContextFold Agent — MAIN-BRANCH branching + visual-augmented context folding
    """

    def __init__(
        self,
        question: str,
        image_paths: List[str],
        dataset_name: str,
        output_dir: str,
        sample_id: str,
        model_name: str = "google/gemini-2.5-flash",
        config: MMContextFoldConfig = None,
    ):
        self.question = question
        self.image_paths = image_paths
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.sample_id = sample_id
        self.config = config or MMContextFoldConfig()

        self.client = ChatClient_OpenRouter(
            api_key=OPENROUTER_API_KEY,
            model=model_name,
            system_prompt=SYSTEM_PROMPT,
        )

        self.search_tools = SearchTools(output_dir, dataset_name=dataset_name)

        self.images: Dict[str, str] = {}
        self.branch_count = 0
        self.trajectory: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────────
    # Image management
    # ─────────────────────────────────────────────────────────────────────

    def _register_image(self, path: str) -> str:
        image_id = f"<image_{len(self.images)}>"
        self.images[image_id] = path
        return image_id

    def _resolve_image_path(self, image_id: str) -> Optional[str]:
        path = self.images.get(image_id)
        if path:
            return path
        cleaned = image_id.strip().strip("<>").strip()
        cleaned_lower = cleaned.lower()
        for prefix in ("image_", "image", "img_", "img", "image #", "image#"):
            if cleaned_lower.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        digits = re.search(r'(\d+)', cleaned)
        if digits:
            return self.images.get(f"<image_{digits.group(1)}>")

    # ─────────────────────────────────────────────────────────────────────
    # Init perception
    # ─────────────────────────────────────────────────────────────────────

    def _perceive_image(self, image_id: str, path: str) -> str:
        log(f"   Perceiving {image_id}: {os.path.basename(path)}")

        response = self.client.send_single_message(
            text=INIT_PERCEPTION_PROMPT,
            image_paths=[path],
            system_prompt="You are a visual analysis assistant.",
        )

        if not response or response.startswith("Error:"):
            log(f"   Perception failed: {response}")
            return f"{image_id}: [Query Image] (perception failed)"

        self.trajectory.append({
            "type": "init_perception",
            "image_id": image_id,
            "image_path": path,
            "response": response,
        })

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                summary = data.get("summary", "")
                source = data.get("source_hint", "")
                return (
                    f"{image_id}: [Query Image]\n"
                    f"  Content: {summary}\n"
                    f"  Source: {source}"
                )
        except (json.JSONDecodeError, AttributeError):
            pass

        return f"{image_id}: [Query Image] {response[:200]}"

    def initialize(self) -> str:
        log(f"Initializing MM-ContextFold Agent")
        log(f"   Question: {self.question[:100]}...")
        log(f"   Images: {len(self.image_paths)}")

        lines = []
        for path in self.image_paths:
            image_id = self._register_image(path)
            desc = self._perceive_image(image_id, path)
            lines.append(desc)

        return "\n\n".join(lines) if lines else "No images provided."

    # ─────────────────────────────────────────────────────────────────────
    # LLM call
    # ─────────────────────────────────────────────────────────────────────

    def _call_llm(self, text: str, image_paths: List[str] = None) -> Optional[str]:
        try:
            response = self.client.send_message(text=text, image_paths=image_paths)
            if response and response.startswith("Error:"):
                log(f"   LLM error: {response}")
                return None
            return response
        except Exception as e:
            log(f"   LLM exception: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Branch execution
    # ─────────────────────────────────────────────────────────────────────

    def _execute_branch(
        self,
        subtask: Dict,
        branch_name: str,
    ) -> Dict:
        saved_history = self.client.message_history.copy()

        task_prompt = subtask["prompt"]
        assigned_images = subtask.get("assigned_images", [])

        branch_image_paths = []
        loaded_image_ids = []
        for img_id in assigned_images:
            path = self._resolve_image_path(img_id)
            if path:
                branch_image_paths.append(path)
                loaded_image_ids.append(img_id)
            else:
                log(f"   Image not found: {img_id}")

        if loaded_image_ids:
            images_note = (
                "\n\n## Assigned Images\n"
                f"The following query images are loaded in this conversation: {', '.join(loaded_image_ids)}\n"
                "**You MUST use `lens_search` on the assigned images as your FIRST action** "
                "to identify visual content before doing any text search. "
                "This is critical — the image contains key information that text search alone cannot provide."
            )
        else:
            images_note = ""

        branch_instruction = BRANCH_PROMPT.replace("{task_prompt}", task_prompt + images_note)

        log(f"   Branch [{branch_name}] starting...")
        log(f"   Images assigned: {assigned_images}, loaded: {loaded_image_ids}")

        messages = [{"role": "user", "content": branch_instruction, "agent": branch_name}]
        return_message = "Branch completed without explicit return."

        response = self._call_llm(
            branch_instruction,
            image_paths=branch_image_paths if branch_image_paths else None,
        )

        consecutive_failures = 0
        for turn in range(self.config.max_branch_turns):
            if response is None:
                break

            messages.append({"role": "assistant", "content": response, "agent": branch_name})
            log(f"   Branch [{branch_name}] turn {turn + 1}: {response[:150]}...")

            tool_call = extract_tool_call(response)

            if tool_call is None:
                answer_in_branch = extract_answer(response)
                if not answer_in_branch and "<answer>" in response:
                    answer_in_branch = response.split("<answer>", 1)[1]
                    answer_in_branch = re.sub(r'</answer>.*', '', answer_in_branch, flags=re.DOTALL).strip()
                if answer_in_branch:
                    log(f"   Branch used <answer> instead of return, extracting as return")
                    return_message = answer_in_branch
                    break
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    log(f"   {consecutive_failures} consecutive parse failures, extracting content as return")
                    return_message = response[:1000] if response.strip() else "Branch failed to produce valid tool calls."
                    break
                error_msg = (
                    "No valid tool call found. Please wrap your tool call in the correct format:\n"
                    '<tool_call>\n{"name": "tool_name", "parameters": {...}}\n</tool_call>\n\n'
                    "Available tools: search, visit, lens_search, return.\n"
                    "Try again with the correct format."
                )
                messages.append({"role": "user", "content": error_msg, "agent": "error_msg"})
                response = self._call_llm(error_msg)
                continue
            consecutive_failures = 0

            tool_name = tool_call.get("name", "")
            params = tool_call.get("parameters", {})

            if tool_name in ("return", "finish"):
                msg = (params.get("message") or "").strip()
                if not msg:
                    error_msg = (
                        "Your `return` call has an empty message. "
                        "Please include your findings with key evidence and sources:\n"
                        '<tool_call>\n{"name": "return", "parameters": {"message": "your findings"}}\n</tool_call>'
                    )
                    messages.append({"role": "user", "content": error_msg, "agent": "error_msg"})
                    response = self._call_llm(error_msg)
                    continue
                if "<tool_call>" in msg:
                    error_msg = (
                        "Your `return` message should contain ONLY your research findings in plain text. "
                        "Do not include <tool_call> tags inside the return message. "
                        "If you still need to use a tool, call that tool instead of `return`. "
                        "If you are done, call `return` with a clean findings summary."
                    )
                    messages.append({"role": "user", "content": error_msg, "agent": "error_msg"})
                    response = self._call_llm(error_msg)
                    continue
                if "<answer>" in msg:
                    answer_match = re.search(r'<answer>(.*?)</answer>', msg, re.DOTALL)
                    if answer_match:
                        msg = answer_match.group(1).strip()
                    else:
                        msg = re.sub(r'</?answer>', '', msg).strip()
                return_message = msg
                break

            remaining_turns = self.config.max_branch_turns - 1 - turn
            deadline_warning = ""
            if remaining_turns <= 1:
                deadline_warning = (
                    "\n\n**You have reached your turn limit. "
                    "You MUST call `return` NOW with your findings so far. "
                    "Do NOT call any other tool.**"
                )
            elif remaining_turns == 2:
                deadline_warning = (
                    "\n\n**You have 2 turns remaining. "
                    "Wrap up your research and call `return` soon.**"
                )

            if tool_name == "lens_search":
                observation, result_image_paths = self._execute_lens_search(
                    params, branch_image_paths
                )
                log(f"   Tool [lens_search]: {observation[:150]}...")
                full_observation = observation + deadline_warning
                messages.append({"role": "user", "content": full_observation, "agent": "lens_search"})
                response = self._call_llm(
                    full_observation,
                    image_paths=result_image_paths if result_image_paths else None,
                )
            else:
                observation = self._execute_tool(tool_name, params, branch_image_paths)
                log(f"   Tool [{tool_name}]: {observation[:150]}...")
                full_observation = observation + deadline_warning
                messages.append({"role": "user", "content": full_observation, "agent": tool_name})
                response = self._call_llm(full_observation)
        else:
            if response:
                tool_call_in_response = extract_tool_call(response)
                if tool_call_in_response and tool_call_in_response.get("name") in ("return", "finish"):
                    msg = (tool_call_in_response.get("parameters", {}).get("message") or "").strip()
                    if msg:
                        return_message = msg
                    else:
                        return_message = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL).strip()
                else:
                    return_message = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL).strip()
                if not return_message:
                    return_message = response[:1000]

        self.client.message_history = saved_history

        self.trajectory.append({
            "type": "branch",
            "branch_name": branch_name,
            "subtask": subtask,
            "return_message": return_message,
            "messages": messages,
        })

        return {
            "return_message": return_message,
            "messages": messages,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Lens search (one-step, <result#N> naming)
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_lens_path(self, image_id: str, branch_image_paths: List[str]) -> Optional[str]:
        path = self._resolve_image_path(image_id)
        if not path:
            match = re.search(r'(\d+)', image_id)
            if match and branch_image_paths:
                idx = int(match.group(1))
                if 0 <= idx < len(branch_image_paths):
                    path = branch_image_paths[idx]
        return path

    def _execute_lens_search(
        self,
        params: Dict,
        branch_image_paths: List[str],
    ) -> tuple:
        image_id = params.get("image_id", "")
        path = self._resolve_lens_path(image_id, branch_image_paths)

        if not path:
            available = list(self.images.keys())
            error = (
                f"Invalid image_id `{image_id}`. "
                f"Available images: {available}. "
                "Please use the correct image_id and try `lens_search` again."
            )
            return error, []

        try:
            result_items, result_image_paths = self.search_tools.lens_search_with_images(path)
        except Exception as e:
            return f"lens_search error: {e}", []

        if not result_items:
            return "lens_search returned no results. Try a different approach such as text search.", []

        formatted = []
        for i, item in enumerate(result_items, 1):
            title = item.get("title", "Result")
            link = item.get("link", "")
            if i <= len(result_image_paths):
                formatted.append(f"<result#{i}>: **{title}**\n   URL: {link}")
            else:
                formatted.append(f"{i}. **{title}**\n   URL: {link}")

        observation = (
            f"Reverse image search results for {image_id}:\n\n"
            + "\n\n".join(formatted)
            + f"\n\nThe result images are attached in the same order (<result#1> ~ <result#{len(result_image_paths)}>). "
            f"Compare each result image with the query image {image_id} and "
            "use both visual and text information to continue your task."
        )

        return observation, result_image_paths

    def _execute_tool(
        self,
        tool_name: str,
        params: Dict,
        branch_image_paths: List[str],
    ) -> str:
        try:
            if tool_name == "search":
                query = params.get("query", "")
                return self.search_tools.text_search(query)

            elif tool_name in ("visit", "open_page"):
                url = params.get("url", "")
                return self.search_tools.visit_page(url)

            elif tool_name == "lens_search":
                image_id = params.get("image_id", "")
                path = self._resolve_lens_path(image_id, branch_image_paths)
                if path:
                    return self.search_tools.lens_search(path)
                available = list(self.images.keys())
                return (
                    f"Invalid image_id `{image_id}`. "
                    f"Available images: {available}. "
                    "Please use the correct image_id and try `lens_search` again."
                )

            return f"Unknown tool: {tool_name}"

        except Exception as e:
            return f"Tool error: {str(e)}"

    # ─────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        log(f"{'=' * 60}")
        log(f"Starting MM-ContextFold Pipeline")
        log(f"{'=' * 60}")

        try:
            image_context = self.initialize()

            user_prompt = (
                MAIN_USER_PROMPT
                .replace("{image_context}", image_context)
                .replace("{question}", self.question)
            )

            self.trajectory.append({
                "type": "main_init",
                "content": user_prompt,
            })

            response = self._call_llm(user_prompt)
            iteration = 0

            while iteration < self.config.max_rounds:
                iteration += 1

                if response is None:
                    log("   LLM returned None, retrying...")
                    response = self._call_llm("Please continue with your analysis.")
                    continue

                log(f"\n{'─' * 50}")
                log(f"MAIN Round {iteration}/{self.config.max_rounds}")
                log(f"   Response: {response[:200]}...")

                self.trajectory.append({
                    "type": "main_reasoning",
                    "round": iteration,
                    "content": response,
                })

                answer = extract_answer(response)
                if answer:
                    log(f"\nGot answer: {answer[:100]}...")
                    return self._build_result(answer, iteration)

                subtask = extract_subtask(response)

                if subtask is None:
                    error_msg = (
                        "Could not parse your response. You must use "
                        "exactly one of:\n"
                        "- `<subtask>{JSON}</subtask>` to create a research subtask\n"
                        "- `<answer>your answer</answer>` to provide the final answer\n\n"
                        "Try again."
                    )
                    response = self._call_llm(error_msg)
                    continue

                if self.branch_count >= self.config.max_branches:
                    log(f"   Branch limit reached ({self.config.max_branches})")
                    response = self._call_llm(
                        "Branch limit reached. Provide your best answer now.\n"
                        "<answer>your answer</answer>"
                    )
                    continue

                description = subtask.get("description", "subtask")
                branch_name = f"#{self.branch_count}-{description.replace(' ', '_')}"
                self.branch_count += 1

                log(f"   Subtask: {description}")
                log(f"   Prompt: {subtask['prompt'][:100]}...")
                log(f"   Images: {subtask.get('assigned_images', [])}")

                branch_result = self._execute_branch(subtask, branch_name)

                continue_prompt = (
                    MAIN_CONTINUE
                    .replace("{description}", description)
                    .replace("{return_message}", branch_result["return_message"])
                )

                response = self._call_llm(continue_prompt)

            log(f"\nMax rounds reached, forcing answer...")
            force_resp = self._call_llm(MAIN_FORCE_ANSWER)

            if force_resp:
                answer = extract_answer(force_resp)
                if answer:
                    return self._build_result(answer, iteration)
                return self._build_result(force_resp[:500], iteration)

            return self._build_result("No answer (max rounds exceeded)", iteration)

        except Exception as e:
            log(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error: {e}",
                "error": str(e),
                "trajectory": self.trajectory,
            }

    def _build_result(self, answer: str, rounds: int) -> Dict:
        return {
            "answer": answer,
            "total_rounds": rounds,
            "total_branches": self.branch_count,
            "trajectory": self.trajectory,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline registration
# ═══════════════════════════════════════════════════════════════════════════

@register_pipeline("mmcontextfold")
class MMContextFoldPipeline(BasePipeline):

    method_name = "mmcontextfold"

    def __init__(self, config: MMContextFoldConfig):
        super().__init__(config)
        self.fold_config = config

    def process_single(self, item: Dict, index: int) -> Dict:
        sample_id = item.get("id", f"sample_{index}")

        save_dir = self.config.output_dir
        img_save_dir = os.path.join(save_dir, "imgs", sample_id)
        log_dir = os.path.join(save_dir, "logs")
        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        logger = SampleLogger(log_dir, sample_id, self.method_name)
        set_logger(logger)

        log(f"[{index}] Processing (MM-ContextFold): {sample_id}")

        question = self._prepare_question(item)
        image_paths = self._prepare_images(item)

        for p in image_paths:
            log(f"   Image: {p}")

        agent = MMContextFoldAgent(
            question=question,
            image_paths=image_paths,
            dataset_name=self.config.dataset,
            output_dir=save_dir,
            sample_id=sample_id,
            model_name=self.config.model_full_name,
            config=self.fold_config,
        )

        try:
            result = agent.run()
            logger.close("success")
            log(f"[{index}] Done: {sample_id}")
            return {
                "id": sample_id,
                "question": question,
                "answer": result.get("answer", ""),
                "gt_answer": item.get("answer", ""),
                "status": "success",
                "num_rounds": result.get("total_rounds", 0),
                "num_branches": result.get("total_branches", 0),
                "messages": result.get("trajectory", []),
            }
        except Exception as e:
            log(f"Error: {e}")
            import traceback
            traceback.print_exc()
            logger.close("error")
            return {
                "id": sample_id,
                "status": "error",
                "error": str(e),
            }
