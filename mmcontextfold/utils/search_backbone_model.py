"""
search_backbone_model.py

LLM client wrapper supporting multi-turn conversations with MAIN/BRANCH history management.
"""

import os
import base64
from pathlib import Path
import mimetypes
from typing import List, Union, Dict, Any, Optional
from openai import OpenAI


class ChatClient_OpenRouter:
    def __init__(self, api_key: str, model: str, system_prompt: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        self.model = model
        self.system_prompt = system_prompt

        self.main_history = [
            {"role": "system", "content": system_prompt}
        ]
        self.branch_history = []

        self.all_messages_history = [
            {"role": "system", "content": system_prompt, "source": "init"}
        ]

        self.message_history = self.main_history

        self.token_usage = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "per_call": [],
        }

    def _record_to_all_history(self, role: str, content, source: str):
        self.all_messages_history.append({
            "role": role,
            "content": content,
            "source": source
        })

    def _track_usage(self, response, source: str = "unknown"):
        usage = getattr(response, 'usage', None)
        prompt_tokens = 0
        completion_tokens = 0
        if usage is not None:
            prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
            completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
        total = prompt_tokens + completion_tokens
        self.token_usage["total_prompt_tokens"] += prompt_tokens
        self.token_usage["total_completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] += total
        self.token_usage["call_count"] += 1
        self.token_usage["per_call"].append({
            "source": source,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
        })

    def get_token_usage(self) -> Dict:
        return self.token_usage.copy()

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_image_mime_type(self, image_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type and mime_type.startswith('image/'):
            return mime_type
        return "image/jpeg"

    def create_message_content(self, text: str = None, image_paths: List[str] = None) -> List[Dict[str, Any]]:
        content = []

        if text and text.strip():
            content.append({"type": "text", "text": text})

        if image_paths:
            for image_path in image_paths:
                try:
                    if not Path(image_path).exists():
                        raise Exception(f"Image file not found: {image_path}")
                    base64_image = self.encode_image(image_path)
                    mime_type = self.get_image_mime_type(image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high"
                        }
                    })
                except Exception as e:
                    print(f"Warning: Could not process image {image_path}: {str(e)}")

        return content if content else [{"type": "text", "text": ""}]

    def send_message(self, text: str = None, image_paths: List[str] = None):
        if not text and not image_paths:
            return "Error: Must provide either text or images"

        content = self.create_message_content(text, image_paths)
        self.message_history.append({"role": "user", "content": content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.message_history,
                max_completion_tokens=8192,
                temperature=0.0,
                top_p=1.0
            )
            self._track_usage(response, source="send_message")
            assistant_message = response.choices[0].message.content
            self.message_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        except Exception as e:
            self.message_history.pop()
            return f"Error: {str(e)}"

    def send_single_message(self, text: str = None, image_paths: List[str] = None,
                            system_prompt: str = None):
        if not text and not image_paths:
            return "Error: Must provide either text or images"

        if system_prompt is None:
            system_prompt = self.message_history[0]["content"]

        messages = [{"role": "system", "content": system_prompt}]
        content = self.create_message_content(text, image_paths)
        messages.append({"role": "user", "content": content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=8192,
                temperature=0.0,
                top_p=1.0
            )
            self._track_usage(response, source="send_single_message")
            return response.choices[0].message.content

        except Exception as e:
            return f"Error: {str(e)}"

    def clear_conversation(self):
        self.message_history = [self.message_history[0]]

    def get_message_history(self):
        return self.message_history

    def get_conversation_count(self):
        return len(self.message_history) - 1
