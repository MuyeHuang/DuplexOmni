"""
api_client.py
功能: 封装对 OpenAI-compatible API 的调用逻辑。
"""
import json
import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI


class APICaller:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=99999,
        )
        self.model = model

    def _is_gemini_model(self) -> bool:
        return "gemini" in self.model.lower()

    def _is_qwen_model(self) -> bool:
        model_lower = self.model.lower()
        return "qwen" in model_lower or "qwq" in model_lower

    def _is_deepseek_model(self) -> bool:
        return "deepseek" in self.model.lower()

    def _build_request_kwargs(self, messages: List[Dict]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 8192,
        }

        if self._is_gemini_model():
            kwargs["reasoning_effort"] = "low"

        if self._is_qwen_model():
            kwargs["temperature"] = 0.6
            kwargs["max_tokens"] = 25000
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

        if self._is_deepseek_model():
            kwargs["temperature"] = 0.6
            kwargs["max_tokens"] = 25000
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "max",
                }
            }

        return kwargs

    def _strip_think_tags(self, text: str) -> str:
        cleaned = re.sub(
            r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _extract_usage_detail(usage) -> Dict[str, int]:
        cached_tokens = 0
        reasoning_tokens = 0
        cache_write_tokens = 0

        ptd = getattr(usage, "prompt_tokens_details", None)
        if ptd is not None:
            cached_tokens = getattr(ptd, "cached_tokens", None) or 0

        if cached_tokens == 0:
            extra = getattr(usage, "model_extra", None) or {}
            cached_tokens = (
                extra.get("cache_read_tokens")
                or extra.get("cached_tokens")
                or extra.get("effectiveCachedTokens")
                or 0
            )

        ctd = getattr(usage, "completion_tokens_details", None)
        if ctd is not None:
            reasoning_tokens = getattr(ctd, "reasoning_tokens", None) or 0

        extra = getattr(usage, "model_extra", None) or {}
        cache_write_tokens = extra.get("cache_write_tokens", 0) or 0

        return {
            "cached_tokens": int(cached_tokens),
            "reasoning_tokens": int(reasoning_tokens),
            "cache_write_tokens": int(cache_write_tokens),
        }

    def call(self, messages: List[Dict]) -> Tuple[str, Any]:
        try:
            response = self.client.chat.completions.create(
                **self._build_request_kwargs(messages)
            )

            if not response or not response.choices:
                print("[API错误] 无有效响应")
                return "", []

            content = response.choices[0].message.content
            usage = response.usage
            completion_tokens = usage.completion_tokens
            prompt_tokens = usage.prompt_tokens
            total_tokens = usage.total_tokens

            if not content:
                return "", []

            raw_response = self._strip_think_tags(content)
            detail = self._extract_usage_detail(usage)

            return raw_response, (completion_tokens, prompt_tokens, total_tokens, detail)

        except Exception as e:
            print(f"[API系统错误] {type(e).__name__}: {str(e)}")
            return "", []
