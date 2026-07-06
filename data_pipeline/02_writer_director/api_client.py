"""
api_client.py
功能: 封装对API的调用逻辑 (基于OpenAI SDK)
"""
import json
import re
from openai import OpenAI
from typing import Dict, List, Tuple, Any


class APICaller:
    def __init__(self, api_key: str, base_url: str, model: str):
        """
        初始化API调用器
        """
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
        """
        从 response.usage 提取完整用量细节，兼容：
          - deepseek-v4-pro（prompt_tokens_details 有值）
          - deepseek-v4-pro-baidu（prompt_tokens_details 为 None，cache 在 model_extra 顶层）
          - 其他模型（尽力而为，缺失字段置 0）

        返回 dict:
          cached_tokens    : prompt 中命中缓存的 tokens
          reasoning_tokens : completion 中推理/thinking tokens
          cache_write_tokens: 本次写入缓存的 tokens（部分模型返回）
        """
        cached_tokens = 0
        reasoning_tokens = 0
        cache_write_tokens = 0

        # ── 1. cached_tokens ──────────────────────────────────────────────────
        # 优先从标准 prompt_tokens_details 取（deepseek-v4-pro 等）
        ptd = getattr(usage, "prompt_tokens_details", None)
        if ptd is not None:
            cached_tokens = getattr(ptd, "cached_tokens", None) or 0

        # 兜底：从 model_extra 顶层取（deepseek-v4-pro-baidu / 旧版接口）
        if cached_tokens == 0:
            extra = getattr(usage, "model_extra", None) or {}
            # 优先 cache_read_tokens，其次 cached_tokens，最后 effectiveCachedTokens
            cached_tokens = (
                extra.get("cache_read_tokens")
                or extra.get("cached_tokens")
                or extra.get("effectiveCachedTokens")
                or 0
            )

        # ── 2. reasoning_tokens ───────────────────────────────────────────────
        ctd = getattr(usage, "completion_tokens_details", None)
        if ctd is not None:
            reasoning_tokens = getattr(ctd, "reasoning_tokens", None) or 0

        # ── 3. cache_write_tokens（部分接口返回）────────────────────────────
        extra = getattr(usage, "model_extra", None) or {}
        cache_write_tokens = extra.get("cache_write_tokens", 0) or 0

        return {
            "cached_tokens": int(cached_tokens),
            "reasoning_tokens": int(reasoning_tokens),
            "cache_write_tokens": int(cache_write_tokens),
        }

    def call(self, messages: List[Dict]) -> Tuple[str, Any]:
        """
        调用API
        :param messages: 消息列表
        :return: (原始响应文本, token_stats)
            token_stats = (completion_tokens, prompt_tokens, total_tokens, detail_dict)
            detail_dict 包含:
              cached_tokens     : prompt 命中缓存的 tokens
              reasoning_tokens  : completion 中推理 tokens
              cache_write_tokens: 写入缓存的 tokens
        """
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
