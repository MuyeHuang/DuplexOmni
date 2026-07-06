from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


DEFAULT_MODEL = "gpt-4.1"
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_JSONL = (
    BASE_DIR
    / "huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/data/train_sft.jsonl"
)
DEFAULT_OUTPUT_JSONL = BASE_DIR / "api_slot/content/train_sft.voiceagent.jsonl"
_THREAD_LOCAL = threading.local()
DEFAULT_RANDOM_SEED = 20260318

SYSTEM_PROMPT = """你是一个中文语音对话数据改写助手。你的任务是把给定的 UltraChat 文本对话，改写成适合 VoiceAgent 训练的中文对话数据。

你必须严格遵守：

1. 保留原始对话的核心任务、主要事实、问题推进路径和解决思路，不要偏题，不要改变原任务的本质。
2. 把 assistant 视为 Agent，把 user 视为用户。
3. 所有 user 和 assistant 的文本都必须改写成适合直接念出来的中文口语表达，听起来像真实人会在语音里说的话。
4. 即使原始内容偏书面、偏列表、偏说明文、偏教程，也要强行改造成自然口语对话，但仍然保留原意。
5. 不要为了“口语化”刻意加入“嗯、啊、呃、哈、哈哈、额”之类语气词。
6. User 问题正常简短，只有原任务确实需要补充背景、条件或限制时才稍长。
7. Agent 回复正常简短，只有原任务确实需要解释、推理、分步骤说明或澄清时才稍长。
8. 可以压缩、改写、重组原句表达，但不要遗漏关键事实、关键条件、关键结论。
9. 不要输出 Markdown、代码块、项目符号、括号动作说明、舞台指令、标题或解释文字。
10. 输出必须是合法 JSON，并且只能输出 JSON 本体。
11. 如果原文表达过于密集，可以在不改变原意的前提下，把信息拆成更适合语音交互的自然轮次推进方式。
"""

_THINK_TAG_RE = re.compile(r"<think>\s*.*?\s*</think>", re.DOTALL)


def strip_thinking_tags(text: str) -> str:
    """
    Qwen3.5 may generate thinking content wrapped by <think>...</think>.
    We remove it before JSON extraction/validation so the pipeline won't fail.
    """
    if not text:
        return ""
    return re.sub(_THINK_TAG_RE, "", text).strip()


class RateLimiter:
    def __init__(self, rpm_limit: int) -> None:
        self.interval = 60.0 / rpm_limit
        self.last_call_time = 0.0
        self.lock = threading.Lock()

    def wait_for_token(self) -> None:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call_time
            wait_time = self.interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_call_time = time.time()


def iter_jsonl(
    path: Path,
    *,
    limit: int | None = None,
    skip: int = 0,
) -> Iterator[Dict[str, Any]]:
    yielded = 0
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if line_index < skip:
                continue
            if limit is not None and yielded >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yielded += 1
            yield json.loads(line)


def count_jsonl_records(path: Path, *, skip: int = 0, limit: int | None = None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if line_index < skip:
                continue
            if limit is not None and count >= limit:
                break
            if line.strip():
                count += 1
    return count


def build_call_type_plan(total: int, random_seed: int) -> List[str]:
    if total <= 0:
        return []
    outbound_count = total // 2
    inbound_count = total - outbound_count
    plan = ["outbound"] * outbound_count + ["inbound"] * inbound_count
    rng = random.Random(random_seed)
    rng.shuffle(plan)
    return plan


def build_output_paths(base_output_path: Path) -> Dict[str, Path]:
    stem = base_output_path.stem
    suffix = base_output_path.suffix or ".jsonl"
    parent = base_output_path.parent
    return {
        "outbound": parent / f"{stem}.outbound{suffix}",
        "inbound": parent / f"{stem}.inbound{suffix}",
    }


def build_dropped_paths(output_paths: Dict[str, Path]) -> Dict[str, Path]:
    return {
        call_type: path.with_name(f"{path.stem}.dropped{path.suffix}")
        for call_type, path in output_paths.items()
    }


def build_user_prompt(
    *,
    sample: Dict[str, Any],
    desired_call_type: str,
) -> str:
    call_type_label = "Agent主动外呼" if desired_call_type == "outbound" else "Agent被动接听"
    system_requirement = (
        "第一条消息必须是 system，内容格式必须是“你是xxx，任务是xxx”这种形式。"
        if desired_call_type == "outbound"
        else "第一条消息必须是 system，内容固定为“你是有用的助手”。"
    )
    first_speaker_requirement = (
        "system 后第一句必须由 assistant 说。"
        if desired_call_type == "outbound"
        else "system 后第一句必须由 user 说。"
    )
    sample_text = json.dumps(sample, ensure_ascii=False, indent=2)

    return f"""请把下面这条 UltraChat 样本改写成 VoiceAgent 训练数据。

[硬性要求]
1. 本条样本必须改造成：{call_type_label}。
2. {system_requirement}
3. {first_speaker_requirement}
4. 保留原对话的核心任务、关键事实、解决路径和大体信息密度，但可以重写表达方式。
5. 把原本偏书面、偏教程、偏网页问答的表达，改成更像真实语音交互的说法。
6. 所有 user 和 assistant 文本都必须是“能直接念出来”的自然口语。
7. 不要加入“嗯、啊、呃、哈哈”等语气词，也不要加入括号动作、旁白、笑声标签、舞台说明。
8. Agent 回复不要太长，尽量 1 到 3 句；用户回复也保持自然，不要过度扩写。
9. role 只能是 system、user、assistant。
10. 输出必须是合法 JSON，且只能输出 JSON。

[输出 JSON 格式]
{{
  "call_type": "{desired_call_type}",
  "task": "根据改写后的对话总结出的任务",
  "messages": [
    {{"role": "system", "content": "..." }},
    {{"role": "assistant", "content": "..." }}
  ]
}}

[额外说明]
- 如果原始内容是教程、说明文、知识问答，请改成电话里也能自然表达的说法。
- 如果原始内容过于书面，可以压缩成更短更口语的表述，但不要漏掉关键结论。
- 对于外呼场景，Agent 需要主动开启话题和任务推进。
- 对于接听场景，用户先表达诉求，Agent 再承接解决。
- messages 中必须同时包含 user 和 assistant。

[原始 UltraChat 样本]
{sample_text}
"""


def extract_json_block(text: str) -> Dict[str, Any] | None:
    stripped = strip_thinking_tags(text).strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def make_client(api_base: str, api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - import guard for runtime env
        raise SystemExit(
            "Missing dependency: openai. Please install it in the current environment first."
        ) from exc
    return OpenAI(base_url=api_base, api_key=api_key)


def get_thread_client(api_base: str, api_key: str) -> Any:
    client = getattr(_THREAD_LOCAL, "client", None)
    client_key = getattr(_THREAD_LOCAL, "client_key", None)
    if client is None or client_key != (api_base, api_key):
        client = make_client(api_base, api_key)
        _THREAD_LOCAL.client = client
        _THREAD_LOCAL.client_key = (api_base, api_key)
    return client


def call_model(
    client: Any,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, int]]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False}
        },
    )
    usage = getattr(response, "usage", None)
    token_stats = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    return response.choices[0].message.content or "", token_stats


def debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[debug] {message}", file=sys.stderr, flush=True)


def make_prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def validate_messages(messages: Any, desired_call_type: str) -> List[Dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    normalized: List[Dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"message[{index}] has invalid role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message[{index}] has empty content")
        normalized.append({"role": role, "content": content.strip()})

    if normalized[0]["role"] != "system":
        raise ValueError("first message must be system")
    if len(normalized) < 3:
        raise ValueError("messages must contain at least system + two dialogue turns")
    if desired_call_type == "outbound" and normalized[1]["role"] != "assistant":
        raise ValueError("outbound sample must start with assistant after system")
    if desired_call_type == "inbound" and normalized[1]["role"] != "user":
        raise ValueError("inbound sample must start with user after system")

    dialogue_roles = {message["role"] for message in normalized[1:]}
    if "user" not in dialogue_roles or "assistant" not in dialogue_roles:
        raise ValueError("dialogue must contain both user and assistant")
    return normalized


def validate_transformed_output(
    payload: Dict[str, Any],
    *,
    desired_call_type: str,
) -> Tuple[str, List[Dict[str, str]]]:
    if not isinstance(payload, dict):
        raise ValueError("model output must be a JSON object")

    call_type = payload.get("call_type")
    if call_type != desired_call_type:
        raise ValueError(
            f"call_type mismatch: expected {desired_call_type!r}, got {call_type!r}"
        )

    task = payload.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")

    messages = validate_messages(payload.get("messages"), desired_call_type)
    if desired_call_type == "outbound":
        system_content = messages[0]["content"]
        if not system_content.startswith("你是") or "任务是" not in system_content:
            raise ValueError("outbound system prompt must contain '你是xxx，任务是xxx'")
    else:
        system_content = messages[0]["content"].strip()
        # Some models may add minor whitespace/formatting; we only require the prefix.
        if not system_content.startswith("你是有用的助手"):
            raise ValueError("inbound system prompt must start with '你是有用的助手'")

    return task.strip(), messages


def build_output_record(
    sample: Dict[str, Any],
    transformed_payload: Dict[str, Any],
    *,
    desired_call_type: str,
) -> Dict[str, Any]:
    _, messages = validate_transformed_output(
        transformed_payload,
        desired_call_type=desired_call_type,
    )
    prompt = next((message["content"] for message in messages[1:] if message["role"] != "system"), "")
    if not prompt:
        raise ValueError("unable to derive prompt from transformed messages")

    output_record = dict(sample)
    output_record["messages"] = messages
    output_record["prompt"] = prompt
    output_record["prompt_id"] = make_prompt_id(prompt)
    return output_record


def process_one_sample(
    *,
    sample: Dict[str, Any],
    sample_index: int,
    desired_call_type: str,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_sleep_seconds: float,
    rate_limiter: RateLimiter | None,
    debug: bool,
) -> Dict[str, Any]:
    sample_id = str(sample.get("prompt_id") or f"sample_{sample_index + 1}")
    prompt = build_user_prompt(
        sample=sample,
        desired_call_type=desired_call_type,
    )

    model_text = ""
    error_message = None
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    attempt_count = 0

    for attempt in range(1, max_retries + 1):
        attempt_count = attempt
        try:
            client = get_thread_client(api_base, api_key)
            if rate_limiter is not None:
                debug_log(debug, f"{sample_id} attempt={attempt} waiting for rate limiter")
                rate_limiter.wait_for_token()
            debug_log(debug, f"{sample_id} attempt={attempt} sending request")
            model_text, latest_token_stats = call_model(
                client=client,
                model=model,
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            token_stats["prompt_tokens"] += latest_token_stats["prompt_tokens"]
            token_stats["completion_tokens"] += latest_token_stats["completion_tokens"]
            token_stats["total_tokens"] += latest_token_stats["total_tokens"]

            model_text_stripped = strip_thinking_tags(model_text)
            payload = extract_json_block(model_text_stripped)
            if payload is None:
                raise ValueError("model output does not contain valid JSON")

            output_record = build_output_record(
                sample,
                payload,
                desired_call_type=desired_call_type,
            )
            return {
                "ok": True,
                "output_record": output_record,
                "sample_id": sample_id,
                "call_type": desired_call_type,
                "input_record": sample,
                "attempt_count": attempt_count,
                "model_output": model_text,
                "model_output_stripped": model_text_stripped,
                **token_stats,
            }
        except Exception as exc:  # pragma: no cover - depends on remote server
            error_message = f"{type(exc).__name__}: {exc}"
            debug_log(debug, f"{sample_id} attempt={attempt} error={error_message}")
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)

    return {
        "ok": False,
        "output_record": None,
        "sample_id": sample_id,
        "call_type": desired_call_type,
        "input_record": sample,
        "attempt_count": attempt_count,
        "model_output": model_text,
        "model_output_stripped": strip_thinking_tags(model_text),
        "error": error_message,
        **token_stats,
    }


def render_progress(
    completed: int,
    total: int,
    *,
    success: int,
    errors: int,
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
    avg_total_tokens: float,
) -> None:
    total = max(total, 1)
    width = 30
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100.0 * completed / total
    line = (
        f"\r[{bar}] {completed}/{total} {percent:5.1f}% "
        f"success={success} error={errors} "
        f"avg_in={avg_prompt_tokens:.1f} avg_out={avg_completion_tokens:.1f} "
        f"avg_total={avg_total_tokens:.1f}"
    )
    print(line, end="", file=sys.stderr, flush=True)
    if completed >= total:
        print(file=sys.stderr, flush=True)


def process_completed_results(
    *,
    completed_results: Dict[int, Dict[str, Any]],
    out_files: Dict[str, Any],
    dropped_files: Dict[str, Any],
    next_to_write: int,
) -> Tuple[int, int]:
    dropped_count = 0
    while next_to_write in completed_results:
        result = completed_results.pop(next_to_write)
        call_type = result["call_type"]
        if result["ok"]:
            out_files[call_type].write(
                json.dumps(result["output_record"], ensure_ascii=False) + "\n"
            )
        else:
            dropped_record = {
                "sample_id": result["sample_id"],
                "call_type": call_type,
                "attempt_count": result["attempt_count"],
                "error": result.get("error"),
                "model_output": result.get("model_output"),
                "model_output_stripped": result.get("model_output_stripped"),
                "input": result.get("input_record"),
            }
            dropped_files[call_type].write(
                json.dumps(dropped_record, ensure_ascii=False) + "\n"
            )
            dropped_count += 1
        next_to_write += 1
    return next_to_write, dropped_count


def process_samples(
    *,
    input_path: Path,
    output_path: Path,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    limit: int | None,
    max_retries: int,
    retry_sleep_seconds: float,
    concurrency: int,
    rpm_limit: int,
    random_seed: int,
    resume_from_output: bool,
    debug: bool,
) -> None:
    output_paths = build_output_paths(output_path)
    dropped_paths = build_dropped_paths(output_paths)
    resumed_success_count = sum(
        count_jsonl_records(path)
        for path in output_paths.values()
        if resume_from_output and path.exists()
    )
    resumed_dropped_count = sum(
        count_jsonl_records(path)
        for path in dropped_paths.values()
        if resume_from_output and path.exists()
    )
    resumed_count = resumed_success_count + resumed_dropped_count
    total = count_jsonl_records(input_path, skip=resumed_count, limit=limit)
    call_type_plan = build_call_type_plan(resumed_count + total, random_seed)
    rate_limiter = RateLimiter(rpm_limit) if rpm_limit > 0 else None
    output_modes = {
        call_type: ("a" if resume_from_output and path.exists() else "w")
        for call_type, path in output_paths.items()
    }
    dropped_modes = {
        call_type: ("a" if resume_from_output and path.exists() else "w")
        for call_type, path in dropped_paths.items()
    }

    processed = 0
    success_count = 0
    error_count = 0
    dropped_count = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_paths["outbound"].open(output_modes["outbound"], encoding="utf-8") as outbound_f:
        with output_paths["inbound"].open(output_modes["inbound"], encoding="utf-8") as inbound_f:
            with dropped_paths["outbound"].open(
                dropped_modes["outbound"], encoding="utf-8"
            ) as outbound_dropped_f:
                with dropped_paths["inbound"].open(
                    dropped_modes["inbound"], encoding="utf-8"
                ) as inbound_dropped_f:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                        future_to_index: Dict[concurrent.futures.Future[Dict[str, Any]], int] = {}
                        completed_results: Dict[int, Dict[str, Any]] = {}
                        next_to_write = 0
                        out_files = {
                            "outbound": outbound_f,
                            "inbound": inbound_f,
                        }
                        dropped_files = {
                            "outbound": outbound_dropped_f,
                            "inbound": inbound_dropped_f,
                        }

                        for sample_index, sample in enumerate(
                            iter_jsonl(input_path, skip=resumed_count, limit=limit)
                        ):
                            global_index = resumed_count + sample_index
                            desired_call_type = call_type_plan[global_index]
                            future = executor.submit(
                                process_one_sample,
                                sample=sample,
                                sample_index=sample_index,
                                desired_call_type=desired_call_type,
                                model=model,
                                api_base=api_base,
                                api_key=api_key,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                max_retries=max_retries,
                                retry_sleep_seconds=retry_sleep_seconds,
                                rate_limiter=rate_limiter,
                                debug=debug,
                            )
                            future_to_index[future] = sample_index

                            if len(future_to_index) < concurrency:
                                continue

                            done, _ = concurrent.futures.wait(
                                future_to_index,
                                return_when=concurrent.futures.FIRST_COMPLETED,
                            )
                            for future in done:
                                index = future_to_index.pop(future)
                                result = future.result()
                                completed_results[index] = result
                                processed += 1
                                if result["ok"]:
                                    success_count += 1
                                else:
                                    error_count += 1
                                prompt_tokens_total += int(result.get("prompt_tokens", 0))
                                completion_tokens_total += int(result.get("completion_tokens", 0))
                                total_tokens_total += int(result.get("total_tokens", 0))
                                avg_divisor = max(success_count + error_count, 1)
                                render_progress(
                                    processed,
                                    total,
                                    success=success_count,
                                    errors=error_count,
                                    avg_prompt_tokens=prompt_tokens_total / avg_divisor,
                                    avg_completion_tokens=completion_tokens_total / avg_divisor,
                                    avg_total_tokens=total_tokens_total / avg_divisor,
                                )
                            next_to_write, dropped_delta = process_completed_results(
                                completed_results=completed_results,
                                out_files=out_files,
                                dropped_files=dropped_files,
                                next_to_write=next_to_write,
                            )
                            dropped_count += dropped_delta

                        for future in concurrent.futures.as_completed(future_to_index):
                            index = future_to_index[future]
                            result = future.result()
                            completed_results[index] = result
                            processed += 1
                            if result["ok"]:
                                success_count += 1
                            else:
                                error_count += 1
                            prompt_tokens_total += int(result.get("prompt_tokens", 0))
                            completion_tokens_total += int(result.get("completion_tokens", 0))
                            total_tokens_total += int(result.get("total_tokens", 0))
                            avg_divisor = max(success_count + error_count, 1)
                            render_progress(
                                processed,
                                total,
                                success=success_count,
                                errors=error_count,
                                avg_prompt_tokens=prompt_tokens_total / avg_divisor,
                                avg_completion_tokens=completion_tokens_total / avg_divisor,
                                avg_total_tokens=total_tokens_total / avg_divisor,
                            )
                            next_to_write, dropped_delta = process_completed_results(
                                completed_results=completed_results,
                                out_files=out_files,
                                dropped_files=dropped_files,
                                next_to_write=next_to_write,
                            )
                            dropped_count += dropped_delta

    print("done")
    print(f"input_path: {input_path}")
    print(f"output_outbound_path: {output_paths['outbound']}")
    print(f"output_inbound_path: {output_paths['inbound']}")
    print(f"dropped_outbound_path: {dropped_paths['outbound']}")
    print(f"dropped_inbound_path: {dropped_paths['inbound']}")
    print(f"resume_from_output: {resume_from_output}")
    print(f"resumed_count: {resumed_count}")
    print(f"resumed_success_count: {resumed_success_count}")
    print(f"resumed_dropped_count: {resumed_dropped_count}")
    print(f"random_seed: {random_seed}")
    print(f"processed: {processed}")
    print(f"success_count: {success_count}")
    print(f"error_count: {error_count}")
    print(f"dropped_count: {dropped_count}")
    print(f"concurrency: {concurrency}")
    print(f"rpm_limit: {rpm_limit}")
    if processed:
        print(f"avg_prompt_tokens: {prompt_tokens_total / processed:.2f}")
        print(f"avg_completion_tokens: {completion_tokens_total / processed:.2f}")
        print(f"avg_total_tokens: {total_tokens_total / processed:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite UltraChat JSONL into VoiceAgent-style spoken dialogues."
    )
    parser.add_argument(
        "--input-jsonl",
        default=str(DEFAULT_INPUT_JSONL),
        help="Input UltraChat JSONL file.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Output JSONL file in the original UltraChat record format.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name exposed by the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for rewriting.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max tokens returned by the model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on how many input records to process.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="How many times to retry a failed rewrite.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=2.0,
        help="Sleep duration between retries.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="How many requests to send concurrently while preserving output order.",
    )
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=10,
        help="Global request-per-minute limit. Use 0 to disable throttling.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Deterministic random seed used to split samples into outbound/inbound halves.",
    )
    parser.add_argument(
        "--resume-from-output",
        action="store_true",
        help="Resume from an existing output JSONL by skipping the same number of input lines.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-sample debug logs to stderr.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_samples(
        input_path=Path(args.input_jsonl),
        output_path=Path(args.output_jsonl),
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        concurrency=args.concurrency,
        rpm_limit=args.rpm_limit,
        random_seed=args.random_seed,
        resume_from_output=args.resume_from_output,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
