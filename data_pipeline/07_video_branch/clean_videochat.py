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
    / "api_generate/video_stream_output/video_tags/video_tags_nextqa_full.jsonl"
)
DEFAULT_OUTPUT_JSONL = BASE_DIR / "api_slot/content/train_sft.videochat.jsonl"
_THREAD_LOCAL = threading.local()
DEFAULT_RANDOM_SEED = 20260318

SYSTEM_PROMPT = """你是一个中文视频通话对话素材生成助手。你的任务是根据给定的视频事件标注，生成适合后续编剧改写的 QA 对话素材。

你必须严格遵守：

1. 对话内容必须严格基于视频事件标注，按事件时序推进，不能提及还未发生的事件。
2. user 的问题必须基于视频画面中可见的具体细节产生真实疑问：为什么这样做、这个东西是什么、这么做有什么效果。user 不复述画面，而是对看到的细节感到好奇或不解。
3. assistant 的回答必须锚定 user 问的那个具体细节，可以补充一点背景知识，但绝对不能把视频里还没出现、还没发生的内容直接说出来。回答要让用户觉得"需要继续看视频才能验证"，而不是把所有答案都交代清楚。
4. 不要生成"不看视频也能说"的通用知识问答。每一轮 QA 必须和视频画面的具体细节紧密绑定。
5. 所有文本都是能直接念出来的中文口语。
6. 对话严格交替：user 之后必须是 assistant，assistant 之后必须是 user，绝对不能连续相同 role。
7. 不要输出 Markdown、代码块、项目符号、括号动作说明、舞台指令或解释文字。
8. 输出必须是合法 JSON，并且只能输出 JSON 本体。
"""

_THINK_TAG_RE = re.compile(r"<think>\s*.*?\s*</think>", re.DOTALL)


def strip_thinking_tags(text: str) -> str:
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


def format_events(event_intervals: List[Dict[str, Any]]) -> str:
    lines = []
    for i, ev in enumerate(event_intervals, 1):
        start = ev.get("start_sec", 0)
        end = ev.get("end_sec", 0)
        duration = end - start
        char_budget = int(duration * 5)
        title = ev.get("event_title", "")
        summary = ev.get("event_summary", "")
        lines.append(
            f"E{i} [{start:.1f}s-{end:.1f}s 时长{duration:.1f}s 字数预算{char_budget}字]：{title}——{summary}"
        )
    return "\n".join(lines)


def format_event_timeline(event_intervals: List[Dict[str, Any]]) -> str:
    return format_events(event_intervals)


def build_user_prompt(
    *,
    sample: Dict[str, Any],
    desired_call_type: str,
) -> str:
    event_tags = sample.get("event_tags", {})
    event_intervals = event_tags.get("event_intervals", [])
    global_notes = event_tags.get("global_notes", "")
    duration_sec = sample.get("meta", {}).get("duration_sec", 0)
    video_id = sample.get("meta", {}).get("video_id", "unknown")

    events_text = format_events(event_intervals)
    event_count = len(event_intervals)

    if desired_call_type == "outbound":
        opening_requirement = (
            "这是 outbound 场景：用户一直没有主动开口，assistant 看了一段视频后主动发起对话，"
            "提出一个关于画面内容的开放性问题或评论，引导用户参与。"
            "system 后第一句必须是 assistant。"
        )
    else:
        opening_requirement = (
            "这是 inbound 场景：用户看到画面后主动发起对话，提出第一个疑问。"
            "system 后第一句必须是 user。"
        )

    return f"""请根据下面的视频事件标注，生成一段视频通话 QA 对话素材。

[视频信息]
- 视频ID：{video_id}
- 总时长：{duration_sec:.1f}秒
- 事件数量：{event_count}个
- 全局说明：{global_notes}

[事件序列]
{events_text}

[硬性要求]
1. 本条样本必须生成：{"外呼（outbound）" if desired_call_type == "outbound" else "内呼（inbound）"}对话。
2. {opening_requirement}
3. system 内容固定为"你是视频通话助手"。
4. 对话内容严格按事件时序推进，不能提及还未发生的事件。
5. user 的问题必须基于视频画面中可见的具体细节，不要让 user 复述画面。
6. assistant 的回答锚定 user 问的细节，可以补充一点背景，但不能把视频里还没出现的内容说出来，更不能生成"不看视频也能说"的通用知识。每个事件对应 1-2 轮 QA。
7. 对话严格交替，绝对不能出现连续两条相同 role。
8. 所有文本是能直接念出来的口语，不要书面表达、列表、括号说明。
9. role 只能是 system、user、assistant。
10. 输出必须是合法 JSON，且只能输出 JSON。

[输出 JSON 格式]
{{
  "call_type": "{desired_call_type}",
  "video_id": "{video_id}",
  "event_timeline": "{events_text.replace(chr(10), ' | ')}",
  "messages": [
    {{"role": "system", "content": "你是视频通话助手"}},
    {{"role": "user", "content": "..."}}
  ]
}}
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
    except ImportError as exc:
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


def make_prompt_id(video_id: str, call_type: str) -> str:
    key = f"{video_id}:{call_type}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


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
    if desired_call_type == "inbound" and normalized[1]["role"] != "user":
        raise ValueError("inbound sample must start with user after system")
    if desired_call_type == "outbound" and normalized[1]["role"] != "assistant":
        raise ValueError("outbound sample must start with assistant after system")

    dialogue_roles = {message["role"] for message in normalized[1:]}
    if "user" not in dialogue_roles or "assistant" not in dialogue_roles:
        raise ValueError("dialogue must contain both user and assistant")
    for i in range(1, len(normalized) - 1):
        if normalized[i]["role"] == normalized[i + 1]["role"]:
            raise ValueError(f"consecutive same role at index {i}: {normalized[i]['role']}")
    return normalized


def validate_transformed_output(
    payload: Dict[str, Any],
    *,
    desired_call_type: str,
) -> List[Dict[str, str]]:
    if not isinstance(payload, dict):
        raise ValueError("model output must be a JSON object")

    call_type = payload.get("call_type")
    if call_type != desired_call_type:
        raise ValueError(
            f"call_type mismatch: expected {desired_call_type!r}, got {call_type!r}"
        )

    messages = validate_messages(payload.get("messages"), desired_call_type)

    system_content = messages[0]["content"].strip()
    if "视频通话助手" not in system_content:
        raise ValueError("system prompt must contain '视频通话助手'")

    event_timeline = str(payload.get("event_timeline", "")).strip()
    return messages, event_timeline


def build_output_record(
    sample: Dict[str, Any],
    transformed_payload: Dict[str, Any],
    *,
    desired_call_type: str,
) -> Dict[str, Any]:
    messages, event_timeline = validate_transformed_output(
        transformed_payload,
        desired_call_type=desired_call_type,
    )
    video_id = sample.get("meta", {}).get("video_id", "unknown")
    event_intervals = sample.get("event_tags", {}).get("event_intervals", [])
    if not event_timeline:
        event_timeline = format_event_timeline(event_intervals)

    return {
        "prompt": event_timeline,
        "prompt_id": make_prompt_id(video_id, desired_call_type),
        "messages": messages,
        "video_id": video_id,
        "call_type": desired_call_type,
        "duration_sec": sample.get("meta", {}).get("duration_sec", 0),
        "event_count": len(event_intervals),
        "source_dataset": sample.get("meta", {}).get("source_dataset", ""),
    }


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
    video_id = sample.get("meta", {}).get("video_id", f"sample_{sample_index + 1}")
    sample_id = f"{video_id}:{desired_call_type}"
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
        except Exception as exc:
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
                            if sample.get("status") != "ok":
                                completed_results[sample_index] = {
                                    "ok": False,
                                    "output_record": None,
                                    "sample_id": f"skipped_{sample_index}",
                                    "call_type": call_type_plan[resumed_count + sample_index],
                                    "input_record": sample,
                                    "attempt_count": 0,
                                    "model_output": "",
                                    "model_output_stripped": "",
                                    "error": "status != ok, skipped",
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0,
                                }
                                continue
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
        description="Generate VideoChat-style spoken dialogues from video event tags."
    )
    parser.add_argument(
        "--input-jsonl",
        default=str(DEFAULT_INPUT_JSONL),
        help="Input video event tags JSONL file.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(DEFAULT_OUTPUT_JSONL),
        help="Output JSONL file.",
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
        default=0.8,
        help="Sampling temperature.",
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
        help="How many times to retry a failed generation.",
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
        help="How many requests to send concurrently.",
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
        help="Random seed for outbound/inbound split.",
    )
    parser.add_argument(
        "--resume-from-output",
        action="store_true",
        help="Resume from an existing output JSONL by skipping already-processed records.",
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
