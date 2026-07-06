"""
Video-chat wrapper for video_pipeline_base.py.

This file intentionally lives under video_stream and does not modify or import
the text pipeline. It keeps the existing video writer/director semantics, adds a
small director-prompt stability layer, and runs requests through multiple local
DeepSeek endpoints with an independent worker pool per endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import video_pipeline_base as base  # noqa: E402


DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"

DEFAULT_PROVIDER_API_BASES = (
    "local=http://localhost:8000/v1"
)

DEFAULT_INBOUND_PATH = (
    "data/content/train_sft.videochat.llava.full.inbound.jsonl"
)
DEFAULT_OUTBOUND_PATH = (
    "data/content/train_sft.videochat.llava.full.outbound.jsonl"
)
DEFAULT_SEED_PATH = (
    "data/seed_video/samples.cleaned.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/video_stream/pipeline_video"
)


VIDEO_DIRECTOR_SURFACE_TAG_GUIDE = """
### 视频通话导演稳定性补充
这部分只增强 video 数据的导演稳定性，不改变原有 S2 / [THINK] / [WAIT] 语义。

1. 视频通话数据只输出 User / Agent 的可说话内容和控制标签，不要输出画面旁白、视频路径、帧号、metadata、caption 字段名或“视频里显示”这类解释性文本。
2. 保留视频通话的即时口语感：短承接、短等待、被打断后的反应要像实时通话，不要改写成客服工单、长邮件或静态摘要。
3. 如果原始剧本是英文，输出 JSON 的 text 字段必须保持英文，不要翻译成中文；中英混合时也保持原脚本的语言比例。
4. `^` 只能表示真实说话触发点。数学、代码、普通文本里的插入符必须改写成口语或安全文本，例如 `x^n` 改成 `x 的 n 次方`，不能原样留在 text 里。
5. 如果触发锚点已经在完整句尾，不要硬加 `^`。句尾裸 `^`、标点前 `^`、`^[CUT]` 都是错误格式。
6. `[CUT]` 必须紧跟在 `^` 后很短的可播放文本之后；中文通常保留二到四个可读汉字/数字/字母，英文必须至少保留一个完整单词，不能把英文单词切开。
7. `[CUT]` 后必须保留未播完的幽灵文本；如果无法保留合理幽灵文本，就不要加 `[CUT]`，改成普通接话或只用 `^` 表达重叠。
8. `[PENDXS]` 只能放在 User 消息开头，表示双方短暂沉默后由用户恢复说话；不要凭空给 Agent 加沉默。
"""


ORIGINAL_DIRECTOR_SYSTEM_PROMPT = base.DIRECTOR_SYSTEM_PROMPT
ORIGINAL_VALIDATE_DIRECTOR_MESSAGES = base.validate_director_messages


def patch_video_prompts_and_validation() -> None:
    base.DIRECTOR_SYSTEM_PROMPT = (
        ORIGINAL_DIRECTOR_SYSTEM_PROMPT.rstrip()
        + "\n\n"
        + VIDEO_DIRECTOR_SURFACE_TAG_GUIDE.strip()
        + "\n"
    )
    base.construct_director_messages = construct_video_director_messages
    base.validate_director_messages = validate_video_director_messages


def _looks_english_script(script: str) -> bool:
    if not script:
        return False
    zh_count = sum(1 for char in script if "\u4e00" <= char <= "\u9fff")
    return zh_count / max(len(script), 1) < 0.1


def construct_video_director_messages(writer_record: Dict[str, Any]) -> List[Dict[str, str]]:
    split = writer_record["split"]
    first_speaker = "Agent" if split == "outbound" else "User"
    seed_summary = writer_record["seed_summary"]
    seed_notes = base.format_seed_constraints(seed_summary)
    director_guidance_lines = base.build_extra_guidance_lines(seed_summary, "director")
    director_extra_guidance = base.format_extra_guidance_block(director_guidance_lines)
    extra_guidance_section = (
        f"\n{base.GUIDANCE_SECTION_TITLE}\n{director_extra_guidance}\n"
    )
    lang_note = (
        "\n【语言要求】原始剧本为英文，输出 JSON 中的 text 字段内容必须保持英文，不要翻译成中文。\n"
        if _looks_english_script(str(writer_record.get("script", "")))
        else ""
    )

    user_content = f"""
请将以下剧本转换为导演标注数据。

【任务背景】
{base.build_director_task_background()}

【seed 字段释义】
{base.describe_seed_fields()}

【输出要求】
1. 只输出 JSON 列表，不要输出解释文字。
2. 只输出 User / Agent 对话消息，不要输出 system。
3. 保留外呼和内呼的首轮差异：outbound 首轮必须是 Agent，inbound 首轮必须是 User。
4. 如果剧本里出现 `[User沉默了X秒后]`，请把它转换成对应 User 消息开头的 `[PENDXS]`，例如 `[User沉默了1秒后]` -> `[PEND1S]`。
5. 当上一条消息是 User 时，下一条 Assistant 绝不能以 `「` 开头；必须先给一句不带 S2 的短承接，再在后续 Assistant 消息中引入 S2。
6. 每一个 `「...」` 都必须是完整且准确的信息片段，不能只写标题、关键词或提纲；Agent 正文不能说出 `「...」` 里没有先到达的关键事实。
7. 你必须严格参考 RAW seed 中的 `S2策略包` 来决定等待感、重试感、失败感和收尾方式，不能只看一眼摘要就忽略。
8. 如果 RAW seed 指向“中延迟”，就要多给一些自然垫话和过渡；如果指向“高延迟”，就要明确演出“仍在查询/推理/操作/仍在确认”；如果指向“故障无返回”，就要演出第一次失败、道歉、再次尝试、再次失败后不乱说并结束对话的过程。
9. 编剧不显式写 S2 消息，但导演必须根据剧本表层现象和 seed 约束，把内部触发、等待、返回、失败、重试准确补出来。
10. 保持视频通话数据特点：只保留对话双方可说出口的内容和控制标签，不要把视频标签、画面描述、路径或元数据写进最终 text。
11. `^` 和 `[CUT]` 必须符合真实语音打断时序；英文按完整单词边界截断，中文不能空切，普通数学/代码插入符不能原样输出。

【样本类型】
{split}

【首轮说话方要求】
{first_speaker}

【保留给助手的 system 设定】
{writer_record["writer_system"]}

【编剧约束摘要】
{seed_notes}

【当前这条样本对应的 RAW seed】
{base.format_raw_seed_block(seed_summary)}

【当前样本的 S2 延迟/失败演绎要求】
{base.build_delay_guidance(seed_summary)}
{extra_guidance_section}
【原始内容主题】
{writer_record["prompt"]}
{lang_note}
【原始剧本】
{writer_record["script"]}
"""
    messages = [
        {"role": "system", "content": base.DIRECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]
    base.assert_guidance_in_prompt(messages, director_guidance_lines, "director")
    return messages


BARE_CARET_PATTERN = re.compile(r"\^(?:\s|$|[，。！？、,.!?;:；：）)\]】}\"'”’])")
EMPTY_CUT_PATTERN = re.compile(r"\^\s*\[CUT\]")
USER_FORBIDDEN_SURFACE_PATTERN = re.compile(r"\^|\[CUT\]")


def validate_video_director_messages(
    split: str, messages: Sequence[Dict[str, str]]
) -> None:
    ORIGINAL_VALIDATE_DIRECTOR_MESSAGES(split, messages)
    for index, message in enumerate(messages):
        role = message["role"]
        content = message["content"].strip()
        if role == "user" and USER_FORBIDDEN_SURFACE_PATTERN.search(content):
            raise ValueError(f"第 {index + 1} 条 User 消息包含 ^/[CUT]，拒绝落盘")
        if role == "assistant":
            if BARE_CARET_PATTERN.search(content):
                raise ValueError(f"第 {index + 1} 条 Assistant 消息包含句尾/标点前裸 ^")
            if EMPTY_CUT_PATTERN.search(content):
                raise ValueError(f"第 {index + 1} 条 Assistant 消息包含 ^[CUT] 空切")
            if "[CUT]" in content and "^" not in content:
                raise ValueError(f"第 {index + 1} 条 Assistant 消息包含无 ^ 的 [CUT]")


class ProviderRateLimiter:
    def __init__(self, rpm: int):
        self.interval = 0.0 if rpm <= 0 else 60.0 / rpm
        self.last_call_time = 0.0
        self.lock = threading.Lock()

    def wait_for_token(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            wait_time = self.interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_call_time = time.time()


class VideoAPICaller:
    def __init__(self, api_key: str, base_url: str, model: str, max_tokens: int):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - runtime environment guard
            raise RuntimeError("无法导入 openai，video multi-provider 调度不可用") from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=99999)
        self.model = model
        self.max_tokens = max_tokens

    def _is_qwen_model(self) -> bool:
        model_lower = self.model.lower()
        return "qwen" in model_lower or "qwq" in model_lower

    def _is_deepseek_model(self) -> bool:
        return "deepseek" in self.model.lower()

    def _build_request_kwargs(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self._is_qwen_model():
            kwargs["temperature"] = 0.6
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        if self._is_deepseek_model():
            kwargs["temperature"] = 0.6
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "max",
                }
            }
        return kwargs

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        cleaned = re.sub(
            r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _extract_usage_detail(usage: Any) -> Dict[str, int]:
        cached_tokens = 0
        reasoning_tokens = 0
        cache_write_tokens = 0
        if usage is None:
            return {
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "cache_write_tokens": 0,
            }

        ptd = getattr(usage, "prompt_tokens_details", None)
        if ptd is not None:
            cached_tokens = getattr(ptd, "cached_tokens", None) or 0

        extra = getattr(usage, "model_extra", None) or {}
        if cached_tokens == 0:
            cached_tokens = (
                extra.get("cache_read_tokens")
                or extra.get("cached_tokens")
                or extra.get("effectiveCachedTokens")
                or 0
            )

        ctd = getattr(usage, "completion_tokens_details", None)
        if ctd is not None:
            reasoning_tokens = getattr(ctd, "reasoning_tokens", None) or 0
        cache_write_tokens = extra.get("cache_write_tokens", 0) or 0

        return {
            "cached_tokens": int(cached_tokens),
            "reasoning_tokens": int(reasoning_tokens),
            "cache_write_tokens": int(cache_write_tokens),
        }

    def call(self, messages: List[Dict[str, str]]) -> Tuple[str, Tuple[int, int, int, Dict[str, int]]]:
        try:
            response = self.client.chat.completions.create(
                **self._build_request_kwargs(messages)
            )
            if not response or not response.choices:
                print("[API错误] 无有效响应")
                return "", (0, 0, 0, {})

            content = response.choices[0].message.content
            if not content:
                return "", (0, 0, 0, {})

            usage = response.usage
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            detail = self._extract_usage_detail(usage)
            return (
                self._strip_think_tags(content),
                (completion_tokens, prompt_tokens, total_tokens, detail),
            )
        except Exception as exc:
            print(f"[API系统错误] {type(exc).__name__}: {exc}")
            return "", (0, 0, 0, {})


@dataclass
class ProviderState:
    name: str
    api_base: str
    model_name: str
    api_key: str
    rpm: int
    max_tokens: int
    caller: VideoAPICaller = field(init=False)
    limiter: ProviderRateLimiter = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock)
    success: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        self.caller = VideoAPICaller(
            self.api_key,
            self.api_base,
            self.model_name,
            self.max_tokens,
        )
        self.limiter = ProviderRateLimiter(self.rpm)

    def mark_success(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        with self.lock:
            self.success += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens

    def mark_failure(self) -> None:
        with self.lock:
            self.failures += 1

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "name": self.name,
                "api_base": self.api_base,
                "model_name": self.model_name,
                "success": self.success,
                "failures": self.failures,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }


@dataclass
class WorkItem:
    seq: int
    payload: Dict[str, Any]
    attempts: int = 0


def parse_named_api_bases(raw: str) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = []
    for index, item in enumerate(part.strip() for part in raw.split(",") if part.strip()):
        if "=" in item:
            name, api_base = item.split("=", 1)
            name = name.strip()
            api_base = api_base.strip()
        else:
            name = f"p{index}"
            api_base = item
        if not name or not api_base:
            raise ValueError(f"非法 provider api_base 配置: {item!r}")
        specs.append((name, api_base))
    if not specs:
        raise ValueError("provider-api-bases 为空")
    return specs


def parse_provider_models(
    raw: str,
    api_specs: Sequence[Tuple[str, str]],
    default_model: str,
) -> List[Tuple[str, str, str]]:
    if not raw.strip():
        return [(name, default_model, api_base) for name, api_base in api_specs]

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) == 1 and "=" not in parts[0]:
        return [(name, parts[0], api_base) for name, api_base in api_specs]

    model_pairs: List[Tuple[str, str]] = []
    for index, item in enumerate(parts):
        if "=" in item:
            name, model = item.split("=", 1)
            model_pairs.append((name.strip(), model.strip()))
        else:
            model_pairs.append((api_specs[index][0], item))

    model_by_name = {name: model for name, model in model_pairs}
    if all(name in model_by_name for name, _ in api_specs):
        return [(name, model_by_name[name], api_base) for name, api_base in api_specs]

    if len(model_pairs) == len(api_specs):
        return [
            (api_name, model, api_base)
            for (api_name, api_base), (_, model) in zip(api_specs, model_pairs)
        ]

    raise ValueError("provider-models 数量或名称与 provider-api-bases 不匹配")


def build_providers(args: argparse.Namespace) -> List[ProviderState]:
    api_specs = parse_named_api_bases(args.provider_api_bases)
    provider_specs = parse_provider_models(
        args.provider_models,
        api_specs,
        args.model_name,
    )
    return [
        ProviderState(
            name=name,
            api_base=api_base,
            model_name=model_name,
            api_key=args.api_key,
            rpm=args.provider_rpm,
            max_tokens=args.max_tokens,
        )
        for name, model_name, api_base in provider_specs
    ]


def token_tuple_from_writer(record: Dict[str, Any]) -> Tuple[int, int, int]:
    return (
        int(record.get("prompt_tokens", 0) or 0),
        int(record.get("completion_tokens", 0) or 0),
        int(record.get("total_tokens", 0) or 0),
    )


def token_tuple_from_director(record: Dict[str, Any]) -> Tuple[int, int, int]:
    meta = record.get("metadata", {}) or {}
    return (
        int(meta.get("director_prompt_tokens", 0) or 0),
        int(meta.get("director_completion_tokens", 0) or 0),
        int(meta.get("director_total_tokens", 0) or 0),
    )


def run_provider_pool(
    *,
    stage: str,
    split: str,
    tasks: Sequence[Dict[str, Any]],
    existing_records: Sequence[Dict[str, Any]],
    output_path: str,
    providers: Sequence[ProviderState],
    workers_per_provider: int,
    save_interval: int,
    task_retries: int,
    process_fn: Callable[[Dict[str, Any], ProviderState], Dict[str, Any]],
    token_fn: Callable[[Dict[str, Any]], Tuple[int, int, int]],
) -> List[Dict[str, Any]]:
    if not tasks:
        print(f"[{stage}:{split}] 没有待处理任务")
        return list(existing_records)
    if workers_per_provider <= 0:
        raise ValueError("provider-workers 必须大于 0")

    records_by_id = {record["sample_id"]: record for record in existing_records}
    task_queue: Queue[WorkItem] = Queue()
    result_queue: Queue[Tuple[str, Any]] = Queue()
    stop_event = threading.Event()

    for seq, payload in enumerate(tasks):
        task_queue.put(WorkItem(seq=seq, payload=payload))

    print(
        f"[{stage}:{split}] provider 出水口: "
        f"{', '.join(f'{p.name}@{p.api_base}' for p in providers)}；"
        f"每口 workers={workers_per_provider}"
    )

    def worker(provider: ProviderState, worker_idx: int) -> None:
        worker_name = f"{provider.name}-{worker_idx}"
        while not stop_event.is_set():
            try:
                item = task_queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                result = process_fn(item.payload, provider)
                result_queue.put(("success", (result, provider.name)))
            except Exception as exc:
                item.attempts += 1
                provider.mark_failure()
                if item.attempts <= task_retries:
                    task_queue.put(item)
                    result_queue.put(
                        (
                            "retry",
                            {
                                "sample_id": item.payload.get("sample_id", ""),
                                "provider": provider.name,
                                "worker": worker_name,
                                "attempts": item.attempts,
                                "error": str(exc)[:500],
                            },
                        )
                    )
                else:
                    result_queue.put(
                        (
                            "failure",
                            {
                                "sample_id": item.payload.get("sample_id", ""),
                                "provider": provider.name,
                                "worker": worker_name,
                                "attempts": item.attempts,
                                "error": str(exc),
                            },
                        )
                    )
            finally:
                task_queue.task_done()

    def dispatcher(provider: ProviderState) -> None:
        with ThreadPoolExecutor(
            max_workers=workers_per_provider,
            thread_name_prefix=f"{stage}-{split}-{provider.name}",
        ) as executor:
            futures = [
                executor.submit(worker, provider, worker_idx)
                for worker_idx in range(workers_per_provider)
            ]
            for future in as_completed(futures):
                if stop_event.is_set():
                    break
                try:
                    future.result()
                except Exception as exc:  # pragma: no cover - worker guard
                    result_queue.put(
                        (
                            "worker_error",
                            {
                                "provider": provider.name,
                                "error": str(exc)[:500],
                            },
                        )
                    )

    threads = [
        threading.Thread(target=dispatcher, args=(provider,), daemon=True)
        for provider in providers
    ]
    for thread in threads:
        thread.start()

    completed = 0
    failures: List[Dict[str, Any]] = []
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0

    pbar = tqdm(total=len(tasks), desc=f"{stage.capitalize()}-{split}", smoothing=0)
    while completed + len(failures) < len(tasks):
        try:
            event_type, payload = result_queue.get(timeout=5.0)
        except Empty:
            if task_queue.empty() and not any(thread.is_alive() for thread in threads):
                break
            continue

        if event_type == "success":
            result, provider_name = payload
            prompt_tokens, completion_tokens, total_tokens = token_fn(result)
            provider = next(p for p in providers if p.name == provider_name)
            provider.mark_success(prompt_tokens, completion_tokens, total_tokens)

            records_by_id[result["sample_id"]] = result
            completed += 1
            session_prompt_tokens += prompt_tokens
            session_completion_tokens += completion_tokens
            session_total_tokens += total_tokens
            avg_tokens = session_total_tokens / completed if completed else 0
            pbar.set_postfix({"Suc": completed, "AvgTok": int(avg_tokens)})
            pbar.update(1)

            if completed % save_interval == 0:
                base.save_jsonl(
                    [records_by_id[key] for key in sorted(records_by_id)],
                    output_path,
                )
        elif event_type == "failure":
            failures.append(payload)
            pbar.update(1)
            tqdm.write(
                f"[{stage}:{split}] {payload['sample_id']} 失败: "
                f"{payload['provider']} {payload['error']}"
            )
        elif event_type == "retry":
            pbar.set_postfix({"Retry": payload["provider"], "Suc": completed})
        elif event_type == "worker_error":
            pbar.set_postfix({"WorkerErr": payload["provider"], "Suc": completed})
            tqdm.write(
                f"[{stage}:{split}] provider {payload['provider']} worker 异常: "
                f"{payload['error']}"
            )

    stop_event.set()
    for thread in threads:
        thread.join(timeout=5.0)
    pbar.close()

    incomplete = len(tasks) - completed - len(failures)
    if incomplete:
        raise RuntimeError(f"[{stage}:{split}] {incomplete} 条任务未完成，拒绝静默完成")

    final_records = [records_by_id[key] for key in sorted(records_by_id)]
    base.save_jsonl(final_records, output_path)
    print(f"[{stage}:{split}] 完成后总记录数: {len(final_records)}")
    print(
        f"[{stage}:{split}] Token 统计 — "
        f"prompt={session_prompt_tokens:,} "
        f"completion={session_completion_tokens:,} "
        f"total={session_total_tokens:,}"
    )
    for provider in providers:
        snap = provider.snapshot()
        print(
            f"[{stage}:{split}] provider {snap['name']} — "
            f"success={snap['success']} failures={snap['failures']} "
            f"prompt={snap['prompt_tokens']} completion={snap['completion_tokens']} "
            f"api_base={snap['api_base']}"
        )
    if failures:
        raise RuntimeError(f"[{stage}:{split}] {len(failures)} 条任务失败，已拒绝静默完成")
    return final_records


def run_writer_stage(
    split: str,
    config: base.PipelineConfig,
    content_items: Sequence[Dict[str, Any]],
    seed_items: Sequence[Dict[str, Any]],
    providers: Sequence[ProviderState],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    paths = base.build_output_paths(config, split)
    existing_records = base.load_jsonl(paths["writer"])
    completed_ids = {record["sample_id"] for record in existing_records}
    expanded_items = base.expand_content_items_with_seeds(content_items, seed_items, config)
    tasks = base.select_tasks(
        expanded_items,
        completed_ids,
        config.debug_limit,
        config.random_seed,
        f"{split}:writer",
    )
    print(
        f"[writer:{split}] 总样本 {len(expanded_items)}，"
        f"已完成 {len(completed_ids)}，待执行 {len(tasks)}"
    )

    def process(item: Dict[str, Any], provider: ProviderState) -> Dict[str, Any]:
        result = base.process_writer_sample(
            item,
            seed_items,
            config,
            provider.caller,
            provider.limiter,
        )
        result["writer_provider"] = provider.name
        result["writer_model_name"] = provider.model_name
        return result

    return run_provider_pool(
        stage="writer",
        split=split,
        tasks=tasks,
        existing_records=existing_records,
        output_path=paths["writer"],
        providers=providers,
        workers_per_provider=args.provider_workers,
        save_interval=config.save_interval,
        task_retries=args.task_retries,
        process_fn=process,
        token_fn=token_tuple_from_writer,
    )


def run_director_stage(
    split: str,
    config: base.PipelineConfig,
    writer_records: Sequence[Dict[str, Any]],
    providers: Sequence[ProviderState],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    paths = base.build_output_paths(config, split)
    existing_records = base.load_jsonl(paths["director"])
    completed_ids = {record["sample_id"] for record in existing_records}
    writer_items = [record for record in writer_records if record["split"] == split]
    tasks = base.select_tasks(
        writer_items,
        completed_ids,
        config.debug_limit,
        config.random_seed,
        f"{split}:director",
    )
    print(
        f"[director:{split}] 总剧本 {len(writer_items)}，"
        f"已完成 {len(completed_ids)}，待执行 {len(tasks)}"
    )
    guidance_config = base.load_guidance_config(config.seed_path)

    def process(writer_record: Dict[str, Any], provider: ProviderState) -> Dict[str, Any]:
        result = base.process_director_record(
            writer_record,
            config,
            guidance_config,
            provider.caller,
            provider.limiter,
        )
        metadata = result.setdefault("metadata", {})
        metadata["director_provider"] = provider.name
        metadata["director_model_name"] = provider.model_name
        return result

    return run_provider_pool(
        stage="director",
        split=split,
        tasks=tasks,
        existing_records=existing_records,
        output_path=paths["director"],
        providers=providers,
        workers_per_provider=args.provider_workers,
        save_interval=config.save_interval,
        task_retries=args.task_retries,
        process_fn=process,
        token_fn=token_tuple_from_director,
    )


def resolve_splits(split: str) -> List[str]:
    if split == "both":
        return ["inbound", "outbound"]
    return [split]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video multi-provider pipeline")
    parser.add_argument(
        "--mode",
        choices=["writer_only", "director_only", "full_pipeline"],
        default="full_pipeline",
    )
    parser.add_argument(
        "--split",
        choices=["inbound", "outbound", "both"],
        default="both",
    )
    parser.add_argument("--debug-limit", type=int, default=None)
    parser.add_argument("--writer-seed-sample-size", type=int, default=3)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--max-workers", type=int, default=1500)
    parser.add_argument("--rpm-limit", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay-base", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--provider-api-bases", default=DEFAULT_PROVIDER_API_BASES)
    parser.add_argument("--provider-models", default="")
    parser.add_argument("--provider-workers", type=int, default=1500)
    parser.add_argument("--provider-rpm", type=int, default=0)
    parser.add_argument("--task-retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--seed-path", default=DEFAULT_SEED_PATH)
    parser.add_argument("--inbound-path", default=DEFAULT_INBOUND_PATH)
    parser.add_argument("--outbound-path", default=DEFAULT_OUTBOUND_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> base.PipelineConfig:
    return base.PipelineConfig(
        api_key=args.api_key,
        api_base=args.api_base,
        model_name=args.model_name,
        seed_path=args.seed_path,
        inbound_path=args.inbound_path,
        outbound_path=args.outbound_path,
        output_dir=args.output_dir,
        mode=args.mode,
        split=args.split,
        debug_limit=args.debug_limit,
        save_interval=args.save_interval,
        max_workers=args.max_workers,
        rpm_limit=args.rpm_limit,
        max_retries=args.max_retries,
        retry_delay_base=args.retry_delay_base,
        random_seed=args.random_seed,
        writer_seed_sample_size=args.writer_seed_sample_size,
    )


def main() -> None:
    args = parse_args()
    patch_video_prompts_and_validation()
    config = build_config(args)
    os.makedirs(config.output_dir, exist_ok=True)

    providers = build_providers(args)
    print(
        f"[provider] count={len(providers)} "
        f"workers_per_provider={args.provider_workers} "
        f"max_tokens={args.max_tokens}"
    )

    all_seed_items: List[Dict[str, Any]] = []
    if config.mode != "director_only":
        all_seed_items = base.load_seed_items(config.seed_path)
        if len(all_seed_items) < config.writer_seed_sample_size:
            raise ValueError("seed 数量不足，无法启动编剧阶段")

    split_to_path = {
        "inbound": config.inbound_path,
        "outbound": config.outbound_path,
    }
    for split in resolve_splits(config.split):
        print(f"\n========== 处理 split: {split} ==========")
        content_items = base.load_content_items(split_to_path[split], split)
        if not content_items:
            print(f"[{split}] 未读取到有效 content")
            continue

        if config.mode == "writer_only":
            run_writer_stage(split, config, content_items, all_seed_items, providers, args)
            continue

        if config.mode == "director_only":
            writer_path = base.build_output_paths(config, split)["writer"]
            writer_records = base.load_jsonl(writer_path)
            if not writer_records:
                print(f"[director:{split}] 未找到编剧中间文件: {writer_path}")
                continue
            run_director_stage(split, config, writer_records, providers, args)
            continue

        writer_records = run_writer_stage(
            split,
            config,
            content_items,
            all_seed_items,
            providers,
            args,
        )
        run_director_stage(split, config, writer_records, providers, args)


if __name__ == "__main__":
    main()
