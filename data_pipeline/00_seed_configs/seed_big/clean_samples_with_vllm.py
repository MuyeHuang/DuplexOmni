from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


DEFAULT_MODEL = (
    "gpt-4.1"
)
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"
_THREAD_LOCAL = threading.local()
VERDICT_REASONABLE = "【最终结论是合理】"
VERDICT_UNREASONABLE = "【最终结论是不合理】"

_THINK_TAG_RE = re.compile(r"<think>\s*.*?\s*</think>", re.DOTALL)


def strip_thinking_tags(text: str) -> str:
    """
    Qwen3.5 thinking mode may wrap intermediate content with <think>...</think>.
    We remove it before verdict detection so both thinking/non-thinking outputs work.
    """
    if not text:
        return ""
    return re.sub(_THINK_TAG_RE, "", text).strip()

SYSTEM_PROMPT = """你是一个剧本标签清洗助手。你的任务是检查输入的“剧本请求”标签包是否自洽、可执行、无明显冲突。
我们的Agent剧本包括S1和S2，S1是和用户交互的交互直觉脑，S2是给直觉脑传递信息的思考大脑。S1和S2之间没有明确的协议。
因此无论S2是什么风格，S1都应该保持它的人设，因此S2风格与S1风格冲突也是正常的。S1和S2风格不一致反而是好事，这代表S1会学到无论如何都保持冷静，整理好语言。
由于用户的需求是不确定的，因此任何轮次长度的对话都是有可能的。一轮指的是User说话，Assist回应，这样一个user-assist对算一轮。
！！不要过度关注轮次数量，除非特别离谱，否则轮次数量不应当作为拒绝的理由。

你需要重点检查：
1. 字段之间是否互相冲突。
2. 交互包、人物风格包、生成约束是否存在明显不合理或强冲突组合。
3. 禁止出现项应被视为全局负面约束，不需要删减。

请按下面格式输出：
最后给出结论的时候必须是以下二选一之一：
【最终结论是合理】
或
【最终结论是不合理】

前置输出：
你首先应该把你看到的标签自然语言化，平铺出来成为一段描述，禁止出现的内容不必复述。例如：
中性的用户和温和的 agent 进行条件修正对话，用户会质疑解题过程，助手需要解释过程，S2 中参与高延迟，对话可不由助手收尾。
然后，你应该观察这段描述对于Agent场景下是否合理。是否存在明显冲突。

要求：
- 不要输出额外的寒暄。
- 不需要返回 JSON，也不要改写输入结构。
"""


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


def iter_jsonl(path: Path, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f, start=1):
            if limit is not None and line_index > limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path, limit=limit))


def load_completed_request_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed: set[str] = set()
    for record in iter_jsonl(path):
        request_id = record.get("request_id")
        if isinstance(request_id, str) and request_id:
            completed.add(request_id)
    return completed


def build_user_prompt(sample: Dict[str, Any]) -> str:
    sample_text = json.dumps(sample, ensure_ascii=False, indent=2)
    return (
        "请检查下面这条剧本标签包，并判断是否合理。\n\n"
        f"{sample_text}"
    )


def detect_verdict(text: str) -> str:
    # Normalize whitespace so minor formatting differences won't break matching.
    compact = re.sub(r"\s+", "", text or "")
    idx_r = compact.rfind(VERDICT_REASONABLE)
    idx_u = compact.rfind(VERDICT_UNREASONABLE)
    if idx_r != -1 and idx_u != -1:
        return "reasonable" if idx_r > idx_u else "unreasonable"
    if idx_r != -1:
        return "reasonable"
    if idx_u != -1:
        return "unreasonable"
    return "missing"


def classify_output_issues(text: str) -> List[str]:
    issues: List[str] = []
    if detect_verdict(text) == "missing":
        issues.append("missing_verdict")
    return issues


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


def render_progress(
    completed: int,
    total: int,
    *,
    reasonable: int,
    unreasonable: int,
    error: int,
    missing_retries: int,
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
    avg_provider_total_tokens: float,
    avg_hidden_tokens: float,
) -> None:
    total = max(total, 1)
    width = 30
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100.0 * completed / total
    line = (
        f"\r[{bar}] {completed}/{total} {percent:5.1f}% "
        f"reasonable={reasonable} unreasonable={unreasonable} "
        f"error={error} missing_retries={missing_retries} "
        f"avg_in={avg_prompt_tokens:.1f} avg_out={avg_completion_tokens:.1f} "
        f"avg_total={avg_provider_total_tokens:.1f} avg_hidden={avg_hidden_tokens:.1f}"
    )
    print(line, end="", file=sys.stderr, flush=True)
    if completed >= total:
        print(file=sys.stderr, flush=True)


def process_one_sample(
    *,
    sample: Dict[str, Any],
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
    request = sample.get("剧本请求", {})
    request_id = request.get("请求ID", "unknown_request_id")
    prompt = build_user_prompt(sample)

    model_text = ""
    error_message = None
    missing_retries = 0
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    attempt_count = 0

    for attempt in range(1, max_retries + 1):
        attempt_count = attempt
        try:
            client = get_thread_client(api_base, api_key)
            if rate_limiter is not None:
                debug_log(debug, f"{request_id} attempt={attempt} waiting for rate limiter")
                rate_limiter.wait_for_token()
            debug_log(debug, f"{request_id} attempt={attempt} sending request")
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
            issues = classify_output_issues(model_text_stripped)
            verdict = detect_verdict(model_text_stripped)
            cleaned_sample = sample if verdict == "reasonable" else None

            if issues:
                missing_retries += 1
                error_message = f"Invalid model output: {', '.join(issues)}"
                debug_log(debug, f"{request_id} attempt={attempt} invalid output={issues}, retrying")
                if attempt < max_retries:
                    time.sleep(retry_sleep_seconds)
                    continue
                return {
                    "request_id": request_id,
                    "verdict": "error",
                    "input": sample,
                    "cleaned": cleaned_sample,
                    "model_output": model_text,
                    "model_output_stripped": model_text_stripped,
                    "error": f"{error_message} Exhausted retries.",
                    "missing_retries": missing_retries,
                    "attempt_count": attempt_count,
                    **token_stats,
                }

            debug_log(debug, f"{request_id} attempt={attempt} verdict={verdict}")
            return {
                "request_id": request_id,
                "verdict": verdict,
                "input": sample,
                "cleaned": cleaned_sample,
                "model_output": model_text,
                "model_output_stripped": model_text_stripped,
                "error": None,
                "missing_retries": missing_retries,
                "attempt_count": attempt_count,
                **token_stats,
            }
        except Exception as exc:  # pragma: no cover - depends on remote server
            error_message = f"{type(exc).__name__}: {exc}"
            debug_log(debug, f"{request_id} attempt={attempt} error={error_message}")
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)

    return {
        "request_id": request_id,
        "verdict": "error",
        "input": sample,
        "cleaned": None,
        "model_output": model_text,
        "model_output_stripped": strip_thinking_tags(model_text),
        "error": error_message,
        "missing_retries": missing_retries,
        "attempt_count": attempt_count,
        **token_stats,
    }


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
    resume_from_output: bool,
    debug: bool,
) -> None:
    samples = load_jsonl(input_path, limit=limit)
    resumed_count = 0
    completed_request_ids: set[str] = set()
    if resume_from_output:
        completed_request_ids = load_completed_request_ids(output_path)
        resumed_count = len(completed_request_ids)
        samples = [
            sample
            for sample in samples
            if sample.get("剧本请求", {}).get("请求ID") not in completed_request_ids
        ]

    total = len(samples)
    processed = 0
    verdict_counts = {"reasonable": 0, "unreasonable": 0, "error": 0}
    missing_retries_total = 0
    success_prompt_tokens_total = 0
    success_completion_tokens_total = 0
    success_total_tokens_total = 0
    success_count = 0
    error_prompt_tokens_total = 0
    error_completion_tokens_total = 0
    error_total_tokens_total = 0
    rate_limiter = RateLimiter(rpm_limit) if rpm_limit > 0 else None

    output_mode = "a" if resume_from_output and output_path.exists() else "w"
    with output_path.open(output_mode, encoding="utf-8") as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_index = {
                executor.submit(
                    process_one_sample,
                    sample=sample,
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_sleep_seconds=retry_sleep_seconds,
                    rate_limiter=rate_limiter,
                    debug=debug,
                ): index
                for index, sample in enumerate(samples)
            }

            completed_results: Dict[int, Dict[str, Any]] = {}
            next_to_write = 0

            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                result = future.result()
                completed_results[index] = result

                processed += 1
                missing_retries_total += int(result.get("missing_retries", 0))
                verdict = result["verdict"]
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

                if verdict == "error":
                    error_prompt_tokens_total += int(result.get("prompt_tokens", 0))
                    error_completion_tokens_total += int(result.get("completion_tokens", 0))
                    error_total_tokens_total += int(result.get("total_tokens", 0))
                else:
                    success_count += 1
                    success_prompt_tokens_total += int(result.get("prompt_tokens", 0))
                    success_completion_tokens_total += int(result.get("completion_tokens", 0))
                    success_total_tokens_total += int(result.get("total_tokens", 0))

                avg_prompt_tokens = (
                    success_prompt_tokens_total / success_count if success_count else 0.0
                )
                avg_completion_tokens = (
                    success_completion_tokens_total / success_count if success_count else 0.0
                )
                avg_provider_total_tokens = (
                    success_total_tokens_total / success_count if success_count else 0.0
                )
                avg_hidden_tokens = (
                    avg_provider_total_tokens - avg_prompt_tokens - avg_completion_tokens
                )

                render_progress(
                    processed,
                    total,
                    reasonable=verdict_counts["reasonable"],
                    unreasonable=verdict_counts["unreasonable"],
                    error=verdict_counts["error"],
                    missing_retries=missing_retries_total,
                    avg_prompt_tokens=avg_prompt_tokens,
                    avg_completion_tokens=avg_completion_tokens,
                    avg_provider_total_tokens=avg_provider_total_tokens,
                    avg_hidden_tokens=avg_hidden_tokens,
                )

                while next_to_write in completed_results:
                    record = completed_results.pop(next_to_write)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    next_to_write += 1

    print("done")
    print(f"input_path: {input_path}")
    print(f"output_path: {output_path}")
    print(f"resume_from_output: {resume_from_output}")
    print(f"resumed_count: {resumed_count}")
    print(f"processed: {processed}")
    print(f"concurrency: {concurrency}")
    print(f"rpm_limit: {rpm_limit}")
    print(f"missing_retries: {missing_retries_total}")
    print(f"success_count: {success_count}")
    avg_prompt_tokens = success_prompt_tokens_total / success_count if success_count else 0.0
    avg_completion_tokens = (
        success_completion_tokens_total / success_count if success_count else 0.0
    )
    avg_total_tokens = success_total_tokens_total / success_count if success_count else 0.0
    avg_hidden_tokens = avg_total_tokens - avg_prompt_tokens - avg_completion_tokens
    print(f"avg_prompt_tokens_success: {avg_prompt_tokens:.2f}")
    print(f"avg_completion_tokens_success: {avg_completion_tokens:.2f}")
    print(f"avg_total_tokens_provider_success: {avg_total_tokens:.2f}")
    print(f"avg_hidden_tokens_provider_success: {avg_hidden_tokens:.2f}")
    if verdict_counts["error"] > 0:
        print(
            f"avg_prompt_tokens_error: "
            f"{error_prompt_tokens_total / verdict_counts['error']:.2f}"
        )
        print(
            f"avg_completion_tokens_error: "
            f"{error_completion_tokens_total / verdict_counts['error']:.2f}"
        )
        print(
            f"avg_total_tokens_provider_error: "
            f"{error_total_tokens_total / verdict_counts['error']:.2f}"
        )
    for key, value in ("reasonable", verdict_counts["reasonable"]), ("unreasonable", verdict_counts["unreasonable"]), ("error", verdict_counts["error"]):
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean slot-label JSONL samples through a local vLLM OpenAI-compatible API."
    )
    parser.add_argument(
        "--input-jsonl",
        default="samples.jsonl",
        help="Input JSONL file containing generated slot-label samples.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="samples.cleaned.jsonl",
        help="Output JSONL file storing model verdicts and cleaned labels.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name exposed by the local vLLM server.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help="OpenAI-compatible base URL of the local vLLM server.",
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
        help="Sampling temperature for the cleaning call.",
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
        help="How many times to retry a failed request.",
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
        "--resume-from-output",
        action="store_true",
        help="Resume from an existing output JSONL by skipping already processed request_id values.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-request debug logs to stderr.",
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
        resume_from_output=args.resume_from_output,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
