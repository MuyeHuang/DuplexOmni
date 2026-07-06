#!/usr/bin/env python3
"""
classify_s1_s2.py — S1/S2 分流

S1：创意写作、文学创作、语言翻译润色、情感支持、简单通识问答、日常对话
S2：数学计算、逻辑推理、代码生成/分析、实时数据检索、需要精确验证的信息

读取 cleaned jsonl，对每条记录分类，输出比例统计。
"""

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-V4-Flash")

# OpenAI-compatible endpoints, comma-separated.
ENDPOINTS = [
    ep.strip()
    for ep in os.environ.get("API_BASES", "http://localhost:8000/v1").split(",")
    if ep.strip()
]
API_KEY = os.environ.get("API_KEY", "EMPTY")

CLASSIFY_PROMPT = """\
你是一个语音助手。判断以下用户消息，你是否能在不调用任何工具或深度思考的情况下立即准确回答。

S1（立即回答，无需工具或推理）：
- 创意写作、文学创作、续写、改写、翻译润色
- 情感支持、闲聊、日常对话
- 广为人知的通识（高中教育水平内的常识、常见历史事件、基础概念解释）

S2（需要工具或深度推理才能准确回答）：
- 任何数学计算（包括简单的，答错有实质影响）
- 逻辑推理、自然语言推断（NLI）、有唯一正确答案的推理题
- 代码生成、代码分析、算法设计
- 实时/动态数据（天气、价格、新闻、服务状态）
- 专业/偏门知识（需要专业背景才能准确回答的：高等数学、高等物理、医学细节、法律条文、工程技术等）
- 需要检索才能确认准确性的具体数据（排名、统计数字、人名细节）

【用户消息】
{first_user}

只回答 S1 或 S2，不要任何解释。"""

# 每个 endpoint 一个固定 client
_clients: dict[str, OpenAI] = {
    ep: OpenAI(api_key=API_KEY, base_url=ep, timeout=120)
    for ep in ENDPOINTS
}


def get_sample_id(record: dict) -> str:
    sample_id = record.get("sample_id")
    if sample_id is None:
        return ""
    return str(sample_id).strip()


def classify_record(record: dict, endpoint: str | None = None) -> str | None:
    orig = record.get("original_messages", [])
    first_user = next((m.get("content", "") for m in orig if m.get("role") == "user"), "")
    if not first_user:
        first_user = record.get("script", "")[:200]

    prompt = CLASSIFY_PROMPT.format(first_user=first_user[:500])
    # endpoint 由调用方绑定，fallback 到第一个
    ep = endpoint or ENDPOINTS[0]

    for attempt in range(3):
        try:
            resp = _clients[ep].chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16193,
                extra_body={
                    "chat_template_kwargs": {
                        "thinking": True,
                        "reasoning_effort": "low",
                    }
                },
            )
            raw = resp.choices[0].message.content or ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip().upper()
            if "S2" in raw:
                return "S2"
            if "S1" in raw:
                return "S1"
        except Exception:
            pass
        time.sleep(1)
    return None


def load_done_ids(output_path: Path | None) -> set[str]:
    done_ids: set[str] = set()
    if not output_path or not output_path.exists():
        return done_ids

    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                sample_id = get_sample_id(json.loads(line))
            except Exception:
                continue
            if sample_id:
                done_ids.add(sample_id)

    return done_ids


def load_pending_records(input_path: str, done_ids: set[str]) -> tuple[list[dict], int, int]:
    records: list[dict] = []
    total_input = 0
    missing_sample_id = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except Exception:
                continue

            total_input += 1
            sample_id = get_sample_id(record)
            if sample_id:
                if sample_id in done_ids:
                    continue
            else:
                missing_sample_id += 1
            records.append(record)

    return records, total_input, missing_sample_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sample", type=int, default=0, help="0=全量")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = []
    write_lock = threading.Lock()
    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(output_path)
    records, total_input, missing_sample_id = load_pending_records(args.input, done_ids)

    if args.sample > 0:
        random.seed(42)
        records = random.sample(records, min(args.sample, len(records)))

    print(
        f"输入共 {total_input} 条，已完成 {len(done_ids)} 条，"
        f"待处理 {len(records)} 条，并发 {args.max_workers}"
    )
    if missing_sample_id:
        print(f"[warn] 有 {missing_sample_id} 条缺少 sample_id，无法参与断点续传去重")
    if not records:
        print("没有待处理样本，退出。")
        return

    # 按 endpoint 数量分片，每个 endpoint 独立 1900 worker
    n_ep = len(ENDPOINTS)
    workers_per_ep = args.max_workers
    shards = [records[i::n_ep] for i in range(n_ep)]

    all_futures: dict = {}
    executors = []
    for ep, shard in zip(ENDPOINTS, shards):
        ex = ThreadPoolExecutor(max_workers=workers_per_ep)
        executors.append(ex)
        for r in shard:
            all_futures[ex.submit(classify_record, r, ep)] = r

    for future in tqdm(as_completed(all_futures), total=len(all_futures)):
        r = all_futures[future]
        try:
            label = future.result()
        except Exception:
            label = None

        entry = {
            "sample_id": get_sample_id(r),
            "label": label,
            "first_user": next(
                (m.get("content", "")[:100] for m in r.get("original_messages", []) if m.get("role") == "user"),
                ""
            ),
        }
        results.append(entry)

        if output_path and label:
            with write_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    for ex in executors:
        ex.shutdown(wait=False)

    counts = Counter(r["label"] for r in results)
    total = len(results)
    failed = counts.get(None, 0)
    print(f"\n=== 分流统计（共 {total} 条）===")
    print(f"  S1 : {counts['S1']:5d}  ({counts['S1']/max(total-failed,1)*100:.1f}%)")
    print(f"  S2 : {counts['S2']:5d}  ({counts['S2']/max(total-failed,1)*100:.1f}%)")
    print(f"  失败: {failed:5d}  ({failed/max(total,1)*100:.1f}%)")

    # 展示 S1/S2 各几条例子
    for label in ("S1", "S2"):
        examples = [r for r in results if r["label"] == label][:5]
        print(f"\n--- {label} 示例 ---")
        for e in examples:
            print(f"  {e['first_user'][:80]}")


if __name__ == "__main__":
    main()
