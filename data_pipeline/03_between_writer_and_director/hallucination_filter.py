#!/usr/bin/env python3
"""
hallucination_filter.py — 幻觉检测后置管线

处理 writer.cleaned.jsonl，对每条数据判断：
  1. User 的问题/任务是否有足够信息可回答（问题完整性）
  2. Assistant 是否凭借幻觉回答了一个信息不足的问题

给每条记录加三个键：
  - hallucination: true / false / null(失败)
  - hallucination_reason: 判断依据
  - hallucination_suggestion: 修改建议

输出到 <stem>.halluc.jsonl，原文件不改动。
支持断点续传、并发处理。
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue

from openai import OpenAI
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-V4-Flash")
API_KEY = os.environ.get("API_KEY", "EMPTY")

ENDPOINTS: list[str] = [
    ep.strip()
    for ep in os.environ.get("API_BASES", "http://localhost:8000/v1").split(",")
    if ep.strip()
]

# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
你是对话数据质检专家。你的任务是判断一段对话剧本中是否存在幻觉。

幻觉的唯一定义：
  用户提出了需要具体数据/内容才能回答的任务，但用户从未在对话中提供这些必要信息，
  而 Agent 却表现得像已经获得了这些信息并给出了答案。

  核心判断标准：用户是否把任务所需的关键信息说清楚了？
  如果关键信息缺失（题目没给、原文没给、案情没说），Agent 却在回答 → 幻觉。
  如果用户信息充分（问了"北京几度"、给了原文、给了案情），Agent 给出具体回答 → 不算幻觉。

  重要：不要以"Agent 能否访问外部数据源"来判断幻觉。
  下游管线会为 Agent 注入实时数据（天气、搜索等），Agent 陈述具体事实结果不是幻觉。
  幻觉只发生在：用户根本没把任务交代清楚，Agent 却在凭空编造条件。

典型幻觉案例（用户没给关键信息，Agent 编造了条件）：
  - 用户说"帮我算这道题"，但没给题目 → 幻觉
  - 用户说"帮我翻译这段话"，但没给原文 → 幻觉
  - 用户说"帮我看看我发的那份文件"，但剧本里从未出现文件 → 幻觉
  - 用户提到"上次说的那个事"，但剧本里没提过"那个事"是什么 → 幻觉
  - 外呼场景：Agent 的任务是解释具体法律条款，但用户未提供任何案情信息，Agent 却断言"根据您的情况..." → 幻觉
  - 外呼场景：Agent 的任务是推荐具体产品，但用户未提供需求，Agent 却直接列出了具体产品型号和价格 → 幻觉

不算幻觉的情况：
  - 用户任务本身不需要具体内容 → 不算
  - 用户给了完整信息，Agent 基于这些作答 → 不算
  - Agent 追问用户补充信息 → 不算
  - Agent 给出通用回答，没说"根据您提供的..." → 不算
  - 用户问了有明确答案的问题（"北京几度""什么是守护进程"），Agent 给出具体答案 → 不算（下游会补数据）
  - Agent 基于用户已提供信息做了合理推论/常识补全，而非凭空编造 → 不算
  - 创意生成类任务（"给几个发明创意""帮我写首诗""编个故事"），Agent 产出具体内容 → 不算（创意任务不需要用户提供具体数据）
  - Agent 声称做过核查/验证（"交叉验证了""查过了""核实过专利"），因为 Agent 具备联网搜索和工具调用能力 → 不算（除非剧本中用户明确质疑且 Agent 承认未核查）

请输出一个 JSON 对象：
{
  "hallucination": true 或 false,
  "reason": "简短说明判断依据（如无幻觉则输出空字符串\"\"）",
  "suggestion": "如果存在幻觉，给出具体的修改建议（如何补充 User 台词中缺失的信息，或如何修改 Agent 回答），如无幻觉则输出空字符串\"\""
}
只输出 JSON，不输出任何解释或 markdown。"""


def build_user_content(record: dict) -> str:
    prompt = record.get("prompt", "")
    script = record.get("script", "")
    orig = record.get("original_messages", [])
    orig_user = next((m.get("content", "") for m in orig if m.get("role") == "user"), "")
    orig_snippet = orig_user[:500] if orig_user else "（无）"

    # outbound: writer_system 中包含 Agent 的任务描述，给判别器作为上下文
    task = record.get("writer_system", "") or record.get("original_system", "")
    task_snippet = task[:800] if task else ""

    if task_snippet:
        return f"【主题】{prompt}\n\n【Agent 任务描述】\n{task_snippet}\n\n【原始对话内容（参考）】\n{orig_snippet}\n\n【剧本】\n{script}"
    else:
        return f"【主题】{prompt}\n\n【原始对话内容（参考）】\n{orig_snippet}\n\n【剧本】\n{script}"


# ──────────────────────────────────────────────────────────────────────────────
# LLM 调用
# ──────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_tok = {"prompt": 0, "completion": 0, "calls": 0}


def _extract_text(choice) -> str:
    msg = choice.message
    if msg.content:
        return msg.content
    reasoning = getattr(msg, "reasoning", None)
    if reasoning:
        return reasoning
    extra = getattr(msg, "model_extra", {}) or {}
    return extra.get("reasoning_content", "") or extra.get("content", "") or ""


def call_llm(client: OpenAI, user_content: str) -> dict:
    """单次调用，异常直接上抛。"""
    resp = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=16193,
        extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "low"}},
    )
    if resp.usage:
        with _lock:
            _tok["prompt"] += resp.usage.prompt_tokens
            _tok["completion"] += resp.usage.completion_tokens
            _tok["calls"] += 1

    raw = _extract_text(resp.choices[0])
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        return json.loads(raw[start:end + 1])
    raise ValueError(f"无法解析 JSON: {raw[:200]}")


# ──────────────────────────────────────────────────────────────────────────────
# 单条处理
# ──────────────────────────────────────────────────────────────────────────────
def process_record(record: dict, client: OpenAI) -> dict:
    """异常直接上抛，由调用方决定重试/退回队列。"""
    result = call_llm(client, build_user_content(record))
    rec = dict(record)
    rec["hallucination"] = bool(result.get("hallucination", False))
    rec["hallucination_reason"] = result.get("reason", "")
    rec["hallucination_suggestion"] = result.get("suggestion", "")
    return rec


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("hallucination_filter 启动", flush=True)

    parser = argparse.ArgumentParser(description="幻觉检测后置管线")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-workers", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    print(f"input={args.input}  max_workers={args.max_workers}  limit={args.limit}", flush=True)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / (input_path.stem + ".halluc.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"output={output_path}", flush=True)

    # 断点续传
    done_ids: set = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    sid = json.loads(line).get("sample_id")
                    if sid:
                        done_ids.add(sid)
                except Exception:
                    pass
        print(f"[resume] 已完成 {len(done_ids)} 条，跳过", flush=True)

    with open(input_path, encoding="utf-8") as f:
        if args.limit > 0:
            all_lines = []
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                all_lines.append(line)
        else:
            all_lines = f.readlines()
    print(f"读取文件完成，共 {len(all_lines)} 行", flush=True)

    todo_lines = []
    for line in all_lines:
        try:
            sid = json.loads(line).get("sample_id")
            if sid not in done_ids:
                todo_lines.append(line)
        except Exception:
            todo_lines.append(line)

    total_input = len(all_lines)
    print(f"总条数: {total_input}，待处理: {len(todo_lines)}，并发: {args.max_workers}", flush=True)
    sys.stdout.flush()

    if not todo_lines:
        print("全部已完成，退出。", flush=True)
        return

    stats = {"ok": 0, "halluc": 0, "null": 0, "error": 0}
    write_lock = threading.Lock()
    stats_lock = threading.Lock()

    def write_record(rec: dict):
        with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 共享任务队列
    task_queue: Queue = Queue()
    for line in todo_lines:
        try:
            task_queue.put(json.loads(line))
        except json.JSONDecodeError:
            pass

    total_tasks = task_queue.qsize()
    result_queue: Queue = Queue()
    stop_event = threading.Event()

    def _is_rate_limit(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "429" in msg or "rate limit" in msg or "too many" in msg

    def provider_worker(client: OpenAI, worker_idx: int) -> None:
        while not stop_event.is_set():
            try:
                record = task_queue.get(timeout=0.5)
            except Empty:
                break
            try:
                rec = process_record(record, client)
                result_queue.put(("done", rec))
            except Exception as exc:
                if _is_rate_limit(exc):
                    task_queue.put(record)
                    result_queue.put(("requeued", None))
                    time.sleep(1)
                else:
                    result_queue.put(("error", None))

    def provider_dispatcher(ep: str) -> None:
        client = OpenAI(api_key=API_KEY, base_url=ep, timeout=300)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(provider_worker, client, i) for i in range(args.max_workers)]
            for f in as_completed(futures):
                if stop_event.is_set():
                    break
                try:
                    f.result()
                except Exception:
                    pass

    dispatcher_threads = [
        threading.Thread(target=provider_dispatcher, args=(ep,), daemon=True)
        for ep in ENDPOINTS
    ]
    for t in dispatcher_threads:
        t.start()

    pbar = tqdm(total=total_tasks, desc="幻觉检测", file=sys.stdout, smoothing=0)
    processed = 0
    while processed < total_tasks:
        try:
            event, rec = result_queue.get(timeout=5.0)
        except Empty:
            if all(not t.is_alive() for t in dispatcher_threads) and task_queue.empty():
                break
            continue

        if event == "requeued":
            continue

        processed += 1
        if event == "done" and rec is not None:
            write_record(rec)
            h = rec.get("hallucination")
            with stats_lock:
                if h is True:
                    stats["halluc"] += 1
                elif h is False:
                    stats["ok"] += 1
                else:
                    stats["null"] += 1
        else:
            with stats_lock:
                stats["error"] += 1

        with stats_lock:
            total_done = stats["ok"] + stats["halluc"] + stats["null"]
        with _lock:
            p, c = _tok["prompt"], _tok["completion"]
        pbar.set_postfix({
            "ok": stats["ok"],
            "halluc": stats["halluc"],
            "halluc%": f"{stats['halluc'] / max(total_done, 1) * 100:.1f}%",
            "ptok": f"{p / 1e6:.2f}M",
        }, refresh=False)
        pbar.update(1)

    stop_event.set()

    total_done = stats["ok"] + stats["halluc"] + stats["null"]
    cumulative = len(done_ids) + total_done
    with _lock:
        p, c = _tok["prompt"], _tok["completion"]

    print(f"\n=== 统计（本次 {total_done} 条，累计 {cumulative}/{total_input}）===", flush=True)
    print(f"  无幻觉        : {stats['ok']} ({stats['ok'] / max(total_done, 1) * 100:.1f}%)", flush=True)
    print(f"  有幻觉        : {stats['halluc']} ({stats['halluc'] / max(total_done, 1) * 100:.1f}%)", flush=True)
    print(f"  判断失败(null): {stats['null']}", flush=True)
    print(f"  异常(未落盘)  : {stats['error']}", flush=True)
    print(f"\n=== Token 消耗 ===", flush=True)
    print(f"  prompt     : {p:,} ({p / 1e6:.3f}M)", flush=True)
    print(f"  completion : {c:,} ({c / 1e6:.3f}M)", flush=True)
    print(f"  输出路径   : {output_path}", flush=True)


if __name__ == "__main__":
    main()
