#!/usr/bin/env python3
"""
hallucination_repair_loop.py

一个自包含的集成式脚本：
- 输入：hallucination 检测后的 jsonl（通常是 *.halluc.jsonl）
- 对 hallucination == true 的样本做最多 5 轮：修复 -> 复检 -> 修复 -> ...
- 对 hallucination != true 的样本直接透传
- 只输出最终通过复检的样本；5 轮后仍未通过的样本不落盘
- 不依赖其他代码文件，内部包含检测与修复逻辑

输出约定：
- 保留输入记录里的所有键
- 将原始剧本移到 old_script
- 将最终修复剧本写到 script
- 新增 repair_round 表示第几轮通过（1~5）
- 新增 repair_error（通过时为空字符串）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Optional

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
# 幻觉检测 Prompt
# ──────────────────────────────────────────────────────────────────────────────
FILTER_PROMPT = """\
你是对话数据质检专家。你的任务是判断一段对话剧本中是否存在幻觉。

幻觉的唯一定义：
  用户提出了需要具体数据/内容才能回答的任务，但用户从未在对话中提供这些必要信息，
  而 Agent 却表现得像已经获得了这些信息并给出了答案。

核心判断标准：用户是否把任务所需的关键信息说清楚了？
如果关键信息缺失（题目没给、原文没给、案情没说），Agent 却在回答 → 幻觉。
如果用户信息充分（问了“北京几度”、给了原文、给了案情），Agent 给出具体回答 → 不算幻觉。

重要：不要以“Agent 能否访问外部数据源”来判断幻觉。
下游管线会为 Agent 注入实时数据（天气、搜索等），Agent 陈述具体事实结果不是幻觉。
幻觉只发生在：用户根本没把任务交代清楚，Agent 却在凭空编造条件。

典型幻觉案例：
- 用户说“帮我算这道题”，但没给题目 → 幻觉
- 用户说“帮我翻译这段话”，但没给原文 → 幻觉
- 用户说“帮我看看我发的那份文件”，但剧本里从未出现文件 → 幻觉
- 用户提到“上次说的那个事”，但剧本里没提过“那个事”是什么 → 幻觉
- 外呼场景：Agent 的任务是解释具体法律条款，但用户未提供任何案情信息，Agent 却断言“根据您的情况...” → 幻觉
- 外呼场景：Agent 的任务是推荐具体产品，但用户未提供需求，Agent 却直接列出了具体产品型号和价格 → 幻觉

不算幻觉的情况：
- 用户任务本身不需要具体内容（闲聊、续写故事、头脑风暴、解释概念） → 不算
- 用户给了完整信息，Agent 基于这些作答 → 不算
- Agent 追问用户补充信息 → 不算
- Agent 给出通用回答，没说“根据您提供的...” → 不算
- 用户问了有明确答案的问题（“北京几度”“什么是守护进程”），Agent 给出具体答案 → 不算（下游会补数据）
- Agent 基于用户已提供信息做了合理推论/常识补全，而非凭空编造 → 不算
- 创意生成类任务（“给几个发明创意”“帮我写首诗”“编个故事”），Agent 产出具体内容 → 不算
- Agent 声称做过核查/验证（“交叉验证了”“查过了”“核实过专利”），因为 Agent 具备联网搜索和工具调用能力 → 不算（除非剧本中用户明确质疑且 Agent 承认未核查）

请输出一个 JSON 对象：
{
  "hallucination": true 或 false,
  "reason": "简短说明判断依据（如无幻觉则输出空字符串\"\"）",
  "suggestion": "如果存在幻觉，给出具体的修改建议（如何补充 User 台词中缺失的信息，或如何修改 Agent 回答），如无幻觉则输出空字符串\"\""
}
只输出 JSON，不输出任何解释或 markdown。"""

# ──────────────────────────────────────────────────────────────────────────────
# 幻觉修复 Prompt
# ──────────────────────────────────────────────────────────────────────────────
REPAIR_PROMPT = """\
你是对话剧本修复专家。一段对话剧本中存在幻觉：用户在对话中没有提供关键信息，但 Agent 却给出了具体答案。

你的任务是：对剧本做最小修改，在 User 台词中补全缺失的信息，让 Agent 的回答变得合理。

【修复原则】
1. 只添加缺失信息到 User 台词中，不删改 User 已有的内容
2. Agent 台词原则上不动（因为修复的目标就是让 Agent 现有的回答变得有依据）
3. 如果 Agent 的回答需要轻微调整才合理，可以做最轻量的删改，但尽量不动
4. 新增的信息应自然嵌入对话，像一个真实用户会说的内容

【补全内容的真实性要求】
- 补全的信息必须是真实的、具体的文字内容。严禁使用占位符或元描述！
  ❌ 错误："[粘贴了销售数据]"
  ❌ 错误："[User pastes code showing counting logic]"
  ❌ 错误："这是题目：[题目内容省略]"
  ✅ 正确：直接生成一段具体的数据、代码或文章内容，像真人写出来的东西
- 题目、翻译原文、案情等也同理——写真实具体的内容，不要用"[此处省略]"等占位

【口语化要求 —— 重要！】
- 补全的信息必须是口语化陈述，像一个真人用户在对话框里打字说出来，千万不要以正式的代码块、矩阵排版、表格等书面格式呈现
- 数值信息用自然语言说出来：不说“1月500箱、2月620箱”，而说“1月卖了五百箱，2月卖了六百二十箱”
- 不要把大段格式化内容塞进一句 User 台词，可以自然地拆分到多句对话中
- 核心判断标准：把补全后的 User 台词读一遍，像不像人说的话？如果看着像复制粘贴的，就是不合格的
- 补全的 User 台词必须使用和原始剧本一致的语言
- 原始剧本是中文 → 补全内容用中文；原始剧本是英文 → 补全内容用英文
- 不要在中英文之间混用

【什么绝对不能动】
- 对话的轮次结构（User/Agent 交替顺序）
- 打断和垫话相关的控制行（如 [User 打断后]、[User 垫话后]、[当 Agent 刚刚说完 'xxx' 的时候打断/垫话] 等），以及紧接这类控制行的前一行对话，两者构成一个交互单元，不可修改
- User 已有的台词内容（只能往里加，不能删）
- 如果让写文章可以直接简单口述而不是说什么发邮箱了，当然如果修改意见时补充邮箱号也可以，编造一个真实的，而不是example。
- 绝对不要补充进长篇大论和代码，因为这是voice。

【可以调整的】
- 普通的 “[User 说完后]”、“[Agent 说完后]”、“[User 沉默了 X 秒后]” 等控制行，可以随对话内容的增减做相应插入或移除
- 剧本的整体节奏，但新增内容应保持对话自然流畅

【输出格式】
直接输出修复后的完整剧本，不要输出任何解释、不要加 markdown 代码块标记。"""

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI 客户端
# ──────────────────────────────────────────────────────────────────────────────
_tok_lock = threading.Lock()
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


def _call(client: OpenAI, messages, max_tokens: int) -> str:
    """单次调用，异常直接上抛。"""
    resp = client.chat.completions.create(
        model=MODEL_PATH,
        messages=messages,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}},
        timeout=1800,
    )
    if resp.usage:
        with _tok_lock:
            _tok["prompt"] += resp.usage.prompt_tokens
            _tok["completion"] += resp.usage.completion_tokens
            _tok["calls"] += 1
    raw = _extract_text(resp.choices[0])
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    if not raw:
        raise ValueError("API 返回空内容")
    return raw


def call_filter(client: OpenAI, user_content: str) -> dict:
    """单次调用，异常上抛。"""
    raw = _call(client, [
        {"role": "system", "content": FILTER_PROMPT},
        {"role": "user", "content": user_content},
    ], max_tokens=16193)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
    raise ValueError(f"无法解析 JSON: {raw[:200]}")


def call_repair(client: OpenAI, user_content: str) -> str:
    """单次调用，异常上抛。"""
    return _call(client, [
        {"role": "system", "content": REPAIR_PROMPT},
        {"role": "user", "content": user_content},
    ], max_tokens=16193)

# ──────────────────────────────────────────────────────────────────────────────
# 构造输入
# ──────────────────────────────────────────────────────────────────────────────
def build_filter_user_content(record: dict) -> str:
    prompt = record.get("prompt", "")
    script = record.get("script", "")
    orig = record.get("original_messages", [])
    orig_user = next((m.get("content", "") for m in orig if m.get("role") == "user"), "")
    orig_snippet = orig_user[:500] if orig_user else "（无）"
    task = record.get("writer_system", "") or record.get("original_system", "")
    task_snippet = task[:800] if task else ""
    if task_snippet:
        return f"【主题】{prompt}\n\n【Agent 任务描述】\n{task_snippet}\n\n【原始对话内容（参考）】\n{orig_snippet}\n\n【剧本】\n{script}"
    return f"【主题】{prompt}\n\n【原始对话内容（参考）】\n{orig_snippet}\n\n【剧本】\n{script}"


def build_repair_user_content(record: dict, reason: str, suggestion: str) -> str:
    script = record.get("script", "")
    return f"【幻觉原因】{reason}\n\n【修复建议】{suggestion}\n\n【原始剧本】\n{script}"

# ──────────────────────────────────────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────────────────────────────────────
def process_record(record: dict, client: OpenAI, max_rounds: int = 5) -> dict:
    rec = dict(record)
    original_script = rec.get("script", "")
    old_script = original_script

    current_script = original_script
    passed_round = None
    last_reason = rec.get("hallucination_reason", "")
    last_suggestion = rec.get("hallucination_suggestion", "")

    for round_idx in range(1, max_rounds + 1):
        # 1) 修复
        tmp_rec = dict(rec)
        tmp_rec["script"] = current_script
        repaired = call_repair(client, build_repair_user_content(tmp_rec, last_reason, last_suggestion))
        if not repaired:
            rec["old_script"] = old_script
            rec["script"] = current_script
            rec["repair_error"] = f"第{round_idx}轮修复失败"
            rec["repair_round"] = None
            return rec

        current_script = repaired

        # 2) 复检
        check_rec = dict(rec)
        check_rec["script"] = current_script
        recheck = call_filter(client, build_filter_user_content(check_rec))
        if recheck is None:
            rec["old_script"] = old_script
            rec["script"] = current_script
            rec["repair_error"] = f"第{round_idx}轮复检失败"
            rec["repair_round"] = None
            return rec

        if bool(recheck.get("hallucination", False)) is False:
            passed_round = round_idx
            rec["old_script"] = old_script
            rec["script"] = current_script
            rec["repair_error"] = ""
            rec["repair_round"] = passed_round
            return rec

        # 继续下一轮
        last_reason = recheck.get("reason", "")
        last_suggestion = recheck.get("suggestion", "")

    # 5 轮都没通过 -> 不落盘
    rec["old_script"] = old_script
    rec["script"] = current_script
    rec["repair_error"] = "5轮后仍未通过复检"
    rec["repair_round"] = None
    return rec

# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="集成式幻觉修复循环管线")
    parser.add_argument("--input", required=True, help="*.halluc.jsonl 输入")
    parser.add_argument("--output", default=None, help="输出 *.repaired.jsonl")
    parser.add_argument("--max-workers", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / (input_path.stem + ".repaired.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("hallucination_repair_loop 启动", flush=True)
    print(f"input={input_path}", flush=True)
    print(f"output={output_path}", flush=True)
    print(f"max_workers={args.max_workers}  max_rounds={args.max_rounds}", flush=True)

    # 断点续传：仅靠最终输出 sample_id
    done_ids: set[str] = set()
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

    # 读取输入
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

    keep_records = []
    need_repair = []
    for line in all_lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = rec.get("sample_id", "")
        if sid in done_ids:
            continue
        if rec.get("hallucination") is True:
            need_repair.append(rec)
        else:
            keep_records.append(rec)

    print(f"透传样本: {len(keep_records)}  待修复样本: {len(need_repair)}", flush=True)

    # 先落盘透传样本
    with open(output_path, "a", encoding="utf-8") as f:
        for rec in keep_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sid = rec.get("sample_id", "")
            if sid:
                done_ids.add(sid)

    if not need_repair:
        print("无待修复样本，退出。", flush=True)
        return

    stats = {"passed": 0, "failed": 0}
    write_lock = threading.Lock()
    stats_lock = threading.Lock()

    def write_record(rec: dict):
        with write_lock:
            if rec.get("repair_round") is not None:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 共享任务队列
    task_queue: Queue = Queue()
    for rec in need_repair:
        task_queue.put(rec)

    total_tasks = task_queue.qsize()
    result_queue: Queue = Queue()
    stop_event = threading.Event()

    def _is_rate_limit(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "429" in msg or "rate limit" in msg or "too many" in msg

    def provider_worker(client: OpenAI, worker_idx: int) -> None:
        while not stop_event.is_set():
            try:
                rec = task_queue.get(timeout=0.5)
            except Empty:
                break
            try:
                result = process_record(rec, client, args.max_rounds)
                result_queue.put(("done", result))
            except Exception as exc:
                if _is_rate_limit(exc):
                    task_queue.put(rec)
                    result_queue.put(("requeued", None))
                    time.sleep(1)
                else:
                    result_queue.put(("error", None))

    def provider_dispatcher(ep: str) -> None:
        client = OpenAI(api_key=API_KEY, base_url=ep, timeout=1800)
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

    pbar = tqdm(total=total_tasks, desc="幻觉修复循环", file=sys.stdout, smoothing=0)
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
            with stats_lock:
                if rec.get("repair_round") is not None:
                    stats["passed"] += 1
                else:
                    stats["failed"] += 1
        else:
            with stats_lock:
                stats["failed"] += 1

        pbar.set_postfix({"passed": stats['passed'], "failed": stats['failed']}, refresh=False)
        pbar.update(1)

    stop_event.set()

    print("\n=== 统计 ===", flush=True)
    print(f"  修复通过: {stats['passed']}", flush=True)
    print(f"  修复失败: {stats['failed']}", flush=True)
    with _tok_lock:
        p, c = _tok['prompt'], _tok['completion']
    print(f"  prompt tokens: {p:,}", flush=True)
    print(f"  completion tokens: {c:,}", flush=True)
    print(f"  输出路径: {output_path}", flush=True)

if __name__ == '__main__':
    main()
