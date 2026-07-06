#!/usr/bin/env python3
"""
clean_all_datasets.py

将 7 个数据集清洗成 video_pipeline_base.py 需要的输入格式。
每条原始记录通过一次 LLM 调用完成：
  - 非中/英文 → 翻译成中文
  - 单轮对话 → 续写成 2~5 轮
  - outbound → 生成 system prompt + 改首轮为 assistant 先说
  - inbound  → system 固定为"你是有用的助手"

支持 --dataset 指定单个数据集或 all，支持断点续传。
用 qwen35 env 运行：
  /path/to/conda/envs/qwen35/bin/python3 clean_all_datasets.py [args]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DATASETS_ROOT = Path(os.environ.get("DATASETS_ROOT", "datasets"))

DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_CONCURRENCY = 500
DEFAULT_RPM_LIMIT = 0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_SLEEP = 2.0
DEFAULT_RANDOM_SEED = 20260318

_THREAD_LOCAL = threading.local()
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# ---------------------------------------------------------------------------
# 数据集定义
# ---------------------------------------------------------------------------

DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "wildchat": {
        "dir": DATASETS_ROOT / "allenai/WildChat-4.8M",
        "out_stem": "train.voiceagent",
        "sample_rate": 1_000_000 / 3_199_860,
        "estimated_total": 500_000,  # 采样后各进程处理约50万
    },
    "mt_sft": {
        "dir": DATASETS_ROOT / "thomas-yanxin/MT-SFT-ShareGPT",
        "out_stem": "train.voiceagent",
        "sample_rate": 1_000_000 / 5_567_452,
        "estimated_total": 500_000,
    },
    "belle": {
        "dir": DATASETS_ROOT / "BelleGroup/multiturn_chat_0.8M",
        "out_stem": "train.voiceagent",
        "estimated_total": 415_000,  # 83万各一半
    },
    "coig": {
        "dir": DATASETS_ROOT / "BAAI/COIG",
        "out_stem": "train.voiceagent",
        "estimated_total": 89_000,
    },
    "coig_cqia": {
        "dir": DATASETS_ROOT / "m-a-p/COIG-CQIA",
        "out_stem": "train.voiceagent",
        "estimated_total": 22_000,
    },
    "oasst2": {
        "dir": DATASETS_ROOT / "OpenAssistant/oasst2",
        "out_stem": "train.voiceagent",
        "estimated_total": 7_000,
    },
    "no_robots": {
        "dir": DATASETS_ROOT / "HuggingFaceH4/no_robots",
        "out_stem": "train.voiceagent",
        "estimated_total": 9_500,
    },
}

ALL_DATASETS = list(DATASET_SPECS.keys())

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
你是一个对话数据清洗助手。你的任务是把给定的原始对话素材转换成符合要求的训练数据。

你必须严格遵守：

1. 输出必须是合法 JSON，只输出 JSON 本体，不要有任何解释文字。
2. 自行判断原始内容的语言：如果不是中文也不是英文，必须把 messages 里每一条 content 都翻译成中文，包括 user 和 assistant 的每一轮；中文和英文保持原样，不要翻译。
3. 语言一致性：续写或扩写轮次时，必须使用与原始对话相同的语言。原文是中文就续写中文，原文是英文就续写英文，不要混用。
4. 保持原始表达风格。
5. role 只能是 system、user、assistant。
6. messages 列表中第一条必须是 system。
7. inbound 样本：system 内容固定为"你是有用的助手"，第一条非 system 消息必须是 user。
8. outbound 样本：system 内容必须是根据对话主题生成的"你是xxx，你的任务是xxx"格式，\
可以根据任务复杂程度写长或写短；第一条非 system 消息必须是 assistant 主动发起，且必须先说明自己的来意和任务，不能直接进入内容。
9. 自行判断原始对话是否只有一轮（一问一答）：如果是，续写成 {target_turns} 轮，\
保持主题一致，自然推进；如果已经是多轮，保持原有轮数不变。
10. 保留原始对话的核心主题、关键事实和解决路径。
"""

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def strip_think(text: str) -> str:
    return re.sub(_THINK_TAG_RE, "", text).strip()


def sanitize_unicode(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8")


def make_prompt_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = strip_think(text).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None




# ---------------------------------------------------------------------------
# API 客户端
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            elapsed = time.time() - self.last
            wait = self.interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()


def get_client(api_base: str, api_key: str) -> Any:
    key = (api_base, api_key)
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None or getattr(_THREAD_LOCAL, "client_key", None) != key:
        from openai import OpenAI
        _THREAD_LOCAL.client = OpenAI(base_url=api_base, api_key=api_key)
        _THREAD_LOCAL.client_key = key
    return _THREAD_LOCAL.client


def call_model(
    api_base: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
) -> Tuple[str, Dict[str, int]]:
    client = get_client(api_base, api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=16384,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    usage = getattr(resp, "usage", None)
    stats = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    return resp.choices[0].message.content or "", stats


# ---------------------------------------------------------------------------
# Reader：各数据集 → raw_item = {"source_id": str, "raw_text": str}
# ---------------------------------------------------------------------------

# --- WildChat ---

def _iter_wildchat(dataset_dir: Path) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq
    parquet_files = sorted((dataset_dir / "data").glob("*.parquet"))
    for pf in parquet_files:
        table = pq.read_table(pf)
        for i in range(table.num_rows):
            row = {c: table.column(c)[i].as_py() for c in table.schema.names}
            if row.get("toxic"):
                continue
            conv = row.get("conversation") or []
            if not conv:
                continue
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in conv
                if m.get("role") in {"user", "assistant"} and m.get("content")
            ]
            if len(messages) < 2:
                continue
            raw_text = json.dumps(messages, ensure_ascii=False)
            yield {
                "source_id": f"wildchat:{row.get('conversation_hash', i)}",
                "raw_text": raw_text,
            }


# --- MT-SFT-ShareGPT ---

def _iter_mt_sft(dataset_dir: Path) -> Iterator[Dict[str, Any]]:
    for subdir in ["zh", "en", "others"]:
        for jsonl_path in sorted((dataset_dir / subdir).glob("*.jsonl")):
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    convs = row.get("conversations") or []
                    if not convs:
                        continue
                    raw_text = json.dumps(convs, ensure_ascii=False)
                    yield {
                        "source_id": f"mt_sft:{row.get('id', hashlib.md5(raw_text.encode()).hexdigest())}",
                        "raw_text": raw_text,
                    }


# --- BelleGroup ---

def _iter_belle(dataset_dir: Path) -> Iterator[Dict[str, Any]]:
    json_path = dataset_dir / "multiturn_chat_0.8M.json"
    with open(json_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            instruction = row.get("instruction") or ""
            if not instruction:
                continue
            yield {
                "source_id": f"belle:{idx}",
                "raw_text": instruction,
            }


# --- BAAI/COIG ---

def _iter_coig(dataset_dir: Path) -> Iterator[Dict[str, Any]]:
    # exam_instructions.jsonl
    exam_path = dataset_dir / "exam_instructions.jsonl"
    if exam_path.exists():
        with open(exam_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_text = json.dumps(row, ensure_ascii=False)
                yield {
                    "source_id": f"coig_exam:{idx}",
                    "raw_text": raw_text,
                }

    # leetcode_instructions.jsonl - 过滤纯代码 output
    leetcode_path = dataset_dir / "leetcode_instructions.jsonl"
    if leetcode_path.exists():
        with open(leetcode_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_text = json.dumps(row, ensure_ascii=False)
                yield {
                    "source_id": f"coig_leetcode:{idx}",
                    "raw_text": raw_text,
                }

    # translated_instructions.jsonl
    for fname in ["translated_instructions.jsonl",
                  "human_value_alignment_instructions_part1.json",
                  "human_value_alignment_instructions_part2.json"]:
        fpath = dataset_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            raw = f.read().strip()
        try:
            records = json.loads(raw)
            if not isinstance(records, list):
                records = [records]
        except json.JSONDecodeError:
            records = []
            for l in raw.splitlines():
                l = l.strip()
                if not l:
                    continue
                try:
                    records.append(json.loads(l))
                except json.JSONDecodeError:
                    continue
        for idx, row in enumerate(records):
            raw_text = json.dumps(row, ensure_ascii=False)
            yield {
                "source_id": f"coig_{fname}:{idx}",
                "raw_text": raw_text,
            }

    # counterfactual multi-round chat (tar.gz)
    tar_path = dataset_dir / "counterfactural_correction_multi_round_chat.tar.gz"
    if tar_path.exists():
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".json"):
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                try:
                    row = json.load(fobj)
                except json.JSONDecodeError:
                    continue
                raw_text = json.dumps(row, ensure_ascii=False)
                yield {
                    "source_id": f"coig_counterfactual:{member.name}",
                    "raw_text": raw_text,
                }


# --- COIG-CQIA ---

def _iter_coig_cqia(dataset_dir: Path) -> Iterator[Dict[str, Any]]:
    # 用 COIG-CQIA-full.jsonl（已合并），避免重复遍历子目录
    full_path = dataset_dir / "COIG-CQIA-full.jsonl"
    if full_path.exists():
        with open(full_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_text = json.dumps(
                    {"instruction": row.get("instruction", ""),
                     "output": row.get("output", "")},
                    ensure_ascii=False,
                )
                yield {
                    "source_id": f"coig_cqia:{idx}",
                    "raw_text": raw_text,
                }
        return

    # fallback：遍历子目录
    for jsonl_path in sorted(dataset_dir.rglob("*.jsonl")):
        if jsonl_path.name.startswith("."):
            continue
        with open(jsonl_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_text = json.dumps(
                    {"instruction": row.get("instruction", ""),
                     "output": row.get("output", "")},
                    ensure_ascii=False,
                )
                yield {
                    "source_id": f"coig_cqia:{jsonl_path.stem}:{idx}",
                    "raw_text": raw_text,
                }


# --- OpenAssistant/oasst2 ---

def _extract_best_path(node: Dict[str, Any]) -> List[Dict[str, str]]:
    """递归取 rank=0（或 rank=None）的最优回复路径。"""
    role_map = {"prompter": "user", "assistant": "assistant"}
    role = role_map.get(node.get("role", ""), None)
    text = (node.get("text") or "").strip()
    messages = []
    if role and text:
        messages.append({"role": role, "content": text})
    replies = node.get("replies") or []
    if replies:
        best = min(replies, key=lambda r: (r.get("rank") or 99))
        messages.extend(_extract_best_path(best))
    return messages


def _iter_oasst2(dataset_dir: Path) -> Iterator[Dict[str, Any]]:
    trees_gz = dataset_dir / "2023-11-05_oasst2_ready.trees.jsonl.gz"
    with gzip.open(trees_gz) as f:
        for idx, line in enumerate(f):
            try:
                tree = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt_node = tree.get("prompt") or {}
            messages = _extract_best_path(prompt_node)
            if len(messages) < 2:
                continue
            raw_text = json.dumps(messages, ensure_ascii=False)
            yield {
                "source_id": f"oasst2:{tree.get('message_tree_id', idx)}",
                "raw_text": raw_text,
            }


# --- HuggingFaceH4/no_robots ---

def _iter_no_robots(dataset_dir: Path) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq
    for pf in sorted((dataset_dir / "data").glob("train*.parquet")):
        table = pq.read_table(pf)
        for i in range(table.num_rows):
            row = {c: table.column(c)[i].as_py() for c in table.schema.names}
            messages = row.get("messages") or []
            if not messages:
                continue
            prompt_id = row.get("prompt_id") or make_prompt_id(json.dumps(messages))
            raw_text = json.dumps(messages, ensure_ascii=False)
            yield {
                "source_id": f"no_robots:{prompt_id}",
                "raw_text": raw_text,
            }


READERS = {
    "wildchat": _iter_wildchat,
    "mt_sft": _iter_mt_sft,
    "belle": _iter_belle,
    "coig": _iter_coig,
    "coig_cqia": _iter_coig_cqia,
    "oasst2": _iter_oasst2,
    "no_robots": _iter_no_robots,
}

# ---------------------------------------------------------------------------
# 构造 prompt & 验证输出
# ---------------------------------------------------------------------------

def build_user_prompt(
    raw_item: Dict[str, Any],
    call_type: str,
    target_turns: int,
) -> str:
    if call_type == "outbound":
        scene = (
            f"本条样本是外呼场景：\n"
            f'- system 内容必须是根据对话主题生成的"你是xxx，你的任务是xxx"格式，任务复杂可写详细。\n'
            f"- 第一条非 system 消息必须是 assistant 主动发起对话。"
        )
    else:
        scene = (
            f"本条样本是内呼场景：\n"
            f'- system 内容固定为"你是有用的助手"。\n'
            f"- 第一条非 system 消息必须是 user。"
        )

    return f"""请把下面的原始对话素材转换成训练数据。

[语言规则]
- 判断原始素材的主要语言。如果是中文或英文，所有 messages 保持该语言，续写也用同一语言。
- 如果是其他语言（如俄语、西班牙语、法语等），把每一条 message 的 content 全部翻译成中文，续写也用中文。
- 严禁在同一条 messages 列表里混用多种语言。

[场景要求]
{scene}

[单轮续写]
如果原始对话只有一轮（一问一答），续写成 {target_turns} 轮，保持主题一致，自然推进；如果已经是多轮，保持原有轮数。

[输出格式]
{{
  "call_type": "{call_type}",
  "prompt": "一句话概括对话主题",
  "messages": [
    {{"role": "system", "content": "..."}},
    {{"role": "user"/"assistant", "content": "..."}}
  ]
}}

[原始素材]
{raw_item['raw_text']}
"""


def validate_output(
    payload: Dict[str, Any],
    call_type: str,
) -> Tuple[str, str, List[Dict[str, str]]]:
    if not isinstance(payload, dict):
        raise ValueError("输出不是 JSON 对象")

    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("缺少 prompt 字段")

    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("messages 不足 3 条")

    normalized: List[Dict[str, str]] = []
    for idx, m in enumerate(messages):
        role = m.get("role")
        content = str(m.get("content") or "").strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{idx}] role 非法: {role!r}")
        if not content:
            raise ValueError(f"messages[{idx}] content 为空")
        normalized.append({"role": role, "content": content})

    if normalized[0]["role"] != "system":
        raise ValueError("第一条不是 system")

    if call_type == "outbound":
        if normalized[1]["role"] != "assistant":
            raise ValueError("outbound 首轮非 assistant")
    else:
        if not normalized[0]["content"].startswith("你是有用的助手"):
            raise ValueError("inbound system 不是'你是有用的助手'")
        if normalized[1]["role"] != "user":
            raise ValueError("inbound 首轮非 user")

    # 清除代理字符，防止 UnicodeEncodeError
    for m in normalized:
        m["content"] = sanitize_unicode(m["content"])
    prompt = sanitize_unicode(prompt)

    return prompt, make_prompt_id(prompt), normalized


# ---------------------------------------------------------------------------
# 处理单条样本
# ---------------------------------------------------------------------------

def process_one(
    raw_item: Dict[str, Any],
    call_type: str,
    target_turns: int,
    api_base: str,
    api_key: str,
    model: str,
    max_retries: int,
    retry_sleep: float,
    rate_limiter: RateLimiter,
) -> Dict[str, Any]:
    source_id = raw_item["source_id"]
    system = SYSTEM_PROMPT.format(target_turns=target_turns)
    user = build_user_prompt(raw_item, call_type, target_turns)
    last_err = None
    total_stats: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for attempt in range(1, max_retries + 1):
        try:
            rate_limiter.wait()
            raw_resp, stats = call_model(api_base, api_key, model, system, user)
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)
            payload = extract_json(raw_resp)
            if payload is None:
                raise ValueError("LLM 输出无法解析为 JSON")
            prompt_text, prompt_id, norm_messages = validate_output(payload, call_type)
            return {
                "ok": True,
                "source_id": source_id,
                "call_type": call_type,
                "output_record": {
                    "prompt": prompt_text,
                    "prompt_id": prompt_id,
                    "messages": norm_messages,
                },
                **total_stats,
            }
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_sleep)

    return {
        "ok": False,
        "source_id": source_id,
        "call_type": call_type,
        "error": str(last_err),
        "raw_text": raw_item["raw_text"],
        **total_stats,
    }


# ---------------------------------------------------------------------------
# 输出路径
# ---------------------------------------------------------------------------

def build_output_paths(dataset_dir: Path, stem: str) -> Dict[str, Path]:
    return {
        "inbound": dataset_dir / f"{stem}.inbound.jsonl",
        "outbound": dataset_dir / f"{stem}.outbound.jsonl",
        "inbound_dropped": dataset_dir / f"{stem}.inbound.dropped.jsonl",
        "outbound_dropped": dataset_dir / f"{stem}.outbound.dropped.jsonl",
    }


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ---------------------------------------------------------------------------
# 主处理函数
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_name: str,
    split: str,
    api_base: str,
    api_key: str,
    model: str,
    concurrency: int,
    rpm_limit: int,
    max_retries: int,
    retry_sleep: float,
    random_seed: int,
    resume: bool,
    limit: Optional[int],
    debug: bool,
) -> None:
    spec = DATASET_SPECS[dataset_name]
    dataset_dir: Path = spec["dir"]
    stem: str = spec["out_stem"]
    paths = build_output_paths(dataset_dir, stem)

    # 统计已完成数量（inbound + outbound）
    resumed_inbound = count_lines(paths["inbound"]) + count_lines(paths["inbound_dropped"]) if resume else 0
    resumed_outbound = count_lines(paths["outbound"]) + count_lines(paths["outbound_dropped"]) if resume else 0
    resumed_total = resumed_inbound + resumed_outbound

    reader_fn = READERS[dataset_name]
    sample_rate: Optional[float] = spec.get("sample_rate")
    rng = random.Random(random_seed)

    # 先过一遍数据生成 call_type 计划（确定 inbound/outbound 分配）
    # 为避免全量预读大数据，用 source_id hash 决定 call_type
    def decide_call_type(source_id: str) -> Optional[str]:
        h = int(hashlib.md5(source_id.encode()).hexdigest(), 16) % 2
        is_inbound = (h == 0)
        if split == "both":
            return "inbound" if is_inbound else "outbound"
        if split == "inbound":
            return "inbound" if is_inbound else None
        if split == "outbound":
            return "outbound" if not is_inbound else None
        return None

    rate_limiter = RateLimiter(rpm_limit)

    open_mode_success = {
        "inbound": "a" if resume and paths["inbound"].exists() else "w",
        "outbound": "a" if resume and paths["outbound"].exists() else "w",
    }
    open_mode_dropped = {
        "inbound": "a" if resume and paths["inbound_dropped"].exists() else "w",
        "outbound": "a" if resume and paths["outbound_dropped"].exists() else "w",
    }

    # 已完成的 source_id 集合（全量读入内存后批量处理）
    _sid_re = re.compile(r'"source_id"\s*:\s*"([^"]+)"')
    _pid_re = re.compile(r'"prompt_id"\s*:\s*"([^"]+)"')
    done_ids: set = set()
    if resume:
        for p in [paths["inbound"], paths["outbound"],
                  paths["inbound_dropped"], paths["outbound_dropped"]]:
            if p.exists():
                raw = p.read_bytes().decode("utf-8", errors="replace")
                for m in _sid_re.finditer(raw):
                    done_ids.add(m.group(1))
                if not done_ids:
                    for m in _pid_re.finditer(raw):
                        done_ids.add(m.group(1))

    estimated_total: int = spec.get("estimated_total", 0)
    print(f"[{dataset_name}] 已完成 {len(done_ids)} 条，预估总量 {estimated_total}，开始处理...", flush=True)

    success_count = 0
    error_count = 0
    total_tokens = 0
    processed = 0
    start_time = time.time()

    with (
        open(paths["inbound"], open_mode_success["inbound"], encoding="utf-8") as inbound_f,
        open(paths["outbound"], open_mode_success["outbound"], encoding="utf-8") as outbound_f,
        open(paths["inbound_dropped"], open_mode_dropped["inbound"], encoding="utf-8") as inbound_drop_f,
        open(paths["outbound_dropped"], open_mode_dropped["outbound"], encoding="utf-8") as outbound_drop_f,
    ):
        out_files = {"inbound": inbound_f, "outbound": outbound_f}
        drop_files = {"inbound": inbound_drop_f, "outbound": outbound_drop_f}

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures: Dict[concurrent.futures.Future, str] = {}

            def flush_done() -> None:
                nonlocal success_count, error_count, total_tokens, processed
                done_futs = [f for f in list(futures) if f.done()]
                for fut in done_futs:
                    ct = futures.pop(fut)
                    try:
                        result = fut.result()
                    except Exception as e:
                        error_count += 1
                        processed += 1
                        if debug:
                            print(f"[{dataset_name}] future error: {e}", file=sys.stderr)
                        continue
                    processed += 1
                    total_tokens += result.get("total_tokens", 0)
                    if result["ok"]:
                        success_count += 1
                        rec = result["output_record"]
                        rec["source_id"] = result["source_id"]
                        out_files[ct].write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_files[ct].flush()
                    else:
                        error_count += 1
                        drop_rec = {
                            "source_id": result["source_id"],
                            "call_type": ct,
                            "error": result.get("error"),
                            "raw_text": result.get("raw_text", ""),
                        }
                        drop_files[ct].write(json.dumps(drop_rec, ensure_ascii=False) + "\n")
                        drop_files[ct].flush()

                if processed % 100 == 0 and processed > 0:
                    elapsed = time.time() - start_time
                    speed = processed / elapsed
                    avg_tok = total_tokens / max(processed, 1)
                    done_so_far = len(done_ids) + processed
                    remaining = max(estimated_total - done_so_far, 0) if estimated_total else 0
                    eta_sec = remaining / speed if speed > 0 and remaining > 0 else 0
                    eta_str = f"{int(eta_sec//3600)}h{int((eta_sec%3600)//60)}m" if eta_sec > 0 else "unknown"
                    pct = f"{100*done_so_far/estimated_total:.1f}%" if estimated_total else "?%"
                    ts = time.strftime("%H:%M:%S")
                    print(
                        f"[{ts}][{dataset_name}] "
                        f"processed={processed}({pct}) "
                        f"ok={success_count} err={error_count} "
                        f"speed={speed:.1f}/s avg_tok={avg_tok:.0f} "
                        f"eta={eta_str}",
                        flush=True,
                    )

            item_count = 0
            for raw_item in reader_fn(dataset_dir):
                if raw_item["source_id"] in done_ids:
                    continue
                if limit is not None and item_count >= limit:
                    break
                # 基于 source_id hash 的确定性随机采样，保证可复现且断点续传安全
                if sample_rate is not None:
                    h = int(hashlib.md5(raw_item["source_id"].encode()).hexdigest()[-8:], 16)
                    if h / 0xFFFFFFFF >= sample_rate:
                        continue
                call_type = decide_call_type(raw_item["source_id"])
                if call_type is None:
                    continue
                item_count += 1

                target_turns = rng.randint(2, 5)

                fut = executor.submit(
                    process_one,
                    raw_item,
                    call_type,
                    target_turns,
                    api_base,
                    api_key,
                    model,
                    max_retries,
                    retry_sleep,
                    rate_limiter,
                )
                futures[fut] = call_type

                # 控制 in-flight 数量
                while len(futures) >= concurrency:
                    concurrent.futures.wait(
                        futures,
                        timeout=0.5,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    flush_done()

            # 等待剩余
            concurrent.futures.wait(futures)
            flush_done()

    elapsed = time.time() - start_time
    avg_tok = total_tokens / max(processed, 1)
    speed = processed / elapsed if elapsed > 0 else 0
    print(
        f"\n[{time.strftime('%H:%M:%S')}][{dataset_name}] 完成："
        f"total={processed} ok={success_count} err={error_count} "
        f"avg_tok={avg_tok:.0f} speed={speed:.1f}/s "
        f"elapsed={int(elapsed//3600)}h{int((elapsed%3600)//60)}m",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗 7 个数据集为 VoiceAgent 训练格式")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=ALL_DATASETS + ["all"],
        help="指定处理哪个数据集，默认 all",
    )
    parser.add_argument(
        "--split",
        default="both",
        choices=["inbound", "outbound", "both"],
        help="只处理 inbound 或 outbound，默认 both",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--rpm-limit", type=int, default=DEFAULT_RPM_LIMIT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-sleep", type=float, default=DEFAULT_RETRY_SLEEP)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--limit", type=int, default=None, help="每个数据集最多处理 N 条")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    for dataset_name in targets:
        print(f"\n{'=' * 60}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'=' * 60}")
        process_dataset(
            dataset_name=dataset_name,
            split=args.split,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            concurrency=args.concurrency,
            rpm_limit=args.rpm_limit,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            random_seed=args.random_seed,
            resume=args.resume,
            limit=args.limit,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
