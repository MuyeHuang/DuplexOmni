"""
unified_pipeline.py
功能:
1. 用 content + 随机采样 seed 的方式生成编剧剧本
2. 将编剧剧本送入导演标注，产出最终训练数据
3. 按 inbound / outbound 分开运行，并支持断点续传与分段调试
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

if TYPE_CHECKING:
    from .api_client import APICaller


WRITER_SYSTEM_PROMPT = """
你是一个专业的话剧编剧。你的任务是将给定的内容主题和约束，改编成一段【纯文本格式】的对话剧本。
剧本由【User】和【Agent】两个角色出演。User 应该口语化。Agent 文本应该可读性强，适合口语播报。数字优先写成中文念法；但英文品牌名、字母数字型号、容易口播的英文短词不必强行翻成中文，例如 RSQ、R8 Spyder、TTS、AI 都可以直接保留。

### 核心规则
1. 纯文本输出：不要输出 JSON，不要输出 Markdown 代码块，直接输出剧本正文。
2. 完整文本原则：即使角色被打断，也必须写出他原本打算说完的完整句子。不要用 "..." 或 "——" 省略，不要直接截断文本。截断是导演阶段的事，编剧必须写全。
3. 状态前置：每一句台词前，必须用 `[...]` 标注这句话的触发时机。
4. 角色一致性：角色的人设、说话习惯、任务目标必须前后一致，不要突然换人设。
5. 首轮发言必须符合任务要求：外呼剧本第一句必须由 Agent 发起，内呼剧本第一句必须由 User 发起。
6. 只有用户可以出现沉默节奏。需要表示双方都沉默、然后由用户恢复说话时，请在用户台词前使用类似 `[User沉默了1秒后]`、`[User沉默了2秒后]` 的状态描述；不要写 `Agent沉默了...`，因为 Agent 默认实时响应。

### 格式范例
[对话开始]
User: "我想问一下这个功能怎么开。"
[User 说完后]
Agent: "可以，我先帮你确认一下你的需求。"
[紧接着]
Agent: "如果你说的是集合页里的图片悬停效果，那我可以一步一步告诉你在哪里设置。"
[当 Agent 刚刚说完 "一步一步告诉你" 的时候打断]
User: "你直接说入口在哪。"
[User 打断后]
Agent: "好，我直接说入口。你登录后台以后，先进入在线商店，再打开主题自定义页面。"

### 触发时机标签
- 正常接话：使用 `[User/Agent 说完后]`
- 正常分段说话：使用 `[紧接着]`
- 抢话/打断：使用 `[当 Agent 刚刚说完 'XX' 的时候打断]`
- 垫话但不打断：使用 `[当 Agent 刚刚说完 'XX' 的时候垫话]`
- 用户沉默后恢复：使用 `[User沉默了X秒后]`，其中 `X` 是正整数秒数，只能用于用户轮次前。

### 绝对禁止
- 上一句完整说完却假装被打断
- 助手主动打断用户
- 明知条件变了却沿用旧结论
- 用户改口后仍然无视新条件
- 提前知道用户性格、未来事件或 S2 策略并直接说出来
- 把容易口播的英文型号硬翻成非常生硬的中文念法
"""


DIRECTOR_SYSTEM_PROMPT = """
你是一个全双工语音助手的数据标注专家（导演）。
你的任务是读取一段自然语言剧本，并将其转换为带有系统控制标签的训练数据。

### 输入格式
一段纯文本剧本，包含 `[触发时机描述] Role: "台词"`。

### 输出格式
一个纯 JSON 列表，包含 `role` 和 `text`。
在 `text` 中必须正确插入以下标签：

### 标签定义与插入规则 (至关重要)

1. **[THINK] (启动思考)**
   - **含义**: 用户提出了复杂问题，Agent 需要开始调用 S2 系统进行推理。
   - **位置**: 插入在用户提出问题的句子末尾。

2. **「...」 (系统2消息注入)**
   - **含义**: S2 思考出的中间步骤或结果。Agent 说出的每一句包含具体事实、推理、查询结果或可执行建议的话，都必须先收到 S2 的条子。
   - **位置**: 插入在 Agent 播报该事实之前。注意**外呼**的第一条assistant不要带S2消息，因为没有THINK去触发，而且assistant本身就是带着任务来的，任务在system里已经交代完毕了。
   - **时序**: 如果上一句是 User，则下一句 Agent 绝对不能立刻在开头就拿到 S2 条子。必须先有一句不带 `「...」` 的自然承接、安抚、确认、圆场或短垫话，然后后续 Agent 句子才能收到 S2 条子。
   - **强约束**: `User -> Assistant(以「开头)` 是错误格式。遇到这种情况，你必须把 Assistant 先拆成一句无 S2 的承接短句，再在下一句 Assistant 中放入 `「...」`。
   - **完整性**: `「...」` 不是标题，不是关键词，也不是提纲占位符。它必须包含足够完整、准确的事实上文，至少要把外面那句 Agent 正文中即将说出的关键信息、数字、条件、对象、结论交代完整。
   - **因果约束**: Agent 正文只能复述、转述、组织、展开已经在 `「...」` 中到达的信息，不能在正文里新增 `「...」` 里没有出现的关键事实。否则就等于 Agent 在编造。
   - **高频短S2规则**: 即使是高频、短促、多次到达的 S2，小条子本身也必须是完整信息片段，而不是残缺提示。例如不能写 `「拉瓜迪亚机场暂停接收」` 然后正文扩写成“因无人指挥而暂停接收 inbound 航班”；S2 必须先完整到达这层信息后，正文才能说出来。


3. **^ (说话触发点)**
   - **含义**: 下一条消息开始说话的时间点。
   - **位置**: 根据剧本中的触发时机，在 Agent 被打断的那句话中找到对应位置。
   - **限制**: 禁止 Agent 打断 User；Agent 也不能打断自己。

4. **[CUT] (物理截断点)**
   - **含义**: Agent 声音停止的位置。但不是文字停止的位置，[CUT] 后面还得有“准备说但没说出口的文字”，不然就不算是打断啊。
   - **位置**: 必须紧跟在 `^` 后面，间隔 2 到 4 个字，模拟很短的反应延迟。**严禁出现 ^ [CUT] 连在一起这种写法**。
   - **限制**: 如果只是垫话，不要加 `[CUT]`。如果 `^` 后面已经几乎到句末，说明这句话其实自然结束了，此时严禁在句子末尾加 `[CUT]`。

5. **[WAIT] (思考挂起)**
   - **含义**: 用户打断了 Agent，导致当前的 S2 思考必须暂停或重置。
   - **位置**: 插入在打断 Agent 的那句 User 台词末尾。
   - **限制**: 如果输出了 `[WAIT]` 后还需要继续推理，必须重新输出 `[THINK]`。同样的，如果[WAIT]后还想接收S2消息，必须有[THINK]去触发S2。同理，如果用户无意添加新条件，或者挂起对话，没必要触发[WAIT]，直接接下一句，等待S2返回即可。

6. **[PENDXS] (空白标)**
   - **含义**: 用户和助手都暂时不说话，发生了一段共同沉默，之后由用户恢复说话。
   - **位置**: 只能插在 User 消息中，通常放在该条 User 文本开头，例如 `[PEND1S]嗯，我想一下。`
   - **来源**: 当剧本里出现 `[User沉默了1秒后]`、`[User沉默了2秒后]` 这类状态时，转换为对应的 `[PEND1S]`、`[PEND2S]`。
   - **限制**: 绝对不能用于 Agent 消息；也不要凭空给 Agent 加沉默。

### 注意事项
1. Agent 的连续发言不要强行合并，导演需要为 S2 条子留出生效空间。
2. 即使 `[CUT]` 后面的字没有播完，也要保留完整幽灵文本。
3. 你需要根据 Agent 的回答内容，反向生成合理的 S2 思考内容。
4. 输出时不要包含 `system`，只输出剧本对应的 User / Agent 消息列表。
5. 容易口播的英文品牌、型号、缩写不要硬转成非常别扭的中文音译。
6. 外面那句 Agent 正文可以概括或者总结 `「...」` 里的信息，但不要编造，并且不要只是照着念，根据对话情况可以简练或扩增非核心内容。
7. `^` 和 `[CUT]` 的距离不能太远，通常只留 2 到 4 个字符；如果你要表达的是重叠说话而不是物理截断，就只放 `^` 不放 `[CUT]`。
8. 内呼（agent）接听电话的对话中，system中放的是agent人设，严禁存放对话背景信息，或者user query。

### 负面例子
错误：
- User: "我想问 RSQ 有什么功能？[THINK]"
- Assistant: "「RSQ 是一辆概念车」它是一辆概念车..."

原因：上一句是 User，Assistant 紧接着就以 `「...」` 开头，S2 到达时序错误。

正确：
- User: "我想问 RSQ 有什么功能？[THINK]"
- Assistant: "这个我先帮你捋一下。"
- Assistant: "「RSQ 是一辆概念车」它是一辆概念车..."

错误：
- Assistant: "「拉瓜迪亚机场暂停接收」最直接的例子就是一月二十五日拉瓜迪亚机场因无人指挥而暂停接收 inbound 航班。"

原因：`「...」` 只给了一个残缺标题，正文却新增了“时间、原因、对象、航班类型”等完整事实，违反因果。并且不要真的只放一个「...」，这里面要写内容的。

正确：
- Assistant: "「一月二十五日拉瓜迪亚机场因塔台无人指挥而暂停接收 inbound 航班」最直接的例子就是一月二十五日，拉瓜迪亚机场因塔台无人指挥而暂停接收 inbound 航班。"
"""


DEFAULT_API_KEY = "EMPTY"
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "gemini-3-flash-preview"

DEFAULT_SEED_PATH = (
    "data/seed/samples.cleaned.jsonl"
)
DEFAULT_INBOUND_PATH = (
    "data/content/train_sft.voiceagent.inbound.jsonl"
)
DEFAULT_OUTBOUND_PATH = (
    "data/content/train_sft.voiceagent.outbound.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/pipeline"
)

DEFAULT_SAVE_INTERVAL = 5
DEFAULT_MAX_WORKERS = 5
DEFAULT_RPM_LIMIT = 10
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY_BASE = 2
DEFAULT_SEED = 42
DEFAULT_WRITER_SEED_SAMPLE_SIZE = 3


@dataclass
class PipelineConfig:
    api_key: str
    api_base: str
    model_name: str
    seed_path: str
    inbound_path: str
    outbound_path: str
    output_dir: str
    mode: str
    split: str
    debug_limit: Optional[int]
    save_interval: int
    max_workers: int
    rpm_limit: int
    max_retries: int
    retry_delay_base: int
    random_seed: int
    writer_seed_sample_size: int


class RateLimiter:
    """线程安全的限流器，确保全局请求速率不超过设定值。"""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.last_call_time = 0.0
        self.lock = threading.Lock()

    def wait_for_token(self) -> None:
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            wait_time = self.interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_call_time = time.time()


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="统一编剧/导演流水线")
    parser.add_argument(
        "--mode",
        choices=["writer_only", "director_only", "full_pipeline"],
        default="full_pipeline",
        help="运行模式",
    )
    parser.add_argument(
        "--split",
        choices=["inbound", "outbound", "both"],
        default="both",
        help="处理数据 split",
    )
    parser.add_argument("--debug-limit", type=int, default=None, help="仅处理前 N 条待执行样本")
    parser.add_argument(
        "--writer-seed-sample-size",
        type=int,
        default=DEFAULT_WRITER_SEED_SAMPLE_SIZE,
        help="每条 content 展开生成多少条 (content, seed) 样本",
    )
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--rpm-limit", type=int, default=DEFAULT_RPM_LIMIT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay-base", type=int, default=DEFAULT_RETRY_DELAY_BASE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed-path", default=DEFAULT_SEED_PATH)
    parser.add_argument("--inbound-path", default=DEFAULT_INBOUND_PATH)
    parser.add_argument("--outbound-path", default=DEFAULT_OUTBOUND_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    return PipelineConfig(
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


def create_api_caller(config: PipelineConfig) -> "APICaller":
    try:
        from .api_client import APICaller as Caller
    except ImportError:
        try:
            from api_client import APICaller as Caller
        except ImportError as exc:
            raise RuntimeError(
                "无法导入 APICaller。请确认从脚本目录运行，且已安装 openai 依赖。"
            ) from exc
    return Caller(config.api_key, config.api_base, config.model_name)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        return []

    if Path(path).suffix == ".json":
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise ValueError(f"{path} 不是 JSON 列表")
        return data

    records: List[Dict[str, Any]] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def save_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        if output_path.suffix == ".json":
            json.dump(list(records), f, ensure_ascii=False, indent=2)
        else:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_path)


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def local_rng(base_seed: int, key: str) -> random.Random:
    return random.Random(base_seed + stable_hash(key) % (10**12))


def ordered_unique(items: Iterable[Any]) -> List[Any]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def choose_representative(values: Sequence[Optional[str]]) -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    best_value = cleaned[0]
    best_count = counts[best_value]
    for value in cleaned:
        count = counts[value]
        if count > best_count:
            best_value = value
            best_count = count
    return best_value


def flatten_messages_for_reference(messages: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            label = "Agent"
        elif role == "user":
            label = "User"
        else:
            label = role.capitalize() or "Unknown"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def normalize_role(role: str) -> Optional[str]:
    role_lower = role.strip().lower()
    if role_lower in {"assistant", "agent"}:
        return "assistant"
    if role_lower == "user":
        return "user"
    return None


def clean_json_string(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end != -1:
        return text[start:end]
    return text


PEND_PATTERN = re.compile(r"\[PEND(\d+)S\]")
HARDCODED_GUIDANCE_FIELDS = {
    "s2_participation_level",
    "s2_message_mode",
    "s2_verbosity_level",
    "s2_delay_level",
    "s2_no_response_recovery",
}
GUIDANCE_SUMMARY_FIELD_MAP = {
    "user_style": ("user_style", False),
    "user_opening": ("user_opening", False),
    "assistant_style": ("assistant_style", False),
    "assistant_opening": ("assistant_opening", False),
    "primary_interaction_type": ("primary_interaction_type", False),
    "secondary_interaction_types": ("secondary_interaction_type", True),
    "required_events": ("required_events", True),
    "target_turn_range": ("target_turn_range", False),
    "must_end_with_assistant": ("must_end_with_assistant", False),
    "forbidden_patterns": ("forbidden_patterns", True),
}
GUIDANCE_SECTION_TITLE = "【当前样本的特殊场景补充】"


def load_guidance_config(seed_path: str) -> Dict[str, Any]:
    guidance_path = Path(seed_path).parent / "guidance.json"
    if not guidance_path.exists():
        raise FileNotFoundError(f"未找到 guidance 配置文件: {guidance_path}")
    raw_text = guidance_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"guidance 配置文件为空: {guidance_path}")
    data = json.loads(raw_text)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{guidance_path} 不是非空 JSON 对象")
    return data


def normalize_guidance_entries(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        entries = []
        for item in value:
            text = str(item).strip()
            if text:
                entries.append(text)
        return entries
    return []


def lookup_guidance_entries(
    guidance_config: Dict[str, Any],
    field_key: str,
    field_value: str,
    guidance_type: str,
) -> List[str]:
    if not field_value or field_key in HARDCODED_GUIDANCE_FIELDS:
        return []
    value_block = (
        guidance_config.get("field_guidances", {})
        .get(field_key, {})
        .get(field_value, {})
    )
    if not isinstance(value_block, dict):
        return []
    return normalize_guidance_entries(value_block.get(guidance_type))


def build_extra_guidance_lines(seed_summary: Dict[str, Any], guidance_type: str) -> List[str]:
    guidance_config = seed_summary.get("guidance_config")
    if not isinstance(guidance_config, dict) or not guidance_config:
        raise ValueError(
            f"seed_summary 缺少 guidance_config，无法构造 {guidance_type} prompt"
        )

    lines = normalize_guidance_entries(guidance_config.get(f"{guidance_type}_global"))
    for summary_key, (field_key, is_multi) in GUIDANCE_SUMMARY_FIELD_MAP.items():
        raw_value = seed_summary.get(summary_key)
        if is_multi:
            values = raw_value if isinstance(raw_value, list) else []
        else:
            values = [raw_value] if raw_value else []
        for value in values:
            lines.extend(
                lookup_guidance_entries(guidance_config, field_key, str(value), guidance_type)
            )

    deduped_lines = ordered_unique(line for line in lines if line)
    if not deduped_lines:
        seed_ids = seed_summary.get("seed_request_ids") or []
        raise ValueError(
            f"guidance_config 未产出 {guidance_type} 侧提示，seed_request_ids={seed_ids}"
        )
    return deduped_lines


def format_extra_guidance_block(lines: Sequence[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def assert_guidance_in_prompt(
    messages: Sequence[Dict[str, str]],
    guidance_lines: Sequence[str],
    guidance_type: str,
) -> None:
    if len(messages) < 2 or messages[1].get("role") != "user":
        raise ValueError(f"{guidance_type} prompt 缺少 user 消息，无法校验 guidance 注入")
    prompt_text = str(messages[1].get("content", ""))
    if GUIDANCE_SECTION_TITLE not in prompt_text:
        raise ValueError(f"{guidance_type} prompt 缺少特殊场景补充区块")
    for line in guidance_lines:
        expected_line = f"- {line}"
        if expected_line not in prompt_text:
            raise ValueError(f"{guidance_type} prompt 缺少 guidance 文本: {expected_line}")


def load_content_items(path: str, split: str) -> List[Dict[str, Any]]:
    records = load_jsonl(path)
    items: List[Dict[str, Any]] = []
    for row in records:
        messages = row.get("messages", [])
        if not messages:
            continue
        system_message = messages[0]
        if str(system_message.get("role", "")).lower() != "system":
            continue
        dialogue_messages = messages[1:]
        first_role = str(dialogue_messages[0].get("role", "")).lower() if dialogue_messages else ""
        expected_first_role = "assistant" if split == "outbound" else "user"
        if first_role != expected_first_role:
            continue

        prompt_id = str(row.get("prompt_id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        sample_id = f"{split}:{prompt_id}"
        items.append(
            {
                "sample_id": sample_id,
                "split": split,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "original_system": str(system_message.get("content", "")).strip(),
                "original_messages": dialogue_messages,
                "first_role": first_role,
            }
        )
    return items


def load_seed_items(path: str) -> List[Dict[str, Any]]:
    guidance_config = load_guidance_config(path)
    seeds = []
    for row in load_jsonl(path):
        cleaned = row.get("cleaned")
        if not isinstance(cleaned, dict):
            continue
        request = cleaned.get("剧本请求")
        if not isinstance(request, dict):
            continue
        seeds.append(
            {
                "request_id": row.get("request_id") or request.get("请求ID") or "",
                "request": request,
                "guidance_config": guidance_config,
            }
        )
    return seeds


def build_seed_summary(sampled_seeds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    requests = [seed["request"] for seed in sampled_seeds]

    def person(seed_request: Dict[str, Any]) -> Dict[str, Any]:
        return seed_request.get("人物风格包", {}) or {}

    def interaction(seed_request: Dict[str, Any]) -> Dict[str, Any]:
        return seed_request.get("交互包", {}) or {}

    def s2(seed_request: Dict[str, Any]) -> Dict[str, Any]:
        return seed_request.get("S2策略包", {}) or {}

    def constraint(seed_request: Dict[str, Any]) -> Dict[str, Any]:
        return seed_request.get("生成约束", {}) or {}

    assistant_style = choose_representative([person(req).get("助手风格") for req in requests])
    assistant_opening = choose_representative([person(req).get("助手开场") for req in requests])
    user_style = choose_representative([person(req).get("用户风格") for req in requests])
    user_opening = choose_representative([person(req).get("用户开场") for req in requests])
    primary_interaction = choose_representative([interaction(req).get("主交互类型") for req in requests])
    secondary_interactions = ordered_unique(
        interaction(req).get("次交互类型")
        for req in requests
        if interaction(req).get("次交互类型")
    )
    required_events = ordered_unique(
        event
        for req in requests
        for event in (interaction(req).get("必须出现的事件") or [])
        if event
    )
    s2_participation = choose_representative([s2(req).get("S2参与程度") for req in requests])
    s2_message_mode = choose_representative([s2(req).get("S2出消息方式") for req in requests])
    s2_verbosity = choose_representative([s2(req).get("S2碎嘴程度") for req in requests])
    s2_delay = choose_representative([s2(req).get("S2延迟程度") for req in requests])
    s2_no_response_recovery = choose_representative(
        [s2(req).get("S2无返回时纠错方式") for req in requests]
    )
    target_turn_range = choose_representative([constraint(req).get("目标轮数范围") for req in requests])
    must_end_with_assistant = choose_representative(
        [constraint(req).get("必须以助手结尾") for req in requests]
    )
    forbidden_patterns = ordered_unique(
        pattern
        for req in requests
        for pattern in (constraint(req).get("禁止出现") or [])
        if pattern
    )

    return {
        "seed_request_ids": [seed["request_id"] for seed in sampled_seeds],
        "assistant_style": assistant_style,
        "assistant_opening": assistant_opening,
        "user_style": user_style,
        "user_opening": user_opening,
        "primary_interaction_type": primary_interaction,
        "secondary_interaction_types": secondary_interactions,
        "required_events": required_events,
        "s2_participation_level": s2_participation,
        "s2_message_mode": s2_message_mode,
        "s2_verbosity_level": s2_verbosity,
        "s2_delay_level": s2_delay,
        "s2_no_response_recovery": s2_no_response_recovery,
        "target_turn_range": target_turn_range,
        "must_end_with_assistant": must_end_with_assistant,
        "forbidden_patterns": forbidden_patterns,
        "seed_snapshots": requests,
    }


def with_runtime_guidance(
    seed_summary: Dict[str, Any], guidance_config: Dict[str, Any]
) -> Dict[str, Any]:
    if not isinstance(guidance_config, dict) or not guidance_config:
        raise ValueError("guidance_config 不能为空，无法注入运行时 prompt")
    if "guidance_config" in seed_summary:
        raise ValueError("seed_summary 不应持久化 guidance_config")
    enriched = dict(seed_summary)
    enriched["guidance_config"] = guidance_config
    return enriched


def build_writer_system(content_item: Dict[str, Any], seed_summary: Dict[str, Any]) -> str:
    base_system = content_item["original_system"].strip() or "你是有用的助手。"
    # SI 场景：system 已经是 SI 专用格式，不追加风格字段
    if "同声传译" in base_system:
        return base_system
    additions = []
    if seed_summary.get("assistant_style"):
        additions.append(f"你的助手风格是：{seed_summary['assistant_style']}。")
    if seed_summary.get("assistant_opening"):
        additions.append(f"你的开场方式要求：{seed_summary['assistant_opening']}。")

    return "\n".join([base_system] + additions).strip()


def format_seed_constraints(seed_summary: Dict[str, Any]) -> str:
    lines = []
    if seed_summary.get("user_style"):
        lines.append(f"- 用户风格：{seed_summary['user_style']}")
    if seed_summary.get("user_opening"):
        lines.append(f"- 用户开场：{seed_summary['user_opening']}")
    if seed_summary.get("primary_interaction_type"):
        lines.append(f"- 主交互类型：{seed_summary['primary_interaction_type']}")
    if seed_summary.get("secondary_interaction_types"):
        lines.append(
            "- 次交互变化："
            + "、".join(seed_summary["secondary_interaction_types"])
        )
    if seed_summary.get("required_events"):
        lines.append(
            "- 必须出现的事件："
            + "、".join(seed_summary["required_events"])
        )

    s2_parts = []
    if seed_summary.get("s2_participation_level"):
        s2_parts.append(f"S2参与程度={seed_summary['s2_participation_level']}")
    if seed_summary.get("s2_message_mode"):
        s2_parts.append(f"S2出消息方式={seed_summary['s2_message_mode']}")
    if seed_summary.get("s2_verbosity_level"):
        s2_parts.append(f"S2碎嘴程度={seed_summary['s2_verbosity_level']}")
    if seed_summary.get("s2_delay_level"):
        s2_parts.append(f"S2延迟程度={seed_summary['s2_delay_level']}")
    if seed_summary.get("s2_no_response_recovery"):
        s2_parts.append(
            f"S2无返回时纠错方式={seed_summary['s2_no_response_recovery']}"
        )
    if s2_parts:
        lines.append("- S2策略： " + "；".join(s2_parts))

    if seed_summary.get("target_turn_range"):
        lines.append(f"- 目标轮数范围：{seed_summary['target_turn_range']}")
    if seed_summary.get("must_end_with_assistant"):
        lines.append(
            f"- 必须以助手结尾：{seed_summary['must_end_with_assistant']}"
        )
    if seed_summary.get("forbidden_patterns"):
        lines.append(
            "- 禁止出现："
            + "、".join(seed_summary["forbidden_patterns"])
        )

    return "\n".join(lines)


def format_raw_seed_block(seed_summary: Dict[str, Any]) -> str:
    payload = {
        "seed_request_ids": seed_summary.get("seed_request_ids", []),
        "seed_snapshots": seed_summary.get("seed_snapshots", []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def describe_seed_fields() -> str:
    return """
- `人物风格包.用户风格`：面向编剧和导演。描述用户在对话中的脾气、表达方式、耐心程度和情绪基调；编剧要把它写进用户台词，导演不能把它标成额外知识，只能忠实演绎。
- `人物风格包.用户开场`：面向编剧和导演。描述用户第一轮或前几轮的开口方式，影响开场是否寒暄、是否直接切题。
- `人物风格包.助手风格`：面向编剧和导演。描述助手稳定的人设与说话风格，应该长期一致地体现在语气、措辞、安抚方式和推进方式里。
- `人物风格包.助手开场`：面向编剧和导演。描述助手的起手方式，主要影响第一句或前几句的开场口吻。
- `交互包.主交互类型`：面向编剧和导演。描述整段对话最主要的推进模式，决定剧情主线怎么往前走。
- `交互包.次交互类型`：面向编剧和导演。描述主线之外的次级变化，用来制造改口、补充条件、转折、追问或节奏变化。
- `交互包.必须出现的事件`：面向编剧和导演。描述剧情里必须实际发生的关键节点，不能漏掉。
- `S2策略包.S2参与程度`：面向编剧和导演。它决定这段对话有多依赖查询、核查、计算、操作或推理。
  `关`：尽量不要把任务写成强事实核查型、强检索型、强计算型问题，更像普通闲聊、轻陪伴、浅建议、常识性回应。
  `低`：只允许少量查询、少量计算、少量核查或简单操作，S2 不是主线。
  `中`：需要中等强度的查询、核查、计算或操作，S2 会多次参与，但不能铺天盖地。
  `高`：需要高强度、较难或多步骤的问题求解、核查或操作，S2 是支撑回答的重要部分。
- `S2策略包.S2出消息方式`：主要面向导演，也给编剧作为节奏参考。
  `大段返回`：内部信息更适合成块到达，导演应让 `「...」` 相对完整、成段，减少过碎切分。
  `小段返回`：内部信息更适合分小块到达，导演应把 `「...」` 分成多次较短但仍完整的信息片段。
- `S2策略包.S2碎嘴程度`：主要面向导演，也给编剧作为表层节奏参考。
  `低频`：S2 很少出条子，导演不要频繁塞 `「...」`。
  `中频`：S2 适中参与，哪里确实需要事实、结果、条件或计算结论，哪里再出条子。
  `高频`：S2 参与比较频繁，但每条 `「...」` 仍必须完整，不能变成残缺关键词。
  `洪泛`：S2 非常密集地参与，导演可以多次插入 `「...」`，但仍要保证时序自然、信息完整、不显得乱。
- `S2策略包.S2延迟程度`：面向编剧和导演。描述内部查询/推理/操作返回速度，决定等待感如何表演。
  `低延迟`：结果回来较快，允许较快进入事实型回答，但仍要自然，不要像瞬移。
  `中延迟`：需要明显的过渡、垫话、确认或安抚，不能刚触发就立刻拿到完整结果。
  `高延迟`：要明确演出“还在查询/还在确认/请稍等”的过程，等待感更强。
  `故障无返回`：第一次尝试可能拿不到结果，要演出失败、致歉、再试一次；如果还不行，就不要编造。
- `S2策略包.S2无返回时纠错方式`：面向编剧和导演。描述查询/推理/操作迟迟没有结果时，助手应该怎样自然补救。注意是表演助手表层话术，不是把内部机制直接说出口。
- `生成约束.目标轮数范围`：面向编剧和导演。描述整段对话大致应该有多少轮，影响长度和节奏密度。
- `生成约束.必须以助手结尾`：面向编剧和导演。描述结尾角色约束，决定最后一轮应该由谁收尾。
- `生成约束.禁止出现`：面向编剧和导演。描述全局禁忌模式，是编剧和导演都必须严格规避的错误行为。
""".strip()


def build_writer_task_background() -> str:
    return """
你现在在做两阶段流水线的第一阶段，也就是编剧阶段。
编剧的职责是把内容主题、system设定和 seed 约束改写成自然、可演、可被导演二次标注的纯文本剧本。
这里默认存在一个“表层助手”和“内部查询/推理系统”的双层工作机制，但编剧阶段不需要显式输出任何 S2 消息，也不要写 `「...」`、`[THINK]`、`[WAIT]`、`[CUT]` 这类导演标签。
你只需要把这种双层机制体现在剧情表面现象上：什么时候助手会先垫一句，什么时候会确认、致歉、改口、等待、重试、收尾，都应该通过自然台词和轮次节奏演出来。
如果 seed 里给了 `S2参与程度=关`，就尽量不要把问题写成强事实核查、强查询、强计算的任务，更像普通闲聊、轻陪伴或无需外部结果支撑的普通交流。
如果 seed 里给了 `S2参与程度=低/中/高`，你要把它理解为剧情对查询、核查、计算、操作或推理的依赖程度不同；强度越高，越要在表层对话里演出等待、核查、改口、重试或确认过程。
如果 seed 里给了低延迟、中延迟、高延迟、故障无返回之类的信息，你要把它理解成“助手表面上会经历等待、查询/推理/操作失败、再次尝试、无法确认、最终收束”的节奏约束，而不是把内部机制直接说出来。
""".strip()


def build_director_task_background() -> str:
    return """
你现在在做两阶段流水线的第二阶段，也就是导演阶段。
编剧负责写自然剧本，导演负责把自然剧本转换成带控制标签的训练数据。
这里默认存在一个“表层助手”和“内部查询/推理系统”的双层工作机制：用户听到的是表层助手的话，`[THINK]` / `「...」` / `[WAIT]` 等标签标的是内部查询或推理的触发、返回、暂停与失败。
你的职责不是改剧情，而是忠实演绎剧本，并把 seed 里的隐藏约束精确落实到标签时序、信息到达顺序、等待感、重试感、失败感和收尾方式上。
尤其要把 seed 里与 S2 延迟、无返回、纠错方式有关的约束真正演出来，而不是只在摘要里看过却没有体现在结果里。
""".strip()


def build_delay_guidance(seed_summary: Dict[str, Any]) -> str:
    participation = seed_summary.get("s2_participation_level", "")
    message_mode = seed_summary.get("s2_message_mode", "")
    verbosity = seed_summary.get("s2_verbosity_level", "")
    delay = seed_summary.get("s2_delay_level", "")
    recovery = seed_summary.get("s2_no_response_recovery", "")
    lines = [
        "- 如果 seed 里出现 S2 延迟相关约束，你必须把“等待中的表层话术”演出来，不能让事实型回答无缘无故瞬间到达。",
        "- 所谓“承认查询/推理/操作有问题”，是指助手只能从表层角度说“我这边刚才没查到”“我这边结果还没出来”“我再帮你试一次”“抱歉刚才那次没成功”这类话，不能直接暴露自己有 S2 或内部模块。",
    ]
    if participation == "关":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“关”：尽量不要把这段对话标成强事实核查、强查询、强计算场景。除非剧本表层已经明确要求，否则不要强行插入大量 `[THINK]` 或 `「...」`，整体应更接近普通闲聊或无需依赖外部结果的自然交流。"
        )
    if participation == "低":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“低”：只在少量确实需要核查、计算、操作或查询的地方触发 `[THINK]` 和 `「...」`，不要铺太满。"
        )
    if participation == "中":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“中”：需要中等强度的查询、核查、计算或操作支撑回答，S2 会多次参与，但仍要克制、自然。"
        )
    if participation == "高":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“高”：这是高强度、较难或多步骤的问题，导演应允许更多 `[THINK]` 与 `「...」` 参与，但每次都要符合时序和因果。"
        )
    if message_mode == "大段返回":
        lines.append(
            "- 当前样本的 `S2出消息方式` 是“大段返回”：相关 `「...」` 更适合成块到达，不要切得过碎。"
        )
    if message_mode == "小段返回":
        lines.append(
            "- 当前样本的 `S2出消息方式` 是“小段返回”：相关 `「...」` 更适合分成多次较短但完整的信息片段到达。"
        )
    if verbosity == "低频":
        lines.append(
            "- 当前样本的 `S2碎嘴程度` 是“低频”：不要频繁插入 `「...」`，只在真正需要时出现。意思就是`「...」`要少。这条规则与`S2参与程度`无关，S2参与程度控制的是难度和S2的需要程度，S2碎嘴程度评价的是S2的风格。"
        )
    if verbosity == "中频":
        lines.append(
            "- 当前样本的 `S2碎嘴程度` 是“中频”：S2 参与频率适中，保持自然节奏，不稀也不过密。意思就是`「...」`要适中。这条规则与`S2参与程度`无关，S2参与程度控制的是难度和S2的需要程度，S2碎嘴程度评价的是S2的风格。"
        )
    if verbosity == "高频":
        lines.append(
            "- 当前样本的 `S2碎嘴程度` 是“高频”：可以较频繁插入 `「...」`，但每条都必须是完整信息，不能退化成关键词。意思就是`「...」`要比较多。这条规则与`S2参与程度`无关，S2参与程度控制的是难度和S2的需要程度，S2碎嘴程度评价的是S2的风格。无论S2多么嘈杂，你与用户的交互必须按照助手性格来，不能错乱。"
        )
    if verbosity == "洪泛":
        lines.append(
            "- 当前样本的 `S2碎嘴程度` 是“洪泛”：S2 可以非常密集地参与，但你仍必须保证信息完整、时序自然、标签不过载。意思就是`「...」`要多。这条规则与`S2参与程度`无关，S2参与程度控制的是难度和S2的需要程度，S2碎嘴程度评价的是S2的风格。无论S2多么嘈杂，你与用户的交互必须按照助手性格来，不能错乱。"
        )
    if delay == "低延迟":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“低延迟”：结果可以比较快回来，但也要保留基本的自然衔接，不要让事实像零延迟瞬移出现。"
        )
    if delay == "中延迟":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“中延迟”：首次 `[THINK]` 之后，通常要先有 1 到 2 句自然垫话、确认、安抚或过渡，再让相关 `「...」` 到达。不要让事实型内容秒回。"
        )
    if delay == "高延迟":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“高延迟”：你必须演出明显等待感。通常要先有 1 到 2 句，表层助手应明确表示还在查询/推理/操作、还在确认、请稍等，而不是假装已经知道答案。"
        )
    if delay == "故障无返回":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“故障无返回”：第一次 `[THINK]` 后，如果没有返回，助手必须承认这次查询/推理/操作没有成功并道歉，然后再次尝试一次 `[THINK]`。"
        )
        lines.append(
            "- 如果第二次尝试后仍然没有可用返回，就不要再编造事实。助手应再次道歉，明确自己当前无法确认，并自然结束对话或结束当前轮次。"
        )
    if recovery:
        lines.append(
            f"- 当前样本的 `S2无返回时纠错方式` 是“{recovery}”：你要把这个纠错思路落实成表层助手的自然话术和收束方式。并且当故障时，不要在对话中加入`「...」`包裹的S2信息。因为S2故障了。"
        )
    return "\n".join(lines)


def build_writer_s2_guidance(seed_summary: Dict[str, Any]) -> str:
    participation = seed_summary.get("s2_participation_level", "")
    delay = seed_summary.get("s2_delay_level", "")
    recovery = seed_summary.get("s2_no_response_recovery", "")
    lines = [
        "- 编剧阶段就要先把这段对话定性清楚：它到底更像普通交流，还是少量查询型、中等查询型、高强度查询/计算/核查型。这个强度必须在剧本结构里提前决定，不能把难题留给导演硬补。",
    ]
    if participation == "关":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“关”：请把剧本写成不依赖外部查询/推理/操作结果也能自然成立的对话。优先写普通闲聊、轻咨询、浅建议、常识性说明、情绪安抚或流程性沟通，不要把核心推进建立在复杂事实核查、复杂计算或多步检索上。"
        )
    if participation == "低":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“低”：剧本里可以有少量查询、核查、计算或操作，但它不是主线。大部分轮次仍应靠自然沟通推进，只在少数节点需要助手去查一下、算一下、核一下。"
        )
    if participation == "中":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“中”：剧本应明显依赖中等强度的查询、核查、计算或操作。多个关键节点都可以需要结果支撑，但对话仍要自然，不能轮轮都在机械查询/推理/操作。"
        )
    if participation == "高":
        lines.append(
            "- 当前样本的 `S2参与程度` 是“高”：编剧阶段就要把它写成高强度、较难或多步骤的问题场景。核心推进必须依赖多次查询、核查、计算、推理或操作；表层台词里要提前埋好等待、确认、分步推进、可能重试或改口的空间，否则导演后面圆不回来。"
        )
    if delay == "低延迟":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“低延迟”：即使结果回来较快，编剧也要保留最基本的自然衔接，不要让回答显得像无因果瞬间知道。"
        )
    if delay == "中延迟":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“中延迟”：编剧要预留 1 到 2 句自然垫话、确认或安抚，让后续导演有空间演出等待感。"
        )
    if delay == "高延迟":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“高延迟”：编剧必须把“还在查、还在确认、请稍等”的表层过程写进剧情，否则导演无法自然演出高延迟。"
        )
    if delay == "故障无返回":
        lines.append(
            "- 当前样本的 `S2延迟程度` 是“故障无返回”：编剧必须提前把失败、致歉、再次尝试、再次失败后不乱说并收束的剧情空间写出来，不能把剧本写成一次查询/推理/操作就顺利给出答案。"
        )
    if recovery:
        lines.append(
            f"- 当前样本的 `S2无返回时纠错方式` 是“{recovery}”：编剧要把这种补救思路提前体现在表层剧情里，让导演只是忠实标注，而不是替你重写剧情。"
        )
    return "\n".join(lines)


def construct_writer_messages(
    content_item: Dict[str, Any],
    writer_system: str,
    seed_summary: Dict[str, Any],
) -> List[Dict[str, str]]:
    split = content_item["split"]
    reference_messages = flatten_messages_for_reference(content_item["original_messages"])
    first_speaker = "Agent" if split == "outbound" else "User"
    task_intro = "外呼任务" if split == "outbound" else "内呼任务"
    writer_guidance_lines = build_extra_guidance_lines(seed_summary, "writer")
    writer_extra_guidance = format_extra_guidance_block(writer_guidance_lines)
    extra_guidance_section = f"\n{GUIDANCE_SECTION_TITLE}\n{writer_extra_guidance}\n"

    user_content = f"""
请根据以下素材生成新的全双工对话剧本：

【任务背景】
{build_writer_task_background()}

【seed 字段释义】
{describe_seed_fields()}

【当前样本的编剧侧 S2 强度指导】
{build_writer_s2_guidance(seed_summary)}
{extra_guidance_section}
【写作要求】
1. 这是重写，不是照抄。保留原始内容的主题、任务目标、核心知识点或服务目标，但具体措辞、轮次组织、打断点、承接方式要重新生成。
2. 第一位说话者必须是 {first_speaker}。
3. 如果是外呼，Agent 要主动发起联系；如果是内呼，User 要先发起诉求。
4. 不要把 seed 的结构化字段名直接说出来，不要把“交互包”“人物风格包”“S2策略包”这些词写进剧本。
5. 助手不能提前知道用户脾气、未来会发生的事件或 S2 细节；这些只能体现为自然发生的剧情。
6. 如果约束之间有轻微冲突，请优先保证角色自然、首轮说话方正确、必须事件出现、剧情可演、结尾满足要求。
7. 为了保证下游导演可标注，严格使用 `[状态]` + `User/Agent: "台词"` 的格式。
8. 容易口播的英文品牌、型号、缩写请直接保留，不要为了口语化硬写成生硬的中文念法，比如 `R8 Spyder`、`RSQ`、`TTS` 没必要强行改写。
9. 如果用户沉默了一小段时间再继续说，可以使用 `[User沉默了1秒后]`、`[User沉默了2秒后]` 这类状态；这个写法只允许出现在用户轮前，不允许写 `Agent沉默了...`。
10. 当前只使用 1 条 seed 来约束当前剧本，不要把多条 seed 混合成一条样本。
11. 编剧阶段不要显式写出 S2 消息，不要输出 `「...」`、`[THINK]`、`[WAIT]`、`[CUT]`。如果 seed 涉及高延迟、故障无返回或重试，请只在表层对话里通过垫话、致歉、再次尝试、无法确认、自然收尾来体现。

【任务类型】
{task_intro}

【保留给助手的 system 设定】
{writer_system}

【当前这条样本对应的 seed 约束摘要】
{format_seed_constraints(seed_summary)}

【当前这条样本对应的 RAW seed】
{format_raw_seed_block(seed_summary)}

【原始内容主题】
{content_item["prompt"]}

【原始对话参考】
{reference_messages}
"""
    messages = [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]
    assert_guidance_in_prompt(messages, writer_guidance_lines, "writer")
    return messages


def construct_director_messages(writer_record: Dict[str, Any]) -> List[Dict[str, str]]:
    split = writer_record["split"]
    first_speaker = "Agent" if split == "outbound" else "User"
    seed_summary = writer_record["seed_summary"]
    seed_notes = format_seed_constraints(seed_summary)
    director_guidance_lines = build_extra_guidance_lines(seed_summary, "director")
    director_extra_guidance = format_extra_guidance_block(director_guidance_lines)
    extra_guidance_section = f"\n{GUIDANCE_SECTION_TITLE}\n{director_extra_guidance}\n"

    user_content = f"""
请将以下剧本转换为导演标注数据。

【任务背景】
{build_director_task_background()}

【seed 字段释义】
{describe_seed_fields()}

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

【样本类型】
{split}

【首轮说话方要求】
{first_speaker}

【保留给助手的 system 设定】
{writer_record["writer_system"]}

【编剧约束摘要】
{seed_notes}

【当前这条样本对应的 RAW seed】
{format_raw_seed_block(seed_summary)}

【当前样本的 S2 延迟/失败演绎要求】
{build_delay_guidance(seed_summary)}
{extra_guidance_section}
【原始内容主题】
{writer_record["prompt"]}

【原始剧本】
{writer_record["script"]}
"""
    messages = [
        {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]
    assert_guidance_in_prompt(messages, director_guidance_lines, "director")
    return messages


def call_api_with_retry(
    caller: "APICaller",
    limiter: RateLimiter,
    messages: List[Dict[str, str]],
    max_retries: int,
    retry_delay_base: int,
) -> Tuple[str, Tuple[int, int, int]]:
    last_exception: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            limiter.wait_for_token()
            raw_text, token_stats = caller.call(messages)
            if raw_text:
                return raw_text, token_stats
            raise ValueError("API returned empty response")
        except Exception as exc:  # pragma: no cover - defensive runtime path
            last_exception = exc
            if attempt < max_retries:
                time.sleep(retry_delay_base * (2 ** attempt))
    raise RuntimeError(f"API 调用失败: {last_exception}")


def build_output_paths(config: PipelineConfig, split: str) -> Dict[str, str]:
    root = Path(config.output_dir)
    suffix = ".json" if config.debug_limit is not None else ".jsonl"
    return {
        "writer": str(root / f"{split}.writer{suffix}"),
        "director": str(root / f"{split}.director{suffix}"),
    }


def select_tasks(
    items: Sequence[Dict[str, Any]],
    completed_ids: set,
    debug_limit: Optional[int],
    random_seed: int,
    key: str,
) -> List[Dict[str, Any]]:
    pending = [item for item in items if item["sample_id"] not in completed_ids]
    if debug_limit is None or debug_limit >= len(pending):
        return pending
    rng = local_rng(random_seed, f"{key}:debug_limit")
    pending_copy = pending[:]
    rng.shuffle(pending_copy)
    return pending_copy[:debug_limit]


def build_sampled_seeds(
    seed_items: Sequence[Dict[str, Any]],
    seed_sample_size: int,
    random_seed: int,
    sample_id: str,
) -> List[Dict[str, Any]]:
    if len(seed_items) < seed_sample_size:
        raise ValueError("可用 seed 数量不足")
    rng = local_rng(random_seed, f"{sample_id}:seed_selection")
    return rng.sample(list(seed_items), seed_sample_size)


def expand_content_items_with_seeds(
    content_items: Sequence[Dict[str, Any]],
    seed_items: Sequence[Dict[str, Any]],
    config: PipelineConfig,
) -> List[Dict[str, Any]]:
    expanded_items: List[Dict[str, Any]] = []
    for content_item in content_items:
        sampled_seeds = build_sampled_seeds(
            seed_items,
            config.writer_seed_sample_size,
            config.random_seed,
            content_item["sample_id"],
        )
        for seed_item in sampled_seeds:
            seed_request_id = str(seed_item.get("request_id", "")).strip() or "seed"
            expanded_item = dict(content_item)
            expanded_item["base_sample_id"] = content_item["sample_id"]
            expanded_item["seed_item"] = seed_item
            expanded_item["sample_id"] = f"{content_item['sample_id']}::{seed_request_id}"
            expanded_items.append(expanded_item)
    return expanded_items


def process_writer_sample(
    content_item: Dict[str, Any],
    seed_items: Sequence[Dict[str, Any]],
    config: PipelineConfig,
    caller: "APICaller",
    limiter: RateLimiter,
) -> Dict[str, Any]:
    if "seed_item" in content_item:
        sampled_seeds = [content_item["seed_item"]]
    else:
        sampled_seeds = build_sampled_seeds(
            seed_items,
            config.writer_seed_sample_size,
            config.random_seed,
            content_item["sample_id"],
        )
    if not sampled_seeds:
        raise ValueError(f"样本 {content_item['sample_id']} 没有可用 seed")
    seed_summary = build_seed_summary(sampled_seeds)
    runtime_seed_summary = with_runtime_guidance(
        seed_summary,
        sampled_seeds[0]["guidance_config"],
    )
    writer_system = build_writer_system(content_item, seed_summary)
    writer_messages = construct_writer_messages(
        content_item,
        writer_system,
        runtime_seed_summary,
    )
    raw_text, token_stats = call_api_with_retry(
        caller,
        limiter,
        writer_messages,
        config.max_retries,
        config.retry_delay_base,
    )
    return {
        "sample_id": content_item["sample_id"],
        "split": content_item["split"],
        "prompt_id": content_item["prompt_id"],
        "prompt": content_item["prompt"],
        "original_system": content_item["original_system"],
        "writer_system": writer_system,
        "original_messages": content_item["original_messages"],
        "seed_summary": seed_summary,
        "writer_messages": writer_messages,
        "script": raw_text.strip(),
        "prompt_tokens": token_stats[1],
        "completion_tokens": token_stats[0],
        "total_tokens": token_stats[2],
    }


def parse_director_output(raw_response: str) -> List[Dict[str, str]]:
    json_str = clean_json_string(raw_response)
    data = json.loads(json_str)
    if not isinstance(data, list):
        raise ValueError("导演输出不是 JSON 列表")

    normalized = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("导演输出项不是对象")
        raw_role = str(item.get("role", ""))
        role = normalize_role(raw_role)
        text = str(item.get("text", "")).strip()
        if not role or not text:
            raise ValueError("导演输出缺少合法 role/text")
        normalized.append({"role": role, "content": text})
    if not normalized:
        raise ValueError("导演输出为空")
    return normalized


def validate_first_role(split: str, messages: Sequence[Dict[str, str]]) -> None:
    expected = "assistant" if split == "outbound" else "user"
    actual = messages[0]["role"]
    if actual != expected:
        raise ValueError(
            f"{split} 首条非 system 角色错误，期望 {expected}，实际 {actual}"
        )


def validate_director_messages(split: str, messages: Sequence[Dict[str, str]]) -> None:
    validate_first_role(split, messages)
    previous_role: Optional[str] = None
    for index, message in enumerate(messages):
        role = message["role"]
        content = message["content"].strip()

        if role == "assistant" and PEND_PATTERN.search(content):
            raise ValueError("PEND 只能用于 User 消息，不能出现在 Assistant 消息中")

        if (
            previous_role == "user"
            and role == "assistant"
            and content.startswith("「")
        ):
            raise ValueError(
                f"第 {index + 1} 条消息违反时序：User 后的第一条 Assistant 不能直接以 S2 开头"
            )

        previous_role = role


def process_director_record(
    writer_record: Dict[str, Any],
    config: PipelineConfig,
    guidance_config: Dict[str, Any],
    caller: "APICaller",
    limiter: RateLimiter,
) -> Dict[str, Any]:
    runtime_writer_record = dict(writer_record)
    runtime_writer_record["seed_summary"] = with_runtime_guidance(
        writer_record["seed_summary"],
        guidance_config,
    )
    director_messages = construct_director_messages(runtime_writer_record)
    last_error: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            raw_response, token_stats = call_api_with_retry(
                caller,
                limiter,
                director_messages,
                0,
                config.retry_delay_base,
            )
            dialogue_messages = parse_director_output(raw_response)
            validate_director_messages(writer_record["split"], dialogue_messages)

            final_messages = [
                {"role": "system", "content": writer_record["writer_system"]},
                *dialogue_messages,
            ]
            return {
                "sample_id": writer_record["sample_id"],
                "split": writer_record["split"],
                "prompt_id": writer_record["prompt_id"],
                "prompt": writer_record["prompt"],
                "messages": final_messages,
                "metadata": {
                    "writer_system": writer_record["writer_system"],
                    "original_system": writer_record["original_system"],
                    "seed_summary": writer_record["seed_summary"],
                    "writer_script": writer_record["script"],
                    "writer_total_tokens": writer_record.get("total_tokens", 0),
                    "director_prompt_tokens": token_stats[1],
                    "director_completion_tokens": token_stats[0],
                    "director_total_tokens": token_stats[2],
                },
            }
        except Exception as exc:
            last_error = exc
            if attempt < config.max_retries:
                time.sleep(config.retry_delay_base * (2 ** attempt))

    raise RuntimeError(f"导演阶段失败: {last_error}")


def run_writer_stage(
    split: str,
    config: PipelineConfig,
    content_items: Sequence[Dict[str, Any]],
    seed_items: Sequence[Dict[str, Any]],
    caller: "APICaller",
    limiter: RateLimiter,
) -> List[Dict[str, Any]]:
    paths = build_output_paths(config, split)
    existing_records = load_jsonl(paths["writer"])
    completed_ids = {record["sample_id"] for record in existing_records}
    expanded_items = expand_content_items_with_seeds(content_items, seed_items, config)
    tasks = select_tasks(
        expanded_items,
        completed_ids,
        config.debug_limit,
        config.random_seed,
        f"{split}:writer",
    )
    if not tasks:
        print(f"[writer:{split}] 没有待处理任务")
        return existing_records

    print(
        f"[writer:{split}] 总样本 {len(expanded_items)}，"
        f"已完成 {len(completed_ids)}，待执行 {len(tasks)}"
    )
    records = existing_records[:]
    records_by_id = {record["sample_id"]: record for record in records}
    session_completed = 0
    session_total_tokens = 0

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(
                process_writer_sample,
                item,
                seed_items,
                config,
                caller,
                limiter,
            ): item["sample_id"]
            for item in tasks
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc=f"Writer-{split}")
        for future in pbar:
            sample_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                pbar.set_postfix({"Fail": sample_id})
                print(f"[writer:{split}] {sample_id} 失败: {exc}")
                continue

            records_by_id[result["sample_id"]] = result
            session_completed += 1
            session_total_tokens += int(result.get("total_tokens", 0))
            avg_tokens = (
                session_total_tokens / session_completed if session_completed else 0
            )
            pbar.set_postfix({"Suc": session_completed, "AvgTok": int(avg_tokens)})

            if session_completed % config.save_interval == 0:
                save_jsonl(
                    [records_by_id[key] for key in sorted(records_by_id)],
                    paths["writer"],
                )

    final_records = [records_by_id[key] for key in sorted(records_by_id)]
    save_jsonl(final_records, paths["writer"])
    print(f"[writer:{split}] 完成后总记录数: {len(final_records)}")
    return final_records


def run_director_stage(
    split: str,
    config: PipelineConfig,
    writer_records: Sequence[Dict[str, Any]],
    caller: "APICaller",
    limiter: RateLimiter,
) -> List[Dict[str, Any]]:
    paths = build_output_paths(config, split)
    existing_records = load_jsonl(paths["director"])
    completed_ids = {record["sample_id"] for record in existing_records}
    writer_items = [record for record in writer_records if record["split"] == split]
    tasks = select_tasks(
        writer_items,
        completed_ids,
        config.debug_limit,
        config.random_seed,
        f"{split}:director",
    )
    if not tasks:
        print(f"[director:{split}] 没有待处理任务")
        return existing_records

    print(
        f"[director:{split}] 总剧本 {len(writer_items)}，"
        f"已完成 {len(completed_ids)}，待执行 {len(tasks)}"
    )
    records_by_id = {record["sample_id"]: record for record in existing_records}
    session_completed = 0
    session_total_tokens = 0
    guidance_config = load_guidance_config(config.seed_path)

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(
                process_director_record,
                writer_record,
                config,
                guidance_config,
                caller,
                limiter,
            ): writer_record["sample_id"]
            for writer_record in tasks
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc=f"Director-{split}")
        for future in pbar:
            sample_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                pbar.set_postfix({"Fail": sample_id})
                print(f"[director:{split}] {sample_id} 失败: {exc}")
                continue

            records_by_id[result["sample_id"]] = result
            meta = result.get("metadata", {})
            session_completed += 1
            session_total_tokens += int(meta.get("director_total_tokens", 0))
            avg_tokens = (
                session_total_tokens / session_completed if session_completed else 0
            )
            pbar.set_postfix({"Suc": session_completed, "AvgTok": int(avg_tokens)})

            if session_completed % config.save_interval == 0:
                save_jsonl(
                    [records_by_id[key] for key in sorted(records_by_id)],
                    paths["director"],
                )

    final_records = [records_by_id[key] for key in sorted(records_by_id)]
    save_jsonl(final_records, paths["director"])
    print(f"[director:{split}] 完成后总记录数: {len(final_records)}")
    return final_records


def resolve_splits(config: PipelineConfig) -> List[str]:
    if config.split == "both":
        return ["inbound", "outbound"]
    return [config.split]


def main() -> None:
    config = parse_args()
    os.makedirs(config.output_dir, exist_ok=True)

    caller = create_api_caller(config)
    limiter = RateLimiter(config.rpm_limit)

    all_seed_items = load_seed_items(config.seed_path)
    if len(all_seed_items) < config.writer_seed_sample_size:
        raise ValueError("seed 数量不足，无法启动编剧阶段")

    split_to_path = {
        "inbound": config.inbound_path,
        "outbound": config.outbound_path,
    }

    for split in resolve_splits(config):
        print(f"\n========== 处理 split: {split} ==========")
        content_items = load_content_items(split_to_path[split], split)
        if not content_items:
            print(f"[{split}] 未读取到有效 content")
            continue

        if config.mode == "writer_only":
            run_writer_stage(split, config, content_items, all_seed_items, caller, limiter)
            continue

        if config.mode == "director_only":
            writer_path = build_output_paths(config, split)["writer"]
            writer_records = load_jsonl(writer_path)
            if not writer_records:
                print(f"[director:{split}] 未找到编剧中间文件: {writer_path}")
                continue
            run_director_stage(split, config, writer_records, caller, limiter)
            continue

        writer_records = run_writer_stage(
            split,
            config,
            content_items,
            all_seed_items,
            caller,
            limiter,
        )
        run_director_stage(split, config, writer_records, caller, limiter)


if __name__ == "__main__":
    main()
