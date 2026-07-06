#!/usr/bin/env python3
"""
clean_scripts.py — 用 LLM 清洗 writer 输出

功能：
  1. 英文原文但中文 script → 翻译为英文（规则检测，可靠）
  2. 全量自然化：让所有剧本措辞更像真实通话中的人说话（不用规则检测，全过 LLM）

设计原则：
  - 不靠规则检测来决定是否处理：规则只能抓到已知 pattern，会换一种单调
  - 不写负面约束（禁止 XX）：负面约束缩小分布，会逼向另一批固定表达
  - 用正向目标描述：让 LLM 自己判断哪里不自然、怎么改

原始文件不动，输出到新文件。
"""

import argparse
import json
import os
import re
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

SYSTEM_PROMPT = """你是对话剧本后处理专家。收到剧本后，输出一个 JSON 对象：
{"script": "...处理后的完整剧本..."}

规则：
1. 不改对话结构和逻辑，不改轮次和因果

2. 锚词（打断/垫话状态行里 'X'）的核心要求：
   锚词是 Agent 正在说话时被打断的那个词，说明 Agent 在说到这个词时对话被打断了，
   因此锚词必须出现在句子的中间位置，其后还有实质性的词语内容，
   不能是句子的最后一个词，不能紧贴标点/省略号，不能在句末只剩标点。

   改写 Agent 台词后，必须同步更新锚词：
   - 从改写后的台词中挑选一个恰好在句子中部的词作为新锚词
   - 确认：锚词之后还有至少 3-5 个实词，句子才算真正没说完就被打断

3. 如果 Agent 台词以省略号（... 或 …）结尾且紧跟打断状态行（后面没有以...开头的续句）：
   说明台词被截断了，必须补写完整：去掉省略号，让锚词在句子中间，后面接合理内容

【规则 2/3 示例——全密文，勿从中学习任何语言风格，只学锚词位置规则】

❌ 错误 A（锚词是句末最后一词）：
Agent: "AABBCC DDEEFF GGHHII JJKKLL。"  ← JJKKLL 是句末，Agent 说完了才打断，不合理
[当 Agent 刚刚说完 'JJKKLL' 的时候打断]
✓ 修正 A（锚词在中间，后面还有内容）：
Agent: "AABBCC DDEEFF，JJKKLL PPQQRR SSTTUU。"  ← 锚词后还有 PPQQRR SSTTUU
[当 Agent 刚刚说完 'JJKKLL' 的时候打断]

❌ 错误 B（台词省略号结尾，锚词紧贴省略号）：
Agent: "YYZZAB CCDDEF，GGHHIJ..."  ← 句子截断，GGHHIJ 后啥都没有
[当 Agent 刚刚说完 'GGHHIJ' 的时候垫话]
✓ 修正 B（补写完整句，锚词居中）：
Agent: "YYZZAB CCDDEF，GGHHIJ KKLLMM NNOOPPQQ。"  ← 锚词 GGHHIJ 后接 KKLLMM NNOOPPQQ
[当 Agent 刚刚说完 'GGHHIJ' 的时候垫话]

❌ 错误 C（改写台词后忘记更新锚词，导致锚词不在台词里）：
原台词: Agent: "AABBCC，DDEEFF GGHH。"  原锚词: 'DDEEFF'
改写后: Agent: "XXYYZZ，AABBCC QQRRSS。"  ← 台词已改，但锚词还是 'DDEEFF'，台词里找不到了
✓ 修正 C（台词改了，锚词也跟着改）：
Agent: "XXYYZZ，AABBCC QQRRSS。"
[当 Agent 刚刚说完 'AABBCC' 的时候垫话]  ← 锚词更新为台词中间的 'AABBCC'

❌ 错误 D（英文翻译后锚词在句末）：
Agent: "AABB CCDD EEFF GGHH."  ← GGHH 是句末，打断无意义
[When Agent just said 'GGHH' interrupt]
✓ 修正 D（锚词在句子中间）：
Agent: "AABB CCDD GGHH IIJJ KKLL."  ← 锚词 GGHH 后还有 IIJJ and KKLL
[When Agent just said 'GGHH' interrupt]

4. 幻觉修复：prompt 末尾会附上【原始对话内容】，这是产生剧本时的真实参考内容。
   检查对话开头，若出现以下情况必须修改：
   a) User 第一句含"之前问过/上次/之前咨询过"等假记忆引用：
      → 直接改写 User 第一句，去掉假记忆表述，改为正常发起请求；
        同时把该请求所需的实际内容（从【原始对话内容】中提取）自然融入 User 的后续台词里，
        让 Agent 基于真实输入回答，不凭空引用任何细节
   b) User 要求处理某份材料（文章、题目等）但剧本里从未提供，Agent 却引用了具体内容：
      → 在 User 的台词里补入该材料内容（从【原始对话内容】提取），让 Agent 基于真实输入回答
   核心原则：Agent 看到内容才能回答，内容必须出现在剧本的 User 台词里。

5. 只输出 JSON，不输出任何解释或 markdown"""

PROMPT_TRANSLATE_S1 = """\
这是一个 S1 类型的对话（创意写作、解释、建议等），Agent 能直接从自身知识和能力作答。请将整个剧本改为英文：
1. 台词和状态行全部统一用英文，不留任何中文
2. 让对话措辞更像真实通话中的人说话：自然、多变、有生活感，不同轮次的表达方式不重复
3. 开场方式因人因境而异，第一句尽量不以 hey/ugh/hmm 等纯语气词开头
4. Agent 直接回答，如有 "let me check" / "give me a sec" / "I'll look that up" 等等待动作，必须改为直接给出答案，不保留任何假等待
5. 不描述内部操作（"querying the database"、"loading data"、"checking the system"这类要改掉）

【剧本】
{script}

【原始对话内容】
{context}"""

PROMPT_TRANSLATE_S2 = """\
这是一个 S2 类型的对话（需要查询数据、计算、执行代码等），Agent 有合理的等待/查询动作。请将整个剧本改为英文：
1. 台词和状态行全部统一用英文，不留任何中文
2. 让对话措辞更像真实通话中的人说话：自然、多变、有生活感，不同轮次的表达方式不重复
3. 开场方式因人因境而异，第一句尽量不以 hey/ugh/hmm 等纯语气词开头
4. Agent 的等待动作是真实的，用自然方式表达（"let me check"、"give me a sec"），不描述内部机制（"querying the database"、"loading data"这类要改掉）

【剧本】
{script}

【原始对话内容】
{context}"""

# 兼容旧接口
PROMPT_TRANSLATE = PROMPT_TRANSLATE_S2

PROMPT_NATURALIZE_S1 = """\
这是一个 S1 类型的对话（创意写作、翻译、解释、建议等），Agent 能直接从自身知识和能力作答，无需任何查询或等待动作。
请让措辞更像真实通话中的人说话：自然、多变、有生活感，不同轮次的表达方式不重复。
开场方式因人因境而异：对话第一句台词尽量不能以语气词（哎、唉、嗯、哦、嘿、喂等）开头。
Agent 直接回答，如有"查一下/稍等哈/核对一下/检索/调取/加载"等等待动作，必须改为直接给出答案。
例外：如果任务本质是内容核查（如判断文本是否合规），Agent 对照规则后直接说结论，不需要假装在"调取规则库"，直接说"我看看"/"对照一下"即可。

【剧本】
{script}

【原始对话内容】
{context}"""

PROMPT_NATURALIZE_S2 = """\
这是一个 S2 类型的对话（需要查询数据、计算、执行代码等），Agent 有合理的等待/查询动作。
请让措辞更像真实通话中的人说话：自然、多变、有生活感，不同轮次的表达方式不重复。
开场方式因人因境而异：对话第一句台词尽量不能以语气词（哎、唉、嗯、哦、嘿、喂等）开头。
Agent 的等待动作是真实的，用口语化方式表达（"我查查"、"稍等哈"、"给我一秒"、"让我算算"），
不描述内部机制（"检索数据库"、"调取接口"、"系统响应"、"核对数据库"、"调取文献库"这类机器术语全部改掉）。

【剧本】
{script}

【原始对话内容】
{context}"""

# 兼容旧接口
PROMPT_NATURALIZE = PROMPT_NATURALIZE_S2

PROMPT_FIX_ANCHORS = """\
以下剧本中有 [当 Agent 刚刚说完 'X' 的时候打断] 或 [当 Agent 刚刚说完 'X' 的时候垫话] 的状态行。
请检查每一处：对应的 Agent 台词中，锚词 X 必须出现在句子中间，其后还有其他实质性内容，不能是台词的最后一个词或短语。
如果某处锚词在句尾，请改写该 Agent 台词，让锚词自然出现在句中，改写后台词仍然符合对话语境。
不改动其他台词。

【剧本】
{script}"""


# ──────────────────────────────────────────────────────────────────────────────
# 语言检测（只用于翻译判断）
# ──────────────────────────────────────────────────────────────────────────────
def is_english_original(record: dict) -> bool:
    orig = record.get("original_messages", [])
    first_user = next((m.get("content", "") for m in orig if m.get("role") == "user"), "")
    if len(first_user) <= 15:
        return False
    zh = sum(1 for c in first_user if "一" <= c <= "鿿")
    asc = sum(1 for c in first_user if "a" <= c <= "z" or "A" <= c <= "Z")
    return zh < len(first_user) * 0.2 and asc > len(first_user) * 0.5


def is_chinese_script(script: str) -> bool:
    if not script:
        return False
    zh = sum(1 for c in script if "一" <= c <= "鿿")
    return zh / len(script) > 0.3


# ──────────────────────────────────────────────────────────────────────────────
# LLM 调用
# ──────────────────────────────────────────────────────────────────────────────
_tok_lock = threading.Lock()
_tok = {"prompt": 0, "completion": 0, "cached": 0, "calls": 0}


def call_llm(client: OpenAI, user_content: str) -> dict:
    """单次 LLM 调用，异常直接上抛，由调用方决定重试/退回队列。"""
    resp = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=16193,
        extra_body={
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": "low",
            }
        },
    )
    if resp.usage:
        cached = getattr(resp.usage.prompt_tokens_details, "cached_tokens", 0) or 0
        with _tok_lock:
            _tok["prompt"] += resp.usage.prompt_tokens
            _tok["completion"] += resp.usage.completion_tokens
            _tok["cached"] += cached
            _tok["calls"] += 1

    raw = resp.choices[0].message.content or ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(raw[start : end + 1])

    raise ValueError(f"无法解析 LLM 返回的 JSON: {raw[:200]}")


# ──────────────────────────────────────────────────────────────────────────────
# 锚词检测 & 定向修复
# ──────────────────────────────────────────────────────────────────────────────
_PUNCT = set(
    '\u3002\uff0c\uff01\uff1f\u3001\uff1b\uff1a\u300c\u300d'
    '\uff08\uff09\u3010\u3011\u2026\u2014~\'\".() \t\n'
    '\u201c\u201d\u2018\u2019'
)


def _anchor_is_sentence_end(agent_line: str, anchor: str) -> bool:
    """锚词后面只剩标点/空白则判定为句尾。"""
    pos = agent_line.find(anchor)
    if pos < 0:
        return False
    after = agent_line[pos + len(anchor):]
    return all(c in _PUNCT for c in after)


def find_broken_anchors(script: str) -> list[tuple[int, str, str]]:
    """
    返回 (line_idx, agent_line, anchor) 列表，表示哪些 Agent 行的锚词在句尾。
    line_idx 是对应 Agent 行在 script.split('\n') 中的索引。
    """
    broken = []
    lines = script.split("\n")
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if "的时候打断" not in stripped and "的时候垫话" not in stripped:
            continue
        m = re.search(r"说完\s*['\"""''「](.*?)['\"""''」]", stripped)
        if not m:
            continue
        anchor = m.group(1).strip()
        if not anchor:
            continue
        # 往前找最近的 Agent 行
        for j in range(i - 1, max(0, i - 6), -1):
            if lines[j].strip().startswith("Agent"):
                if _anchor_is_sentence_end(lines[j], anchor):
                    broken.append((j, lines[j], anchor))
                break
    return broken


def fix_one_agent_line(agent_line: str, anchor: str) -> str:
    """针对单条 Agent 行做定向修复，让锚词出现在句子中间。"""
    prompt = (
        f"以下是一句对话台词，其中「{anchor}」是打断/垫话的锚词，"
        f"但「{anchor}」目前在句尾，用户打断时后面没有内容了。\n"
        f"请改写这句台词，使「{anchor}」出现在句子**中间**，其后还有其他内容，"
        f"改写要自然，符合对话语境，不要凭空加无关内容。\n"
        f"只输出改写后的台词文本，不要引号以外的任何解释。\n\n"
        f"原台词：{agent_line.strip()}"
    )
    # 这次不需要 JSON 格式，直接返回文本
    for attempt in range(3):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL_PATH,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.5,
            )
            raw = resp.choices[0].message.content or ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = raw.strip('"').strip()
            if raw and anchor in raw:
                # 验证锚词不再在句尾
                if not _anchor_is_sentence_end(raw, anchor):
                    return raw
        except Exception:
            pass
        time.sleep(1 * (attempt + 1))
    return agent_line  # 修不好就保留原行


def apply_line_fixes(script: str, fixes: dict[int, str]) -> str:
    """把修复好的行替换回脚本。"""
    if not fixes:
        return script
    lines = script.split("\n")
    for idx, new_line in fixes.items():
        if idx < len(lines):
            # 保留原行的缩进
            indent = len(lines[idx]) - len(lines[idx].lstrip())
            lines[idx] = " " * indent + new_line
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 单条处理
# ──────────────────────────────────────────────────────────────────────────────
def get_context(record: dict, max_chars: int = 2000) -> str:
    orig = record.get("original_messages", [])
    first_user = next((m.get("content", "") for m in orig if m.get("role") == "user"), "")
    return first_user[:max_chars] if first_user else "（无）"


_s1s2_labels: dict[str, str] = {}


def process_record(record: dict, client: OpenAI) -> dict:
    """处理单条记录，异常直接上抛。返回 None 表示跳过不落盘（缺标签等）。"""
    script = record.get("script", "")
    result = dict(record)

    needs_translate = is_english_original(record) and is_chinese_script(script)
    context = get_context(record)

    sid = record.get("sample_id", "")
    if _s1s2_labels:
        task_type = _s1s2_labels.get(sid)
        if task_type is None:
            return None  # 有标签文件但此条无标签，跳过不落盘
    else:
        task_type = "S2"  # 未加载标签文件时默认 S2

    if needs_translate:
        prompt1 = PROMPT_TRANSLATE_S1.format(script=script, context=context) if task_type == "S1" \
            else PROMPT_TRANSLATE_S2.format(script=script, context=context)
    else:
        prompt1 = PROMPT_NATURALIZE_S1.format(script=script, context=context) if task_type == "S1" \
            else PROMPT_NATURALIZE_S2.format(script=script, context=context)

    def _script_is_wrong_language(s: str) -> bool:
        if not needs_translate:
            return False
        zh = sum(1 for c in s if '一' <= c <= '鿿')
        return zh / max(len(s), 1) > 0.15

    MAX_RETRIES = 2
    last_exc = None
    for _ in range(MAX_RETRIES + 1):
        parsed = call_llm(client, prompt1)  # 异常上抛
        new_script = parsed.get("script", script)
        broken = find_broken_anchors(new_script)
        wrong_lang = _script_is_wrong_language(new_script)
        if not broken and not wrong_lang:
            result["script"] = new_script
            result["_needs_translate"] = needs_translate
            result["_task_type"] = task_type
            return result

    # 超过重试次数仍有问题，不落盘
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Clean writer scripts with LLM")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None,
                        help="输出路径，默认为输入文件同目录下的 <stem>.cleaned.jsonl")
    parser.add_argument("--max-workers", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--s1s2-labels", default=None,
                        help="S1/S2 分流标签文件（jsonl），含 sample_id 和 label 字段")
    args = parser.parse_args()

    # 加载 S1/S2 标签
    if args.s1s2_labels:
        print(f"[s1s2] 加载标签文件: {args.s1s2_labels}")
        with open(args.s1s2_labels, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    sid = r.get("sample_id")
                    lbl = r.get("label")
                    if sid and lbl:
                        _s1s2_labels[sid] = lbl
                except Exception:
                    pass
        s1_count = sum(1 for v in _s1s2_labels.values() if v == "S1")
        s2_count = sum(1 for v in _s1s2_labels.values() if v == "S2")
        print(f"[s1s2] 加载 {len(_s1s2_labels)} 条标签：S1={s1_count} S2={s2_count}")
    else:
        print("[s1s2] 未指定标签文件，全部使用 S2 prompt（保守模式）")

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / (input_path.stem + ".cleaned.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        print(f"[resume] 已完成 {len(done_ids)} 条，跳过")

    with open(input_path, encoding="utf-8") as f:
        all_lines = f.readlines()

    if args.limit > 0:
        all_lines = all_lines[: args.limit]

    todo_lines = []
    for line in all_lines:
        try:
            sid = json.loads(line).get("sample_id")
            if sid not in done_ids:
                todo_lines.append(line)
        except Exception:
            todo_lines.append(line)

    total_input = len(all_lines)
    print(f"总条数：{total_input}，待处理：{len(todo_lines)}，并发：{args.max_workers}")

    if not todo_lines:
        print("全部已完成，退出。")
        return

    stats = {"translate": 0, "naturalize": 0, "error": 0}
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

    def provider_worker(client: OpenAI, ep_name: str, worker_idx: int) -> None:
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
                    # 退回队列，让其他 endpoint 处理
                    task_queue.put(record)
                    result_queue.put(("requeued", None))
                    time.sleep(1)
                else:
                    result_queue.put(("error", None))

    def provider_dispatcher(ep: str, ep_name: str) -> None:
        client = OpenAI(api_key=API_KEY, base_url=ep, timeout=1800)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(provider_worker, client, ep_name, i)
                for i in range(args.max_workers)
            ]
            for f in as_completed(futures):
                if stop_event.is_set():
                    break
                try:
                    f.result()
                except Exception:
                    pass

    dispatcher_threads = []
    for ep_name, ep_url in zip(
        [f"ep{i}" for i in range(len(ENDPOINTS))],
        ENDPOINTS,
    ):
        t = threading.Thread(target=provider_dispatcher, args=(ep_url, ep_name), daemon=True)
        t.start()
        dispatcher_threads.append(t)

    pbar = tqdm(total=total_tasks, smoothing=0)
    processed = 0
    while processed < total_tasks:
        try:
            event, rec = result_queue.get(timeout=5.0)
        except Empty:
            if all(not t.is_alive() for t in dispatcher_threads) and task_queue.empty():
                break
            continue

        if event == "requeued":
            continue  # 不计入 processed，等重新处理

        processed += 1
        if event == "done" and rec is not None:
            write_record(rec)
            with stats_lock:
                if rec.get("_needs_translate"):
                    stats["translate"] += 1
                else:
                    stats["naturalize"] += 1
        else:
            with stats_lock:
                stats["error"] += 1

        with stats_lock:
            done_now = stats["translate"] + stats["naturalize"]
        with _tok_lock:
            p, c, ca = _tok["prompt"], _tok["completion"], _tok["cached"]
        pbar.set_postfix({
            "succ": done_now,
            "fail": stats["error"],
            "ptok": f"{p/1e6:.2f}M",
            "ctok": f"{c/1e6:.2f}M",
            "cache%": f"{ca/max(p,1)*100:.0f}%",
        }, refresh=False)
        pbar.update(1)

        if processed % 500 == 0:
            print(f"\n[tok] prompt={p/1e6:.3f}M  completion={c/1e6:.3f}M  cached={ca/1e6:.3f}M({ca/max(p,1)*100:.0f}%)")

    stop_event.set()
    done_now = stats["translate"] + stats["naturalize"]

    done_now = stats["translate"] + stats["naturalize"]
    total_done = len(done_ids) + done_now
    with _tok_lock:
        p, c, ca = _tok["prompt"], _tok["completion"], _tok["cached"]

    print(f"\n=== 统计（本次成功 {done_now} 条，失败 {stats['error']} 条，累计 {total_done}/{total_input}）===")
    print(f"  翻译+自然化 : {stats['translate']} ({stats['translate']/max(done_now,1)*100:.1f}%)")
    print(f"  仅自然化    : {stats['naturalize']} ({stats['naturalize']/max(done_now,1)*100:.1f}%)")
    print(f"  失败(未落盘) : {stats['error']} ({stats['error']/max(done_now+stats['error'],1)*100:.1f}%)")
    print(f"\n=== Token 消耗 ===")
    print(f"  prompt     : {p:,} ({p/1e6:.3f}M)")
    print(f"  completion : {c:,} ({c/1e6:.3f}M)")
    print(f"  cached     : {ca:,} ({ca/max(p,1)*100:.1f}%)")
    print(f"  总计       : {(p+c):,} ({(p+c)/1e6:.3f}M)")

    tok_log = output_path.parent / (output_path.stem + ".tok.json")
    tok_log.write_text(json.dumps({
        "prompt_tokens": p, "completion_tokens": c, "cached_tokens": ca,
        "total_tokens": p + c, "calls": _tok["calls"],
        "success": done_now, "failed": stats["error"], "total_input": total_input,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  token 日志 : {tok_log}")


if __name__ == "__main__":
    main()
