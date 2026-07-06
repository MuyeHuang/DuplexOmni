import argparse
import ast
import io
import json
import multiprocessing as mp
import os
import random
import shutil
import struct
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

MAX_EXAMPLE_RECORDS_PER_RESULT = 4

HM = os.environ.get("VOICEAGENT_ROOT", ".")
DEFAULT_INPUT_PARQUET_DIR = f"{HM}/api_generate/pipeline_outputs_parquet/finalcut_sessions"
DEFAULT_INPUT_JSONL_DIR   = (
    f"{HM}/api_generate/pipeline_outputs_BIG_director_repairedloop5_qwen397b_5node_full_20260609"
)
DEFAULT_OUTPUT_DIR      = f"{HM}/api_generate/pipeline_outputs_parquet/training_v7"
DEFAULT_OUTPUT_FILENAME = "train_data_v7"   # parquet batch prefix
DEFAULT_NUM_WORKERS     = min(32, max(4, os.cpu_count() or 1))
SESSIONS_PER_BATCH      = 100000           # session 数 / 输出 parquet 文件
ROW_GROUP_SESSIONS      = 256              # session 数 / parquet row group，控制写盘瞬时内存
DEFAULT_PREFETCH_FILES  = 2                # 并发预读 parquet 文件数；文件大时不要太高
DEFAULT_POOL_CHUNKSIZE  = 1                # 大二进制结果经 IPC 返回，必须小 chunksize
DEFAULT_WRITE_QUEUE_SIZE = 64              # row group 写队列深度，吸收写盘抖动

_SESSION_PARQUET_SCHEMA = pa.schema([
    pa.field("session_id",    pa.string()),
    pa.field("sample_id",     pa.string()),
    pa.field("split",         pa.string()),
    pa.field("system_prompt", pa.string()),
    pa.field("voice_plan",    pa.string()),
    pa.field("agent_speaker", pa.string()),
    pa.field("user_speaker",  pa.string()),
    pa.field("messages",      pa.list_(pa.struct([
        pa.field("role",    pa.string()),
        pa.field("content", pa.string()),
    ]))),
    pa.field("audios",        pa.list_(pa.large_binary())),
])


class MissingAudioFileError(FileNotFoundError):
    """Raised when a referenced audio file is missing for a session."""


class RollingParquetSessionWriter:
    """Write small row groups while keeping the old N-sessions-per-file layout."""

    def __init__(
        self,
        output_dir: str,
        output_filename: str,
        batch_idx: int,
        sessions_per_file: int,
        schema: pa.Schema,
    ) -> None:
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.batch_idx = batch_idx
        self.sessions_per_file = sessions_per_file
        self.schema = schema
        self.writer = None
        self.tmp_path = None
        self.final_path = None
        self.current_file_sessions = 0
        self.total_sessions = 0

    def _open_next_file(self) -> None:
        self.final_path = os.path.join(
            self.output_dir, f"{self.output_filename}_{self.batch_idx:06d}.parquet")
        self.tmp_path = self.final_path + ".tmp"
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)
        self.writer = pq.ParquetWriter(
            self.tmp_path,
            self.schema,
            compression="snappy",
        )
        self.current_file_sessions = 0

    def write_rows(self, rows: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        closed_files = []
        offset = 0
        while offset < len(rows):
            if self.writer is None:
                self._open_next_file()

            capacity = self.sessions_per_file - self.current_file_sessions
            take = min(capacity, len(rows) - offset)
            part = rows[offset:offset + take]
            table = pa.Table.from_pylist(part, schema=self.schema)
            self.writer.write_table(table)
            self.current_file_sessions += take
            self.total_sessions += take
            offset += take

            if self.current_file_sessions >= self.sessions_per_file:
                closed_files.append(self.close_current_file())
        return closed_files

    def close_current_file(self) -> Tuple[str, int]:
        if self.writer is None:
            raise RuntimeError("close_current_file called without an open parquet writer")
        self.writer.close()
        os.rename(self.tmp_path, self.final_path)
        closed = (self.final_path, self.current_file_sessions)
        self.writer = None
        self.tmp_path = None
        self.final_path = None
        self.current_file_sessions = 0
        self.batch_idx += 1
        return closed

    def close(self) -> List[Tuple[str, int]]:
        if self.writer is None:
            return []
        return [self.close_current_file()]

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.tmp_path and os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)
        self.tmp_path = None
        self.final_path = None
        self.current_file_sessions = 0


def parse_content(msg: Dict[str, Any]) -> Dict[str, Any]:
    """解析 message content，支持 dict 或 str(dict) 两种格式。"""
    content = msg["content"]
    return content if isinstance(content, dict) else ast.literal_eval(content)


def dump_content(obj: Dict[str, Any]) -> str:
    """将 dict 序列化为字符串，用于写回 message content。"""
    return str(obj)


def collect_invalid_s2_cases(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    invalid_cases = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        user_content = parse_content(msg)
        from_s2 = (user_content.get("from_s2") or "").strip()
        if not from_s2:
            continue

        has_think = False
        last_ctrl_idx = None
        last_ctrl = None
        nearest_asr_idx = None
        nearest_asr_text = ""
        for j in range(i - 1, -1, -1):
            if messages[j].get("role") != "assistant":
                continue
            assistant_content = parse_content(messages[j])
            s2_ctrl = (assistant_content.get("system2_control") or "").strip()
            if s2_ctrl:
                if last_ctrl_idx is None:
                    last_ctrl_idx = j
                    last_ctrl = s2_ctrl
                if "[THINK]" in s2_ctrl:
                    has_think = True
                    break

        for j in range(i - 1, -1, -1):
            if messages[j].get("role") != "assistant":
                continue
            assistant_content = parse_content(messages[j])
            asr_text = (assistant_content.get("asr") or "").strip()
            if asr_text:
                nearest_asr_idx = j
                nearest_asr_text = asr_text
                break

        reasons = []
        if last_ctrl and "[WAIT]" in last_ctrl:
            reasons.append("latest_WAIT")
        if not has_think:
            reasons.append("missing_THINK")
        if not reasons:
            continue

        invalid_cases.append(
            {
                "user_idx": i,
                "from_s2": from_s2,
                "reasons": reasons,
                "last_ctrl_idx": last_ctrl_idx,
                "last_ctrl": last_ctrl or "",
                "nearest_asr_idx": nearest_asr_idx,
                "nearest_asr": nearest_asr_text,
            }
        )
    return invalid_cases


def repair_invalid_s2_in_messages(
    messages: List[Dict[str, Any]],
    invalid_cases: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    修复非法 S2 消息：
    - 若最近控制符是 [WAIT]，则把该 [WAIT] 替换成 [THINK]。
    - 否则若前面没有 [THINK]，则找到最近有 ASR 的 assistant 轮，给它加 [THINK]。
    返回修复后的 messages 以及修复记录，便于打印统计和样例。
    """
    if invalid_cases is None:
        invalid_cases = collect_invalid_s2_cases(messages)

    repair_records = []
    for case in invalid_cases:
        target_idx = None
        repair_action = None
        before_content = None
        after_content = None

        if "latest_WAIT" in case["reasons"] and case["last_ctrl_idx"] is not None:
            target_idx = case["last_ctrl_idx"]
            target_msg = messages[target_idx]
            target_content = parse_content(target_msg)
            before_content = dict(target_content)
            old_ctrl = (target_content.get("system2_control") or "").strip()
            new_ctrl = old_ctrl.replace("[WAIT]", "[THINK]")
            if new_ctrl != old_ctrl:
                target_content["system2_control"] = new_ctrl
                messages[target_idx] = {"role": "assistant", "content": dump_content(target_content)}
                after_content = dict(target_content)
                repair_action = "wait_to_think"
        elif "missing_THINK" in case["reasons"] and case["nearest_asr_idx"] is not None:
            target_idx = case["nearest_asr_idx"]
            target_msg = messages[target_idx]
            target_content = parse_content(target_msg)
            before_content = dict(target_content)
            old_ctrl = (target_content.get("system2_control") or "").strip()
            if "[THINK]" not in old_ctrl:
                new_ctrl = (old_ctrl + " [THINK]").strip() if old_ctrl else "[THINK]"
                target_content["system2_control"] = new_ctrl
                messages[target_idx] = {"role": "assistant", "content": dump_content(target_content)}
                after_content = dict(target_content)
                repair_action = "add_think_to_nearest_asr"

        if repair_action is not None:
            repair_records.append(
                {
                    "user_idx": case["user_idx"],
                    "from_s2": case["from_s2"],
                    "reasons": list(case["reasons"]),
                    "last_ctrl": case["last_ctrl"],
                    "nearest_asr": case["nearest_asr"],
                    "target_idx": target_idx,
                    "repair_action": repair_action,
                    "before_content": before_content,
                    "after_content": after_content,
                }
            )

    return messages, repair_records


def validate_s2_messages(messages: List[Dict[str, Any]]) -> bool:
    """
    校验所有 from_s2 是否满足规则：
    - 前面必须有 [THINK]
    - 最近控制符不能是 [WAIT]
    """
    return not collect_invalid_s2_cases(messages)


def get_s2_think_distance_and_last_ctrl(messages: List[Dict[str, Any]], s2_idx: int) -> Tuple[Optional[int], str]:
    think_dist = 0
    last_ctrl = ""
    for j in range(s2_idx - 1, -1, -1):
        if messages[j].get("role") != "assistant":
            continue
        think_dist += 1
        assistant_content = parse_content(messages[j])
        s2_ctrl = (assistant_content.get("system2_control") or "").strip()
        if not last_ctrl and s2_ctrl:
            last_ctrl = s2_ctrl
        if "[THINK]" in s2_ctrl:
            return think_dist, last_ctrl
    return None, last_ctrl


def count_assistant_between(messages: List[Dict[str, Any]], left_idx: int, right_idx: int) -> int:
    count = 0
    for j in range(left_idx + 1, right_idx):
        if messages[j].get("role") == "assistant":
            count += 1
    return count


def find_previous_user_idx(messages: List[Dict[str, Any]], source_idx: int, shift: int) -> Optional[int]:
    user_seen = 0
    for j in range(source_idx - 1, -1, -1):
        if messages[j].get("role") != "user":
            continue
        user_seen += 1
        if user_seen == shift:
            return j
    return None


def collect_adjacent_same_role_cases(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cases = []
    for i in range(1, len(messages)):
        prev_role = messages[i - 1].get("role")
        curr_role = messages[i].get("role")
        if prev_role == curr_role and curr_role in {"user", "assistant"}:
            cases.append({"left_idx": i - 1, "right_idx": i, "role": curr_role})
    return cases


def perturb_shift_s2_forward(
    messages: List[Dict[str, Any]],
    rng: Optional[random.Random] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    保守前移扰动：
    - 对每个合法 S2（已满足 [THINK] 且最近不是 [WAIT]），回溯最近 [THINK] 的距离。
    - 若距离 > 4，则随机前移 1–3 个周期。
    - 若前一个 S2 消息很近，则前移距离 = min(与前一个 S2 的距离, 3)。
    - 只前移 user.content 里的 from_s2 字段，不改动 message 序列本身。
    - 按从早到晚的顺序处理，每次用更新后的 messages 再处理下一条 S2。
    返回更新后的 messages 与扰动记录。
    """
    rng = rng or random.Random()
    perturb_records = []

    idx = 0
    prev_s2_idx = None
    while idx < len(messages):
        msg = messages[idx]
        if msg.get("role") != "user":
            idx += 1
            continue

        user_content = parse_content(msg)
        from_s2 = (user_content.get("from_s2") or "").strip()
        if not from_s2:
            idx += 1
            continue

        think_dist, last_ctrl = get_s2_think_distance_and_last_ctrl(messages, idx)
        if think_dist is None or "[WAIT]" in last_ctrl or think_dist <= 4:
            prev_s2_idx = idx
            idx += 1
            continue

        max_shift = 3
        if prev_s2_idx is not None:
            dist_to_prev_s2 = count_assistant_between(messages, prev_s2_idx, idx)
            max_shift = min(max_shift, dist_to_prev_s2)

        if max_shift <= 0:
            prev_s2_idx = idx
            idx += 1
            continue

        shift = rng.randint(1, max_shift)
        target_user_idx = find_previous_user_idx(messages, idx, shift)

        if target_user_idx is None or target_user_idx >= idx:
            prev_s2_idx = idx
            idx += 1
            continue

        target_user_content = parse_content(messages[target_user_idx])
        target_from_s2 = (target_user_content.get("from_s2") or "").strip()
        if target_from_s2:
            prev_s2_idx = idx
            idx += 1
            continue

        source_user_content = dict(user_content)
        source_user_content["from_s2"] = ""
        target_user_content["from_s2"] = from_s2

        messages[target_user_idx] = {"role": "user", "content": dump_content(target_user_content)}
        messages[idx] = {"role": "user", "content": dump_content(source_user_content)}
        perturb_records.append(
            {
                "source_user_idx": idx,
                "target_user_idx": target_user_idx,
                "shift": shift,
                "think_dist": think_dist,
                "from_s2": from_s2,
            }
        )

        prev_s2_idx = target_user_idx
        idx += 1

    return messages, perturb_records


_SILENCE_WAV_BYTES = None
_G_SP_INDEX = None  # 每个 worker 进程的 sp_index（通过 initializer 注入）


def _worker_init_sp(sp_idx: dict) -> None:
    global _G_SP_INDEX
    _G_SP_INDEX = sp_idx


def _process_job_build_and_run(job: tuple) -> dict:
    """worker 进程：build session_info/chunks（用进程本地 _G_SP_INDEX）+ process。"""
    sn, raw_rows, idx = job
    session_info, raw_chunks = build_session_info_and_chunks(raw_rows, _G_SP_INDEX)
    return _process_single_session_job((sn, session_info, raw_chunks, idx))


def _build_silence_wav_bytes(duration=0.48, sample_rate=24000):
    num_samples = int(duration * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setparams((1, 2, sample_rate, num_samples, "NONE", "not compressed"))
        silence_frame = struct.pack("h", 0)
        wav_file.writeframes(silence_frame * num_samples)
    return buf.getvalue()


def _get_silence_bytes():
    global _SILENCE_WAV_BYTES
    if _SILENCE_WAV_BYTES is None:
        _SILENCE_WAV_BYTES = _build_silence_wav_bytes()
    return _SILENCE_WAV_BYTES



def _build_silence_messages() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    user_content = dump_content({"audio_input": "<audio>", "self_audio": "<audio>", "from_s2": ""})
    assistant_content = dump_content({"tts_control": "", "system2_control": "", "asr": "", "tts": ""})
    return (
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    )


def _prepend_inbound_silence_cycles(
    *,
    session_name: str,
    session_info: Dict[str, Any],
    messages: List[Dict[str, Any]],
    audios_list: List,
) -> int:
    split = str(session_info.get("split") or "").strip().lower()
    if split != "inbound":
        return 0
    session_rng = random.Random(f"{session_name}:inbound_silence")
    silence_turns = session_rng.randint(1, 5)
    silence_user_msg, silence_assistant_msg = _build_silence_messages()
    silence_bytes = _get_silence_bytes()
    for _ in range(silence_turns):
        messages.append(dict(silence_user_msg))
        messages.append(dict(silence_assistant_msg))
        audios_list.append(silence_bytes)  # r1 slot
        audios_list.append(silence_bytes)  # r2 slot
    return silence_turns


def build_system_prompt_index(jsonl_dir: str) -> Dict[str, str]:
    """从 inbound/outbound.director.jsonl 建立 {sample_id → system_prompt} 索引。"""
    index: Dict[str, str] = {}
    for fname in ("inbound.director.jsonl", "outbound.director.jsonl"):
        fpath = os.path.join(jsonl_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sid = obj.get("sample_id") or ""
                    sp = (obj.get("system_prompt") or obj.get("system") or "").strip()
                    if not sp:
                        msgs = obj.get("messages") or []
                        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
                            sp = (msgs[0].get("content") or "").strip()
                    if sid and sp:
                        index[sid] = sp
                except Exception:
                    pass
    return index


def iter_parquet_sessions(parquet_dir: str):
    """
    逐文件流式读取 parquet_dir 下的 batch_*.parquet，
    每次 yield (session_id, sorted_rows_list)，不把全部数据读进内存。
    """
    import pandas as pd
    base = Path(parquet_dir)
    parquets = sorted(base.glob("rank_*/batch_*.parquet")) or sorted(base.glob("batch_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"在 {parquet_dir} 找不到 batch_*.parquet")
    for p in parquets:
        df = pq.read_table(str(p)).to_pandas()
        for sid, grp in df.groupby("session_id", sort=False):
            yield sid, grp.sort_values("chunk_id").to_dict("records")


def build_session_info_and_chunks(rows: List[Dict], sp_index: Dict[str, str]):
    """从 chunk 级 parquet rows 还原 session_info 和 chunks 列表。"""
    first = rows[0]
    sample_id = first.get("sample_id", "")
    session_info = {
        "sample_id":    sample_id,
        "split":        first.get("split", ""),
        "system_prompt": sp_index.get(sample_id, ""),
        "voice_plan":   json.loads(first.get("voice_plan") or "{}"),
        "agent_speaker": first.get("agent_speaker", ""),
        "user_speaker":  first.get("user_speaker", ""),
    }
    chunks = []
    for row in rows:
        chunks.append({
            "chunk_id": int(row["chunk_id"]),
            "inputs": {
                "audio_input": bytes(row["r1_audio_bytes"]),
                "self_audio":  bytes(row["r2_audio_bytes"]),
                "from_s2":     row.get("from_s2") or "",
                "asr_context": row.get("asr_context") or "",
            },
            "outputs": {
                "asr":              row.get("asr") or "",
                "tts":              row.get("tts") or "",
                "tts_control":      row.get("tts_control") or "",
                "system2_control":  row.get("system2_control") or "",
            },
        })
    return session_info, chunks


def parse_args():
    parser = argparse.ArgumentParser(description="Pack chunk-level parquet into session-level training parquet")
    parser.add_argument("--input-parquet-dir", default=DEFAULT_INPUT_PARQUET_DIR,
                        help="chunk 级 parquet 根目录（含 rank_X/ 子目录）")
    parser.add_argument("--input-jsonl-dir", default=DEFAULT_INPUT_JSONL_DIR,
                        help="原始 director JSONL 目录，用于反查 system_prompt")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-filename", default=DEFAULT_OUTPUT_FILENAME)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-sessions", type=int, default=0, help="0=全量，>0=只处理前N个新session（烟测用）")
    parser.add_argument("--sessions-per-batch", type=int, default=SESSIONS_PER_BATCH,
                        help="每个输出 parquet 文件包含的 session 数")
    parser.add_argument("--row-group-sessions", type=int, default=ROW_GROUP_SESSIONS,
                        help="每次写入 parquet row group 的 session 数；越小峰值内存越低")
    parser.add_argument("--prefetch-files", type=int, default=DEFAULT_PREFETCH_FILES,
                        help="并发预读输入 parquet 文件数")
    parser.add_argument("--pool-chunksize", type=int, default=DEFAULT_POOL_CHUNKSIZE,
                        help="multiprocessing Pool chunksize；inline audio bytes 场景建议保持 1")
    parser.add_argument("--write-queue-size", type=int, default=DEFAULT_WRITE_QUEUE_SIZE,
                        help="待写 row group 队列深度；越大越不易反压处理进程，但占用更多内存")
    parser.add_argument("--resume", action="store_true", help="断点续传：跳过已写入 manifest.txt 的 session")
    return parser.parse_args()


def chunk_has_effective_payload(item):
    inputs = item.get("inputs") or {}
    outputs = item.get("outputs") or {}
    if (inputs.get("from_s2") or "").strip():
        return True
    return any(str(outputs.get(key) or "").strip() for key in ["tts_control", "system2_control", "asr", "tts"])


def process_single_session(
    session_name: str,
    session_info: Dict[str, Any],
    raw_chunks: List[Dict[str, Any]],
    order_idx: int,
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    audios_list: List[bytes] = []

    system_prompt = (session_info.get("system_prompt") or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    prepended_silence_turns = _prepend_inbound_silence_cycles(
        session_name=session_name,
        session_info=session_info,
        messages=messages,
        audios_list=audios_list,
    )

    for item in raw_chunks:
        if item.get("chunk_id") == 0 and not chunk_has_effective_payload(item):
            continue
        inputs  = dict(item.get("inputs")  or {})
        outputs = dict(item.get("outputs") or {})
        inputs.pop("asr_context", None)

        for key in ["audio_input", "self_audio"]:
            audio_bytes = inputs.get(key)
            if not audio_bytes:
                raise ValueError(f"{session_name} chunk {item.get('chunk_id')} 缺少 {key}")
            audios_list.append(bytes(audio_bytes))
            inputs[key] = "<audio>"

        messages.append({"role": "user",      "content": str(inputs)})
        messages.append({"role": "assistant", "content": str(outputs)})

    if not messages:
        raise ValueError(f"{session_name} 没有可打包的 chunk")

    invalid_cases_before = collect_invalid_s2_cases(messages)
    messages, repair_records = repair_invalid_s2_in_messages(messages, invalid_cases_before)

    session_rng = random.Random(f"{session_name}:s2_perturb")
    messages, perturb_records = perturb_shift_s2_forward(messages, rng=session_rng)

    adjacent_same_role_cases = collect_adjacent_same_role_cases(messages)
    if adjacent_same_role_cases:
        raise ValueError(f"{session_name} 存在相邻同角色消息: {adjacent_same_role_cases[:3]}")

    invalid_cases_after = collect_invalid_s2_cases(messages)
    remaining_user_idxs = {case["user_idx"] for case in invalid_cases_after}
    repaired_examples = [r for r in repair_records if r["user_idx"] not in remaining_user_idxs]

    return {
        "order_idx": order_idx,
        "session_name": session_name,
        "row": {
            "session_id":    session_name,
            "sample_id":     session_info.get("sample_id", ""),
            "split":         session_info.get("split", ""),
            "system_prompt": session_info.get("system_prompt", ""),
            "voice_plan":    json.dumps(session_info.get("voice_plan") or {}, ensure_ascii=False),
            "agent_speaker": session_info.get("agent_speaker", ""),
            "user_speaker":  session_info.get("user_speaker", ""),
            "messages":      messages,
            "audios":        audios_list,
        },
        "invalid_before_count":   len(invalid_cases_before),
        "invalid_before_session": bool(invalid_cases_before),
        "invalid_after_count":    len(invalid_cases_after),
        "invalid_after_session":  bool(invalid_cases_after),
        "repaired_examples":      repaired_examples,
        "perturb_records":        perturb_records,
        "prepended_silence_turns": prepended_silence_turns,
        "warning": None if not invalid_cases_after
                   else f"[WARN] {session_name} 修复后仍有非法 S2 消息: {len(invalid_cases_after)}",
    }


def _process_single_session_job(job: Tuple) -> Dict[str, Any]:
    session_name, session_info, raw_chunks, order_idx = job
    try:
        return process_single_session(session_name, session_info, raw_chunks, order_idx)
    except Exception as exc:
        raise RuntimeError(f"处理 Session 失败: {session_name}") from exc


def _get_mp_context():
    if os.name == "posix":
        try:
            return mp.get_context("fork")
        except ValueError:
            pass
    return mp.get_context("spawn")


def _compute_chunksize(total_jobs: int, num_workers: int) -> int:
    if total_jobs <= 0 or num_workers <= 0:
        return 1
    target_chunks = max(num_workers * 8, 1)
    chunksize = (total_jobs + target_chunks - 1) // target_chunks
    return max(1, min(16, chunksize))


def _build_job_chunks(
    jobs: List[Tuple[str, str, str, int]], sessions_per_chunk: int
) -> List[Tuple[int, List[Tuple[str, str, str, int]]]]:
    chunks = []
    for chunk_id, start in enumerate(range(0, len(jobs), sessions_per_chunk)):
        chunks.append((chunk_id, jobs[start:start + sessions_per_chunk]))
    return chunks


def _append_limited_examples(dest: List[Dict[str, Any]], records, limit: int = MAX_EXAMPLE_RECORDS_PER_RESULT) -> None:
    if len(dest) >= limit:
        return
    remaining = limit - len(dest)
    dest.extend(list(records)[:remaining])


def _process_job_chunk(task: Tuple) -> Dict[str, Any]:
    chunk_id, job_chunk, shard_dir = task
    shard_path = os.path.join(shard_dir, f"chunk_{chunk_id:06d}.parquet")
    invalid_before_count = invalid_before_sessions = 0
    invalid_after_count  = invalid_after_sessions  = 0
    repaired_examples = []; repaired_record_count = 0
    perturb_examples  = []; perturb_record_count  = 0
    warnings = []
    prepended_silence_turns = prepended_silence_sessions = 0
    written = 0
    rows = []

    for job in job_chunk:
        item = _process_single_session_job(job)
        rows.append(item["row"])
        written += 1
        invalid_before_count   += item["invalid_before_count"]
        invalid_before_sessions += int(bool(item["invalid_before_session"]))
        invalid_after_count    += item["invalid_after_count"]
        invalid_after_sessions  += int(bool(item["invalid_after_session"]))
        prepended_silence_turns    += item["prepended_silence_turns"]
        prepended_silence_sessions += int(item["prepended_silence_turns"] > 0)
        chunk_repaired = [{"session_name": item["session_name"], **r} for r in item["repaired_examples"]]
        repaired_record_count += len(chunk_repaired)
        _append_limited_examples(repaired_examples, chunk_repaired)
        chunk_perturb = [{"session_name": item["session_name"], **r} for r in item["perturb_records"]]
        perturb_record_count += len(chunk_perturb)
        _append_limited_examples(perturb_examples, chunk_perturb)
        if item["warning"] and len(warnings) < MAX_EXAMPLE_RECORDS_PER_RESULT:
            warnings.append(item["warning"])

    if rows:
        tbl = pa.Table.from_pylist(rows, schema=_SESSION_PARQUET_SCHEMA)
        tmp = shard_path + ".tmp"
        pq.write_table(tbl, tmp, compression="snappy")
        os.rename(tmp, shard_path)

    return {
        "chunk_id": chunk_id, "shard_path": shard_path, "written": written,
        "invalid_before_count": invalid_before_count, "invalid_before_sessions": invalid_before_sessions,
        "invalid_after_count": invalid_after_count,   "invalid_after_sessions": invalid_after_sessions,
        "repaired_record_count": repaired_record_count, "repaired_examples": repaired_examples,
        "perturb_record_count": perturb_record_count,   "perturb_examples": perturb_examples,
        "warnings": warnings,
        "prepended_silence_turns": prepended_silence_turns,
        "prepended_silence_sessions": prepended_silence_sessions,
    }


def process_and_save_parquet(args):
    os.makedirs(args.output_dir, exist_ok=True)
    _get_silence_bytes()

    print("正在加载 system_prompt 索引...")
    sp_index = build_system_prompt_index(args.input_jsonl_dir)
    print(f"  system_prompt 索引条目数: {len(sp_index)}")

    if args.num_workers <= 0:
        raise ValueError(f"num_workers 必须大于 0")
    if args.sessions_per_batch <= 0:
        raise ValueError(f"sessions_per_batch 必须大于 0")
    if args.row_group_sessions <= 0:
        raise ValueError(f"row_group_sessions 必须大于 0")
    if args.row_group_sessions > args.sessions_per_batch:
        args.row_group_sessions = args.sessions_per_batch
    if args.prefetch_files <= 0:
        raise ValueError(f"prefetch_files 必须大于 0")
    if args.pool_chunksize <= 0:
        raise ValueError(f"pool_chunksize 必须大于 0")
    if args.write_queue_size <= 0:
        raise ValueError(f"write_queue_size 必须大于 0")

    # 断点续传：从已有输出 parquet 读 session_id 列
    done_sessions: set = set()
    if args.resume:
        for p in sorted(Path(args.output_dir).glob(f"{args.output_filename}_*.parquet")):
            tbl = pq.read_table(str(p), columns=["session_id"])
            done_sessions.update(tbl.column("session_id").to_pylist())
        if done_sessions:
            print(f"  断点续传：已跳过 {len(done_sessions)} 个已处理 session")

    # 输出 batch 编号从已有文件数续接
    batch_idx = len(sorted(Path(args.output_dir).glob(f"{args.output_filename}_*.parquet")))

    # 找所有输入 parquet 文件
    _base = Path(args.input_parquet_dir)
    _all_parquets = sorted(_base.glob("rank_*/batch_*.parquet")) or \
                    sorted(_base.glob("batch_*.parquet"))
    if not _all_parquets:
        raise FileNotFoundError(f"找不到 parquet: {args.input_parquet_dir}")
    print(f"共 {len(_all_parquets)} 个输入 parquet 文件")

    def _load_parquet_file_io_only(pfile):
        """线程只做 I/O：读文件，过滤 done_sessions，返回 raw row dicts（不做 CPU build）。"""
        df = pq.read_table(str(pfile)).to_pandas()
        results = []
        for sid, grp in df.groupby("session_id", sort=False):
            if sid in done_sessions:
                continue
            results.append((sid, grp.sort_values("chunk_id").to_dict("records")))
        return results

    def _job_generator():
        """读盘线程只做 I/O，yield raw rows；build_session_info_and_chunks 交给 worker 进程。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        idx = 0
        parquet_iter = iter(_all_parquets)
        with ThreadPoolExecutor(max_workers=args.prefetch_files) as tex:
            futures = {}
            for _ in range(args.prefetch_files):
                try:
                    futures[tex.submit(_load_parquet_file_io_only, next(parquet_iter))] = True
                except StopIteration:
                    break
            remaining = parquet_iter
            while futures:
                for fut in as_completed(futures):
                    del futures[fut]
                    try:
                        futures[tex.submit(_load_parquet_file_io_only, next(remaining))] = True
                    except StopIteration:
                        pass
                    for sn, raw_rows in fut.result():
                        yield (sn, raw_rows, idx)
                        idx += 1
                        if args.max_sessions > 0 and idx >= args.max_sessions:
                            return
                    break

    ctx = _get_mp_context()
    worker_count = args.num_workers

    # sp_index 用 initializer 传给每个 worker 进程一次，不走每条 job 的 IPC
    _sp_index_shared = sp_index

    # 写盘线程：按小 row group 流式写同一个 parquet 文件，避免攒满 10 万 session 后瞬时复制。
    import threading, queue as _queue
    _write_q = _queue.Queue(maxsize=args.write_queue_size)
    _write_done = threading.Event()
    _writer_error = []

    def _writer_thread():
        nonlocal batch_idx
        writer = RollingParquetSessionWriter(
            args.output_dir,
            args.output_filename,
            batch_idx,
            args.sessions_per_batch,
            _SESSION_PARQUET_SCHEMA,
        )
        while True:
            item = _write_q.get()
            try:
                if item is None:
                    for path, count in writer.close():
                        print(f"\n  写出 {count} session → {os.path.basename(path)}")
                    batch_idx = writer.batch_idx
                    _write_done.set()
                    return
                rows = item
                for path, count in writer.write_rows(rows):
                    print(f"\n  写出 {count} session → {os.path.basename(path)}")
                batch_idx = writer.batch_idx
            except Exception as exc:
                _writer_error.append(exc)
                try:
                    writer.abort()
                finally:
                    _write_done.set()
                return

    wt = threading.Thread(target=_writer_thread, daemon=True)
    wt.start()

    total_written = 0
    pending_rows: list = []

    print(f"启动 {worker_count} 个 worker，流式处理所有 parquet...")
    with ctx.Pool(processes=worker_count,
                  initializer=_worker_init_sp,
                  initargs=(_sp_index_shared,)) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_job_build_and_run, _job_generator(), chunksize=args.pool_chunksize),
            desc="sessions"
        ):
            if _writer_error:
                raise RuntimeError("写盘线程失败") from _writer_error[0]

            pending_rows.append(result["row"])
            total_written += 1
            if len(pending_rows) >= args.row_group_sessions:
                _write_q.put(pending_rows)
                pending_rows = []

    if pending_rows:
        _write_q.put(pending_rows)

    _write_q.put(None)
    _write_done.wait()
    if _writer_error:
        raise RuntimeError("写盘线程失败") from _writer_error[0]

    print(f"全部完成，写入 {total_written} session，共 {batch_idx} 个输出 parquet")


if __name__ == "__main__":
    process_and_save_parquet(parse_args())
