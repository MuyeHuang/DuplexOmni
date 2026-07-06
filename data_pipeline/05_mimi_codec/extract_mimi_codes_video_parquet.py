#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interleaved E2E 训练用：从原始 jsonl 中提取所有可监督 assistant 音频，
逐段用 Mimi encode，再把 codes 按时间拼成一个完整 codec 文件，并输出
只带单个 `codec` 路径字段的训练 jsonl。

关键规则：
- 原始 `audios` 中 assistant self_audio 落在奇数位。
- 但 `audios[1]` 是无效的预热 self_audio，不用于监督。
- 真正参与监督的 assistant 音频从 `audios[3], audios[5], audios[7], ...` 开始。
- 每段音频必须先单独 encode，再按时间拼 code；不能先拼 wav 再 encode。
- 多卡路径按“音频段”切分任务，即使单条样本很长，也能把 8 张卡尽量打满。
- 每张卡内部会把同长度音频凑成 batch 再做 Mimi encode，以提高吞吐。

输出的 codec 文件包含：
- `codes`: LongTensor[16, T_total]
- `turn_boundaries`: [0, end_1, end_2, ..., end_n]
- `assistant_turn_indices`: 与每段 codec 对齐的 assistant 轮次索引（0-based assistant ordinal）
- `turn_audio_indices`: 对应原始 `audios` 的索引（例如 3, 5, 7, ...）
"""

import io
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.signal import resample
from tqdm import tqdm

from config import FRAME_MS, NUM_MIMI_LAYERS_USE, TARGET_AUDIO_MS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = Path(
    os.environ.get("INPUT_JSONL", "data/train_data.jsonl")
)
DEFAULT_MIMI_PATH = Path(
    os.environ.get("MIMI_PATH", "models/mimi")
)
DEFAULT_MIMI_CODES_DIR = PROJECT_ROOT / "mimi_codes_e2e_codec"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "train_e2e_with_codec.jsonl"
N_GPUS = 8
DEFAULT_BATCH_SIZE = 512
DEFAULT_LOAD_THREADS = 8
DEFAULT_PREFETCH_BATCHES = 4
DEFAULT_WRITE_THREADS = 2

HM = os.environ.get("VOICEAGENT_ROOT", ".")
DEFAULT_INPUT_PARQUET_DIR  = f"{HM}/api_generate/video_stream_output/pipeline_video/training_v9"
DEFAULT_OUTPUT_PARQUET_DIR = f"{HM}/api_generate/video_stream_output/pipeline_video/training_v9_codec"
DEFAULT_OUTPUT_FILENAME    = "train_e2e_v_codec"
SESSIONS_PER_OUTPUT_BATCH  = 10000
DEFAULT_READ_BATCH_ROWS = 512
DEFAULT_WRITE_ROW_GROUP_SESSIONS = 128
DEFAULT_WRITE_QUEUE_BATCHES = 8
DEFAULT_TASK_QUEUE_SIZE = 1024
DEFAULT_RESULT_QUEUE_BATCHES = 64
DEFAULT_PREFETCH_SHARDS = 4
DEFAULT_INPUT_POLL_SECONDS = 60
DEFAULT_SYSTEM_PROMPT_CHECK_SECONDS = 60

# 输出 schema：继承 build_training_parquet.py 的所有字段，追加 codec
_STEP3_OUTPUT_SCHEMA = pa.schema([
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
    pa.field("codec",         pa.large_binary()),
])

# images 字段（仅当输入 parquet 含 images 列时追加）
_IMAGES_PA_FIELD = pa.field("images", pa.list_(pa.struct([
    pa.field("bytes", pa.binary()),
    pa.field("path",  pa.string()),
])))


def _build_output_schema(has_images: bool) -> pa.Schema:
    if not has_images:
        return _STEP3_OUTPUT_SCHEMA
    return pa.schema(list(_STEP3_OUTPUT_SCHEMA) + [_IMAGES_PA_FIELD])


def _detect_has_images(input_parquet_dir: str) -> bool:
    """读第一个输入 parquet 的 schema，判断是否含 images 列。"""
    try:
        base = Path(input_parquet_dir)
        candidates = sorted(base.glob("*.parquet")) if base.is_dir() else [base]
        for p in candidates:
            if p.suffix == ".parquet" and not p.name.endswith(".tmp"):
                schema = pq.read_schema(str(p))
                return "images" in schema.names
    except Exception:
        pass
    return False


def _convert_content_to_placeholder(content):
    if isinstance(content, str):
        return content, [], [], []
    if not isinstance(content, list):
        return "", [], [], []
    text_parts = []
    images = []
    videos = []
    audios = []
    for part in content:
        if not isinstance(part, dict):
            if part is not None:
                text_parts.append(str(part))
            continue
        part_type = part.get("type")
        if part_type == "text":
            text_parts.append(part.get("text", ""))
        elif part_type == "image":
            text_parts.append("<image>")
            images.append(part.get("image"))
        elif part_type == "video":
            text_parts.append("<video>")
            videos.append(part.get("video"))
        elif part_type == "audio":
            text_parts.append("<audio>")
            audios.append(part.get("audio"))
    return "".join(text_parts), images, videos, audios


def canonicalize_item_format(item: Dict[str, Any]) -> Dict[str, Any]:
    out_item = dict(item)
    messages = item.get("messages", [])
    top_images = list(item.get("images") or [])
    top_videos = list(item.get("videos") or [])
    top_audios = list(item.get("audios") or [])
    if not top_audios and item.get("audio"):
        top_audios = [item["audio"]]
    if messages and any(isinstance(msg.get("content"), list) for msg in messages):
        images = []
        videos = []
        audios = []
        out_messages = []
        for msg in messages:
            content, msg_images, msg_videos, msg_audios = _convert_content_to_placeholder(msg.get("content", ""))
            out_messages.append({
                "role": msg.get("role", "user"),
                "content": content,
            })
            images.extend(p for p in msg_images if p)
            videos.extend(p for p in msg_videos if p)
            audios.extend(p for p in msg_audios if p)
        out_item["messages"] = out_messages
        out_item["images"] = images or top_images
        out_item["videos"] = videos or top_videos
        out_item["audios"] = audios or top_audios
    else:
        out_item["messages"] = list(messages or [])
        out_item["images"] = top_images
        out_item["videos"] = top_videos
        out_item["audios"] = top_audios
    out_item.pop("audio", None)
    return out_item


def _resolve_media_path(media_path: str, base_dir: Path) -> Optional[str]:
    if not media_path:
        return None
    path = Path(media_path)
    if path.is_absolute():
        return str(path)
    return str(base_dir / path)


def _assistant_message_count(messages: List[Dict[str, Any]]) -> int:
    return sum(1 for msg in messages if msg.get("role") == "assistant")


def _non_system_message_count(messages: List[Dict[str, Any]]) -> int:
    return sum(1 for msg in messages if msg.get("role") != "system")


def trim_last_user_assistant_round_for_training(item: Dict[str, Any]) -> Dict[str, Any]:
    out_item = dict(item)
    messages = list(out_item.get("messages") or [])
    audios = list(out_item.get("audios") or [])

    if len(audios) < 2:
        raise ValueError(f"Expect at least 2 audios before trimming, got {len(audios)}")

    last_assistant_idx = None
    last_user_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].get("role")
        if role == "system":
            continue
        if last_assistant_idx is None:
            if role != "assistant":
                raise ValueError(
                    f"Expect the last non-system message to be assistant, got role={role!r} at idx={idx}")
            last_assistant_idx = idx
            continue
        if role != "user":
            raise ValueError(
                f"Expect the message before the last assistant to be user, got role={role!r} at idx={idx}")
        last_user_idx = idx
        break

    if last_assistant_idx is None or last_user_idx is None:
        raise ValueError("Failed to find the trailing user-assistant round for trimming.")

    for idx in sorted([last_user_idx, last_assistant_idx], reverse=True):
        del messages[idx]

    audios = audios[:-2]
    out_item["messages"] = messages
    out_item["audios"] = audios

    non_system_messages = _non_system_message_count(messages)
    if non_system_messages != len(audios):
        raise ValueError(
            "After trimming the last round, the number of non-system messages must equal the number of audios: "
            f"non_system_messages={non_system_messages}, audios={len(audios)}")

    return out_item


def collect_supervised_assistant_audio_entries(item: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    item = canonicalize_item_format(item)
    audios = list(item.get("audios") or [])
    assistant_count = _assistant_message_count(item.get("messages", []))
    entries = []
    for audio_idx in range(3, len(audios), 2):
        # `audios[3]` supervises the first assistant turn because `audios[1]` is the delayed warm-up self_audio.
        assistant_turn_idx = (audio_idx - 3) // 2
        if assistant_turn_idx >= assistant_count:
            break
        resolved_path = _resolve_media_path(audios[audio_idx], base_dir)
        if resolved_path is None:
            continue
        entries.append({
            "assistant_turn_idx": assistant_turn_idx,
            "audio_idx": audio_idx,
            "audio_path": resolved_path,
        })
    return {
        "item": item,
        "entries": entries,
    }


def build_sample_list_from_jsonl(jsonl_path: Path, max_samples: int = None):
    logger.info("开始读取 jsonl: %s", jsonl_path)
    samples = []
    skip_reasons = {
        "json_error": 0,
        "no_supervised_audio": 0,
    }
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skip_reasons["json_error"] += 1
                continue
            collected = collect_supervised_assistant_audio_entries(item, jsonl_path.parent)
            if not collected["entries"]:
                skip_reasons["no_supervised_audio"] += 1
                continue
            samples.append({
                "idx": idx,
                "item": collected["item"],
                "assistant_audio_entries": collected["entries"],
            })
            if max_samples is not None and len(samples) >= max_samples:
                logger.info("已达到 max_samples=%d，停止解析", max_samples)
                break
    logger.info("有效样本数: %d，跳过统计: %s", len(samples), skip_reasons)
    return samples


def collect_supervised_assistant_audio_entries_from_parquet(audios: List[bytes]) -> Dict[str, Any]:
    """从 parquet audios 列表（list of bytes）中收集受监督的 assistant 音频。
    同原版逻辑：audios[3], audios[5], audios[7], ... 为受监督音频。
    """
    entries = []
    for audio_idx in range(3, len(audios), 2):
        audio_bytes = audios[audio_idx]
        if not audio_bytes:
            continue
        assistant_turn_idx = (audio_idx - 3) // 2
        entries.append({
            "assistant_turn_idx": assistant_turn_idx,
            "audio_idx": audio_idx,
            "audio_bytes": audio_bytes if isinstance(audio_bytes, bytes) else bytes(audio_bytes),
        })
    return {"entries": entries}


def _row_to_plain_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in row.items():
        if hasattr(value, "as_py"):
            value = value.as_py()
        out[key] = value
    return out


def _iter_parquet_rows(parquet_path: Path, batch_rows: int):
    pfile = pq.ParquetFile(str(parquet_path))
    for batch in pfile.iter_batches(batch_size=batch_rows):
        columns = batch.to_pydict()
        if not columns:
            continue
        first_col = next(iter(columns.values()))
        for row_idx in range(len(first_col)):
            yield {name: values[row_idx] for name, values in columns.items()}


def _is_stable_readable_parquet(path: Path, settle_seconds: float = 5.0) -> bool:
    if not path.is_file() or path.suffix != ".parquet" or path.name.endswith(".tmp"):
        return False
    try:
        size0 = path.stat().st_size
        if size0 <= 0:
            return False
        time.sleep(max(0.0, settle_seconds))
        size1 = path.stat().st_size
        if size0 != size1:
            return False
        pq.ParquetFile(str(path)).metadata
        return True
    except Exception:
        return False


def _discover_input_parquets(input_path: str, settle_seconds: float) -> List[Path]:
    base = Path(input_path)
    if base.is_file():
        return [base] if _is_stable_readable_parquet(base, settle_seconds=0.0) else []
    candidates = sorted(base.glob("*.parquet"))
    return [p for p in candidates if _is_stable_readable_parquet(p, settle_seconds=settle_seconds)]


def _input_has_unfinished_upstream(upstream_pattern: str) -> bool:
    if not upstream_pattern:
        return False
    try:
        return subprocess.run(
            ["pgrep", "-f", upstream_pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    except Exception:
        return False


def _first_system_prompt_ready(parquet_path: Path) -> bool:
    try:
        pfile = pq.ParquetFile(str(parquet_path))
        if pfile.metadata.num_rows <= 0:
            return False
        batch = next(pfile.iter_batches(batch_size=1, columns=["system_prompt"]), None)
        if batch is None or batch.num_rows <= 0:
            return False
        value = batch.column(0)[0].as_py()
        return isinstance(value, str) and bool(value.strip())
    except Exception:
        return False


def _row_group_ready_chunks(
    parquet_path: Path,
    done_sessions: set,
    batch_rows: int,
    system_prompt_check_seconds: float,
):
    base = parquet_path.name
    while not _first_system_prompt_ready(parquet_path):
        logger.info("WAIT system_prompt not ready: %s; sleep %.0fs", base, system_prompt_check_seconds)
        time.sleep(max(1.0, float(system_prompt_check_seconds)))

    logger.info("读取输入 parquet: %s", parquet_path)
    for row in _iter_parquet_rows(parquet_path, batch_rows=batch_rows):
        row = _row_to_plain_dict(row)
        sid = row.get("session_id")
        if sid in done_sessions:
            continue
        audios = list(row.get("audios") or [])
        collected = collect_supervised_assistant_audio_entries_from_parquet(audios)
        entries = collected["entries"]
        if not entries:
            continue
        yield row, entries


def _iter_parquet_ready_rows(
    parquet_path: Path,
    done_sessions: set,
    batch_rows: int,
    system_prompt_check_seconds: float,
):
    yield from _row_group_ready_chunks(
        parquet_path,
        done_sessions,
        batch_rows,
        system_prompt_check_seconds,
    )


def _load_done_sessions(output_dir: Path, output_filename: str) -> set:
    done_sessions: set = set()
    for p in sorted(output_dir.glob(f"{output_filename}_*.parquet")):
        if p.name.endswith(".tmp"):
            continue
        try:
            tbl = pq.read_table(str(p), columns=["session_id"])
        except Exception as exc:
            logger.warning("跳过不可读输出 parquet: %s err=%s", p, exc)
            continue
        done_sessions.update(tbl.column("session_id").to_pylist())
    return done_sessions


def _next_output_index(output_dir: Path, output_filename: str) -> int:
    max_idx = -1
    prefix = f"{output_filename}_"
    for path in output_dir.glob(f"{output_filename}_*.parquet"):
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix):]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx + 1


def _prepare_output_row(row: Dict[str, Any], codec_bytes: bytes) -> Dict[str, Any]:
    out = dict(row)
    audios = list(out["audios"]) if out.get("audios") is not None else []
    messages = list(out["messages"]) if out.get("messages") is not None else []
    if len(audios) >= 2 and len(messages) >= 2:
        audios = audios[:-2]
        messages = messages[:-2]
    out["audios"] = audios
    out["messages"] = messages
    out["codec"] = codec_bytes
    return out


def _worker_main_init() -> None:
    try:
        import torch as th
        th.set_num_threads(1)
        th.set_num_interop_threads(1)
    except Exception:
        pass


class _ParquetBatchWriter:
    def __init__(self, output_dir: Path, output_filename: str, row_group_sessions: int,
                 schema: pa.Schema = None):
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.row_group_sessions = max(1, int(row_group_sessions))
        self.schema = schema if schema is not None else _STEP3_OUTPUT_SCHEMA
        self.output_index = _next_output_index(output_dir, output_filename)
        self._writer = None
        self._rows = 0
        self._current_path = None
        self._current_tmp = None

    def _open_next(self) -> None:
        self._current_path = self.output_dir / f"{self.output_filename}_{self.output_index:06d}.parquet"
        self._current_tmp = self._current_path.with_name(self._current_path.name + ".tmp")
        if self._current_tmp.exists():
            self._current_tmp.unlink()
        self._writer = pq.ParquetWriter(str(self._current_tmp), self.schema, compression="snappy")

    def write_rows(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        if self._writer is None:
            self._open_next()
        written = 0
        for start in range(0, len(rows), self.row_group_sessions):
            chunk = rows[start:start + self.row_group_sessions]
            table = pa.Table.from_pylist(chunk, schema=self.schema)
            self._writer.write_table(table, row_group_size=self.row_group_sessions)
            self._rows += len(chunk)
            written += len(chunk)
        if self._writer is not None and self._rows >= SESSIONS_PER_OUTPUT_BATCH:
            self.close_current()
        return written

    def close_current(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        os.replace(str(self._current_tmp), str(self._current_path))
        logger.info("写出 %s: %d session", self._current_path.name, self._rows)
        self.output_index += 1
        self._writer = None
        self._rows = 0
        self._current_path = None
        self._current_tmp = None

    def close(self) -> None:
        self.close_current()


def build_sample_list_from_parquet(parquet_dir: str, done_sessions: set,
                                   max_samples: Optional[int] = None,
                                   batch_rows: int = DEFAULT_READ_BATCH_ROWS) -> List[Dict[str, Any]]:
    """流式读取 session 级 parquet，构建与 build_sample_list_from_jsonl 相同格式的 sample list。"""
    base = Path(parquet_dir)
    if base.is_file() and base.suffix == ".parquet":
        parquets = [base]
    else:
        parquets = sorted(base.glob("*.parquet")) or sorted(base.glob("**/*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"在 {parquet_dir} 找不到 *.parquet")
    samples = []
    skip_no_audio = 0
    for pfile in parquets:
        for row in _iter_parquet_rows(pfile, batch_rows=batch_rows):
            row = _row_to_plain_dict(row)
            sid = row["session_id"]
            if sid in done_sessions:
                continue
            audios = list(row["audios"])
            collected = collect_supervised_assistant_audio_entries_from_parquet(audios)
            if not collected["entries"]:
                skip_no_audio += 1
                continue
            samples.append({
                "idx": len(samples),
                "session_id": sid,
                "row": row,
                "audios": audios,
                "assistant_audio_entries": collected["entries"],
            })
            if max_samples and len(samples) >= max_samples:
                logger.info("已达 max_samples=%d，停止读取", max_samples)
                logger.info("有效样本数: %d，跳过无音频: %d", len(samples), skip_no_audio)
                return samples
    logger.info("有效样本数: %d，跳过无音频: %d", len(samples), skip_no_audio)
    return samples


def load_wav_scale_to_duration(audio_path, target_sr: int, target_duration_ms: int):
    # parquet 版只接受 bytes/bytearray：这里的数据源是裸 PCM，不走文件路径回退
    if isinstance(audio_path, (bytes, bytearray)):
        raw = bytes(audio_path)
        # 检测是否已有 WAV header（RIFF magic）；若无则视为裸 PCM int16 @ 24kHz mono
        if raw[:4] != b"RIFF":
            import wave
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(raw)
            buf.seek(0)
            audio_path = buf
        else:
            audio_path = io.BytesIO(raw)
    else:
        raise TypeError(f"parquet audio must be bytes/bytearray, got {type(audio_path).__name__}")

    import soundfile as sf

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    duration_sec = len(audio) / sr
    duration_ms = duration_sec * 1000
    target_ms = round(duration_ms / FRAME_MS) * FRAME_MS
    target_ms = min(target_ms, target_duration_ms)
    target_sec = target_ms / 1000.0
    target_n = int(round(target_sec * sr))
    if target_n <= 0:
        target_n = int(round((target_duration_ms / 1000.0) * sr))
    audio = resample(audio, target_n).astype(np.float32)
    return audio, sr


def normalize_codes_layout(codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim == 3:
        if codes.shape[0] != 1:
            raise ValueError(f"Unsupported Mimi code shape: {tuple(codes.shape)}")
        codes = codes[0]
    if codes.ndim != 2:
        raise ValueError(f"Unsupported Mimi code shape after squeeze: {tuple(codes.shape)}")
    if codes.shape[0] >= NUM_MIMI_LAYERS_USE:
        return codes[:NUM_MIMI_LAYERS_USE].contiguous().long()
    if codes.shape[1] >= NUM_MIMI_LAYERS_USE:
        return codes[:, :NUM_MIMI_LAYERS_USE].transpose(0, 1).contiguous().long()
    raise ValueError(f"Cannot normalize Mimi code layout: {tuple(codes.shape)}")


def _batched(items: Sequence[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _build_codec_payload(sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered_results = sorted(sample_results, key=lambda item: item["entry_order"])
    merged_codes = []
    turn_boundaries = [0]
    assistant_turn_indices = []
    turn_audio_indices = []
    turn_audio_paths = []
    for item in ordered_results:
        codes = torch.from_numpy(item["codes"]).contiguous().long()
        if codes.shape[-1] <= 0:
            continue
        merged_codes.append(codes)
        turn_boundaries.append(turn_boundaries[-1] + int(codes.shape[-1]))
        assistant_turn_indices.append(int(item["assistant_turn_idx"]))
        turn_audio_indices.append(int(item["audio_idx"]))
        turn_audio_paths.append(item["audio_path"])
    if not merged_codes:
        raise RuntimeError("No valid Mimi codes collected while building codec payload.")
    return {
        "codes": torch.cat(merged_codes, dim=-1).long(),
        "turn_boundaries": torch.tensor(turn_boundaries, dtype=torch.long),
        "assistant_turn_indices": torch.tensor(assistant_turn_indices, dtype=torch.long),
        "turn_audio_indices": torch.tensor(turn_audio_indices, dtype=torch.long),
        "turn_audio_paths": turn_audio_paths,
    }


def _build_pending_segment_tasks(sample_list: List[Dict[str, Any]], codes_dir: Path,
                                 overwrite: bool) -> Tuple[List[Dict[str, Any]], Dict[int, int], int]:
    pending_tasks: List[Dict[str, Any]] = []
    sample_task_counts: Dict[int, int] = {}
    skipped_existing = 0
    for sample in sample_list:
        sample_idx = int(sample["idx"])
        out_path = codes_dir / f"codec_{sample_idx:05d}.pt"
        if out_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        count = 0
        for entry_order, entry in enumerate(sample.get("assistant_audio_entries", [])):
            audio_path = entry.get("audio_path")
            if not audio_path:
                continue
            pending_tasks.append({
                "sample_idx": sample_idx,
                "entry_order": entry_order,
                "assistant_turn_idx": int(entry["assistant_turn_idx"]),
                "audio_idx": int(entry["audio_idx"]),
                "audio_path": audio_path,
            })
            count += 1
        if count > 0:
            sample_task_counts[sample_idx] = count
    return pending_tasks, sample_task_counts, skipped_existing


def _distribute_segment_tasks(tasks: List[Dict[str, Any]], n_gpus: int) -> Tuple[List[List[Dict[str, Any]]], List[int]]:
    chunks = [[] for _ in range(n_gpus)]
    for idx, task in enumerate(tasks):
        chunks[idx % n_gpus].append(task)
    loads = [len(chunk) for chunk in chunks]
    return chunks, loads


def _emit_task_failed(result_queue, task: Dict[str, Any], error: str) -> None:
    result_queue.put({
        "type": "task_failed",
        "sample_idx": int(task["sample_idx"]),
        "entry_order": int(task["entry_order"]),
        "audio_idx": int(task["audio_idx"]),
        "audio_path": task.get("audio_path", "<bytes>"),
        "error": error,
    })


def worker_encode_segments(rank: int,
                           segment_tasks: List[Dict[str, Any]],
                           mimi_path: Path,
                           batch_size: int,
                           result_queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch as th
    from transformers import AutoFeatureExtractor, MimiModel

    try:
        try:
            th.set_num_threads(1)
            th.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if not segment_tasks:
            logger.info("[rank %d] 无分配音频段，退出", rank)
            result_queue.put({"type": "worker_done", "rank": rank})
            return
        if th.cuda.is_available():
            th.backends.cuda.matmul.allow_tf32 = True
            th.backends.cudnn.benchmark = True
        batch_size = max(int(batch_size), 1)
        load_workers = max(1, min(DEFAULT_LOAD_THREADS, len(segment_tasks)))
        prefetch_limit = max(batch_size * DEFAULT_PREFETCH_BATCHES, batch_size)
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        feature_extractor = AutoFeatureExtractor.from_pretrained(str(mimi_path))
        mimi_model = MimiModel.from_pretrained(str(mimi_path))
        target_sr = feature_extractor.sampling_rate
        mimi_model = mimi_model.to(device)
        mimi_model.eval()
        logger.info(
            "[rank %d] 分配音频段=%d, batch_size=%d, load_workers=%d, prefetch_limit=%d",
            rank,
            len(segment_tasks),
            batch_size,
            load_workers,
            prefetch_limit,
        )

        def _load_task_audio(task: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray]:
            audio_src = task.get("audio_bytes")
            if audio_src is None:
                raise TypeError(f"parquet task missing audio_bytes for sample_idx={task.get('sample_idx')}")
            audio, _sr = load_wav_scale_to_duration(audio_src, target_sr, TARGET_AUDIO_MS)
            return task, audio

        def _encode_audio_batch(batch_items: List[Tuple[Dict[str, Any], np.ndarray]]) -> List[Dict[str, Any]]:
            audio_batch = [audio for _, audio in batch_items]
            inputs = feature_extractor(raw_audio=audio_batch, sampling_rate=target_sr, return_tensors="pt", padding=True)
            audio_tensor = inputs["input_values"].to(device=device, dtype=mimi_model.dtype, non_blocking=True)
            if audio_tensor.ndim == 2:
                audio_tensor = audio_tensor.unsqueeze(1)
            with th.inference_mode():
                codes_batch = mimi_model.encode(audio_tensor).audio_codes.cpu()
            encoded_items = []
            for batch_pos, (task, _audio) in enumerate(batch_items):
                codes = normalize_codes_layout(codes_batch[batch_pos:batch_pos + 1]).cpu().numpy()
                encoded_items.append({
                    "sample_idx": int(task["sample_idx"]),
                    "entry_order": int(task["entry_order"]),
                    "assistant_turn_idx": int(task["assistant_turn_idx"]),
                    "audio_idx": int(task["audio_idx"]),
                    "audio_path": task.get("audio_path", "<bytes>"),
                    "codes": codes,
                })
            return encoded_items

        def _flush_ready_batches(length_buckets: Dict[int, List[Tuple[Dict[str, Any], np.ndarray]]], force: bool = False) -> int:
            processed = 0
            ready_lengths = [
                bucket_len
                for bucket_len, bucket_items in length_buckets.items()
                if len(bucket_items) >= batch_size or (force and bucket_items)
            ]
            for bucket_len in sorted(ready_lengths):
                bucket_items = length_buckets[bucket_len]
                while len(bucket_items) >= batch_size or (force and bucket_items):
                    batch_items = bucket_items[:batch_size]
                    del bucket_items[:len(batch_items)]
                    encoded_items = _encode_audio_batch(batch_items)
                    result_queue.put({"type": "encoded_batch", "items": encoded_items})
                    batch_len = len(batch_items)
                    progress.update(batch_len)
                    processed += batch_len
                if not bucket_items:
                    length_buckets.pop(bucket_len, None)
            return processed

        progress = tqdm(total=len(segment_tasks), desc=f"Mimi rank{rank}", position=rank)
        length_buckets: Dict[int, List[Tuple[Dict[str, Any], np.ndarray]]] = defaultdict(list)
        future_to_task: Dict[Future, Dict[str, Any]] = {}
        task_index = 0

        with ThreadPoolExecutor(max_workers=load_workers) as load_pool:
            def _submit_prefetch() -> None:
                nonlocal task_index
                while task_index < len(segment_tasks) and len(future_to_task) < prefetch_limit:
                    task = segment_tasks[task_index]
                    task_index += 1
                    future = load_pool.submit(_load_task_audio, task)
                    future_to_task[future] = task

            _submit_prefetch()
            while future_to_task:
                _flush_ready_batches(length_buckets, force=False)
                done_futures, _ = wait(tuple(future_to_task.keys()), return_when=FIRST_COMPLETED)
                for future in done_futures:
                    task = future_to_task.pop(future)
                    try:
                        loaded_task, audio = future.result()
                        length_buckets[len(audio)].append((loaded_task, audio))
                    except Exception as exc:
                        _emit_task_failed(result_queue, task, str(exc))
                        progress.update(1)
                _submit_prefetch()
            _flush_ready_batches(length_buckets, force=True)
        progress.close()
        logger.info("[rank %d] 音频段编码完成", rank)
    finally:
        result_queue.put({"type": "worker_done", "rank": rank})


def worker_encode_segments_streaming(rank: int,
                                     task_queue,
                                     mimi_path: Path,
                                     batch_size: int,
                                     result_queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch as th
    from transformers import AutoFeatureExtractor, MimiModel

    try:
        try:
            th.set_num_threads(1)
            th.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if th.cuda.is_available():
            th.backends.cuda.matmul.allow_tf32 = True
            th.backends.cudnn.benchmark = True
        batch_size = max(int(batch_size), 1)
        load_workers = max(1, DEFAULT_LOAD_THREADS)
        prefetch_limit = max(batch_size * DEFAULT_PREFETCH_BATCHES, batch_size)
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        feature_extractor = AutoFeatureExtractor.from_pretrained(str(mimi_path))
        mimi_model = MimiModel.from_pretrained(str(mimi_path))
        target_sr = feature_extractor.sampling_rate
        mimi_model = mimi_model.to(device)
        mimi_model.eval()
        logger.info(
            "[rank %d] streaming worker ready, batch_size=%d, load_workers=%d, prefetch_limit=%d",
            rank,
            batch_size,
            load_workers,
            prefetch_limit,
        )

        progress = tqdm(desc=f"Mimi rank{rank}", position=rank)
        length_buckets: Dict[int, List[Tuple[Dict[str, Any], np.ndarray]]] = defaultdict(list)
        future_to_task: Dict[Future, Dict[str, Any]] = {}
        pending_tasks = []
        stop_seen = False

        def _load_task_audio(task: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray]:
            audio_src = task.get("audio_bytes") or task.get("audio_path")
            audio, _sr = load_wav_scale_to_duration(audio_src, target_sr, TARGET_AUDIO_MS)
            return task, audio

        def _encode_audio_batch(batch_items: List[Tuple[Dict[str, Any], np.ndarray]]) -> List[Dict[str, Any]]:
            audio_batch = [audio for _, audio in batch_items]
            inputs = feature_extractor(raw_audio=audio_batch, sampling_rate=target_sr, return_tensors="pt", padding=True)
            audio_tensor = inputs["input_values"].to(device=device, dtype=mimi_model.dtype, non_blocking=True)
            if audio_tensor.ndim == 2:
                audio_tensor = audio_tensor.unsqueeze(1)
            with th.inference_mode():
                codes_batch = mimi_model.encode(audio_tensor).audio_codes.cpu()
            encoded_items = []
            for batch_pos, (task, _audio) in enumerate(batch_items):
                codes = normalize_codes_layout(codes_batch[batch_pos:batch_pos + 1]).cpu().numpy()
                encoded_items.append({
                    "sample_idx": int(task["sample_idx"]),
                    "entry_order": int(task["entry_order"]),
                    "assistant_turn_idx": int(task["assistant_turn_idx"]),
                    "audio_idx": int(task["audio_idx"]),
                    "audio_path": task.get("audio_path", "<bytes>"),
                    "codes": codes,
                })
            return encoded_items

        def _flush_ready_batches(force: bool = False) -> int:
            processed = 0
            ready_lengths = [
                bucket_len
                for bucket_len, bucket_items in length_buckets.items()
                if len(bucket_items) >= batch_size or (force and bucket_items)
            ]
            for bucket_len in sorted(ready_lengths):
                bucket_items = length_buckets[bucket_len]
                while len(bucket_items) >= batch_size or (force and bucket_items):
                    batch_items = bucket_items[:batch_size]
                    del bucket_items[:len(batch_items)]
                    encoded_items = _encode_audio_batch(batch_items)
                    result_queue.put({"type": "encoded_batch", "items": encoded_items})
                    batch_len = len(batch_items)
                    progress.update(batch_len)
                    processed += batch_len
                if not bucket_items:
                    length_buckets.pop(bucket_len, None)
            return processed

        with ThreadPoolExecutor(max_workers=load_workers) as load_pool:
            while not stop_seen or pending_tasks or future_to_task or length_buckets:
                while not stop_seen and len(pending_tasks) + len(future_to_task) < prefetch_limit:
                    task = task_queue.get()
                    if task is None:
                        stop_seen = True
                        break
                    pending_tasks.append(task)
                while pending_tasks and len(future_to_task) < prefetch_limit:
                    task = pending_tasks.pop(0)
                    future = load_pool.submit(_load_task_audio, task)
                    future_to_task[future] = task
                _flush_ready_batches(force=False)
                if future_to_task:
                    done_futures, _ = wait(tuple(future_to_task.keys()), timeout=0.1,
                                           return_when=FIRST_COMPLETED)
                    for future in done_futures:
                        task = future_to_task.pop(future)
                        try:
                            loaded_task, audio = future.result()
                            length_buckets[len(audio)].append((loaded_task, audio))
                        except Exception as exc:
                            _emit_task_failed(result_queue, task, str(exc))
                            progress.update(1)
                elif pending_tasks:
                    continue
                else:
                    _flush_ready_batches(force=True)
        progress.close()
        logger.info("[rank %d] streaming 音频段编码完成", rank)
    finally:
        result_queue.put({"type": "worker_done", "rank": rank})


def _write_codec_file(sample_idx: int, sample_results: List[Dict[str, Any]], codes_dir: Path) -> None:
    payload = _build_codec_payload(sample_results)
    out_path = codes_dir / f"codec_{sample_idx:05d}.pt"
    torch.save(payload, out_path)


def _build_pending_segment_tasks_parquet(sample_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """从 parquet sample list 构建任务：audio_bytes 替代 audio_path。"""
    pending_tasks: List[Dict[str, Any]] = []
    sample_task_counts: Dict[int, int] = {}
    for sample in sample_list:
        sample_idx = int(sample["idx"])
        count = 0
        for entry_order, entry in enumerate(sample.get("assistant_audio_entries", [])):
            audio_bytes = entry.get("audio_bytes")
            if not audio_bytes:
                continue
            pending_tasks.append({
                "sample_idx":         sample_idx,
                "entry_order":        entry_order,
                "assistant_turn_idx": int(entry["assistant_turn_idx"]),
                "audio_idx":          int(entry["audio_idx"]),
                "audio_bytes":        audio_bytes,   # bytes，不再是路径
            })
            count += 1
        if count > 0:
            sample_task_counts[sample_idx] = count
    return pending_tasks, sample_task_counts


def _codec_payload_to_bytes(payload: Dict[str, Any]) -> bytes:
    """将 codec payload dict 序列化为 bytes（供 parquet large_binary 存储）。"""
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()



def _consume_segment_results(result_queue,
                             n_workers: int,
                             sample_task_counts: Dict[int, int],
                             codes_dir: Path) -> Dict[str, int]:
    pending_counts = dict(sample_task_counts)
    sample_results: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    sample_failures: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    completed_samples = 0
    failed_segments = 0
    done_workers = 0
    logged_encoded_payload = False
    write_workers = max(1, min(DEFAULT_WRITE_THREADS, max(len(sample_task_counts), 1)))
    write_futures: Dict[Future, int] = {}

    def _collect_finished_writes(wait_for_all: bool = False) -> None:
        nonlocal completed_samples
        if not write_futures:
            return
        if wait_for_all:
            done_futures = list(write_futures.keys())
            wait(tuple(done_futures))
        else:
            done_futures = [future for future in write_futures if future.done()]
        for future in done_futures:
            sample_idx = write_futures.pop(future)
            future.result()
            completed_samples += 1
            logger.info("sample=%d codec 已写入", sample_idx)

    with ThreadPoolExecutor(max_workers=write_workers) as write_pool:
        def _finalize_sample(sample_idx: int) -> None:
            results = sample_results.pop(sample_idx, [])
            pending_counts.pop(sample_idx, None)
            if sample_failures.get(sample_idx):
                logger.error("sample=%d 存在失败音频段，拒绝写入 codec。", sample_idx)
                return
            if not results:
                return
            future = write_pool.submit(_write_codec_file, sample_idx, results, codes_dir)
            write_futures[future] = sample_idx

        while done_workers < n_workers:
            _collect_finished_writes(wait_for_all=False)
            message = result_queue.get()
            msg_type = message.get("type")
            if msg_type == "worker_done":
                done_workers += 1
                continue
            if msg_type == "encoded_batch":
                if not logged_encoded_payload and message.get("items"):
                    first_codes = message["items"][0]["codes"]
                    logger.info("result_queue encoded payload type=%s, shape=%s", type(first_codes).__name__,
                                getattr(first_codes, "shape", None))
                    logged_encoded_payload = True
                touched_samples = set()
                for item in message.get("items", []):
                    sample_idx = int(item["sample_idx"])
                    sample_results[sample_idx].append(item)
                    if sample_idx in pending_counts:
                        pending_counts[sample_idx] -= 1
                    touched_samples.add(sample_idx)
                for sample_idx in touched_samples:
                    if pending_counts.get(sample_idx) == 0:
                        _finalize_sample(sample_idx)
                continue
            if msg_type == "task_failed":
                failed_segments += 1
                sample_idx = int(message["sample_idx"])
                sample_failures[sample_idx].append(message)
                if sample_idx in pending_counts:
                    pending_counts[sample_idx] -= 1
                logger.warning("跳过音频段 sample=%d audio_idx=%s path=%s err=%s", sample_idx, message.get("audio_idx"),
                               message.get("audio_path"), message.get("error"))
                if pending_counts.get(sample_idx) == 0:
                    _finalize_sample(sample_idx)
                continue
            raise RuntimeError(f"Unknown worker message type: {msg_type}")

        for sample_idx in list(pending_counts.keys()):
            if pending_counts[sample_idx] == 0:
                _finalize_sample(sample_idx)
        _collect_finished_writes(wait_for_all=True)
    incomplete = sum(1 for v in pending_counts.values() if v > 0)
    if incomplete > 0:
        raise RuntimeError(f"There are {incomplete} samples with unfinished segment tasks after all workers exited.")
    if failed_segments > 0:
        first_failure = None
        for failures in sample_failures.values():
            if failures:
                first_failure = failures[0]
                break
        logger.warning(
            "Mimi encode: %d segments failed and were skipped. "
            f"first_failure_sample={first_failure.get('sample_idx') if first_failure else 'na'}, "
            f"audio_idx={first_failure.get('audio_idx') if first_failure else 'na'}, "
            f"path={first_failure.get('audio_path') if first_failure else 'na'}, "
            f"error={first_failure.get('error') if first_failure else 'na'}",
            failed_segments)
    return {
        "completed_samples": completed_samples,
        "failed_segments": failed_segments,
    }


def _consume_segment_results_parquet(result_queue, n_workers: int,
                                     sample_task_counts: Dict[int, int]) -> Dict[int, bytes]:
    """与 _consume_segment_results 相同逻辑，但不写文件，直接返回 {sample_idx: codec_bytes}。"""
    pending_counts    = dict(sample_task_counts)
    sample_results:   Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    sample_failures:  Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    codec_bytes_map:  Dict[int, bytes] = {}
    failed_segments   = done_workers = 0
    logged_payload    = False

    def _finalize_sample(sample_idx: int) -> None:
        results = sample_results.pop(sample_idx, [])
        pending_counts.pop(sample_idx, None)
        if sample_failures.get(sample_idx):
            logger.error("sample=%d 存在失败音频段，跳过。", sample_idx)
            return
        if not results:
            return
        try:
            payload = _build_codec_payload(results)
            codec_bytes_map[sample_idx] = _codec_payload_to_bytes(payload)
            logger.debug("sample=%d codec bytes 已生成", sample_idx)
        except Exception as e:
            logger.error("sample=%d codec 构建失败: %s", sample_idx, e)

    while done_workers < n_workers:
        message = result_queue.get()
        msg_type = message.get("type")
        if msg_type == "worker_done":
            done_workers += 1
            continue
        if msg_type == "encoded_batch":
            if not logged_payload and message.get("items"):
                first_codes = message["items"][0]["codes"]
                logger.info("encoded payload shape=%s", getattr(first_codes, "shape", None))
                logged_payload = True
            touched = set()
            for item in message.get("items", []):
                sidx = int(item["sample_idx"])
                sample_results[sidx].append(item)
                if sidx in pending_counts:
                    pending_counts[sidx] -= 1
                touched.add(sidx)
            for sidx in touched:
                if pending_counts.get(sidx) == 0:
                    _finalize_sample(sidx)
            continue
        if msg_type == "task_failed":
            failed_segments += 1
            sidx = int(message["sample_idx"])
            sample_failures[sidx].append(message)
            if sidx in pending_counts:
                pending_counts[sidx] -= 1
            logger.warning("跳过音频段 sample=%d audio_idx=%s err=%s",
                           sidx, message.get("audio_idx"), message.get("error"))
            if pending_counts.get(sidx) == 0:
                _finalize_sample(sidx)
            continue
        raise RuntimeError(f"Unknown worker message type: {msg_type}")

    for sidx in list(pending_counts.keys()):
        if pending_counts[sidx] == 0:
            _finalize_sample(sidx)

    if failed_segments:
        logger.warning("共 %d 个音频段 encode 失败并跳过", failed_segments)
    return codec_bytes_map


def _consume_segment_results_streaming(
    result_queue,
    n_workers: int,
    sample_task_counts: Dict[int, int],
    sample_meta: Dict[int, Dict[str, Any]],
    reader_done: threading.Event,
    reader_error: List[BaseException],
    output_dir: Path,
    output_filename: str,
    row_group_sessions: int,
    write_queue_batches: int,
    schema: pa.Schema = None,
) -> Dict[str, int]:
    pending_counts: Dict[int, int] = {}
    sample_results: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    sample_failures: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    done_workers = 0
    failed_segments = 0
    completed_samples = 0
    completed_rows = 0
    pending_rows: List[Dict[str, Any]] = []
    write_queue: "queue.Queue[Optional[Tuple[Path, List[Dict[str, Any]]]]]" = queue.Queue(
        maxsize=max(1, int(write_queue_batches))
    )
    writer_done = threading.Event()
    writer_error: List[BaseException] = []
    writer = _ParquetBatchWriter(output_dir, output_filename, row_group_sessions, schema=schema)

    def _writer_loop() -> None:
        nonlocal completed_rows
        try:
            while True:
                item = write_queue.get()
                if item is None:
                    break
                rows = item
                completed_rows += writer.write_rows(rows)
        except BaseException as exc:
            writer_error.append(exc)
        finally:
            try:
                writer.close()
            except BaseException as exc:
                writer_error.append(exc)
            writer_done.set()

    def _finalize_sample(sample_idx: int) -> None:
        nonlocal completed_samples, pending_rows
        results = sample_results.pop(sample_idx, [])
        pending_counts.pop(sample_idx, None)
        sample_task_counts.pop(sample_idx, None)
        if sample_failures.get(sample_idx):
            logger.error("sample=%d 存在失败音频段，跳过写入。", sample_idx)
            sample_failures.pop(sample_idx, None)
            sample_meta.pop(sample_idx, None)
            return
        if not results:
            sample_failures.pop(sample_idx, None)
            sample_meta.pop(sample_idx, None)
            return
        try:
            payload = _build_codec_payload(results)
            codec_bytes = _codec_payload_to_bytes(payload)
            meta = sample_meta.pop(sample_idx, None)
            if meta is None:
                return
            row = _prepare_output_row(meta["row"], codec_bytes)
            pending_rows.append(row)
            completed_samples += 1
            if len(pending_rows) >= row_group_sessions:
                write_queue.put(pending_rows)
                pending_rows = []
            sample_failures.pop(sample_idx, None)
        except Exception as exc:
            logger.error("sample=%d codec 构建失败: %s", sample_idx, exc)
            sample_failures.pop(sample_idx, None)
            sample_meta.pop(sample_idx, None)

    writer_thread = threading.Thread(target=_writer_loop, name="codec-writer", daemon=True)
    writer_thread.start()

    try:
        while done_workers < n_workers:
            if writer_error:
                raise writer_error[0]
            if reader_error:
                raise reader_error[0]
            message = result_queue.get()
            msg_type = message.get("type")
            if msg_type == "worker_done":
                done_workers += 1
                continue
            if msg_type == "encoded_batch":
                touched = set()
                for item in message.get("items", []):
                    sample_idx = int(item["sample_idx"])
                    if sample_idx not in pending_counts:
                        expected = sample_task_counts.get(sample_idx)
                        if expected is not None:
                            pending_counts[sample_idx] = int(expected)
                    sample_results[sample_idx].append(item)
                    if sample_idx in pending_counts:
                        pending_counts[sample_idx] -= 1
                    touched.add(sample_idx)
                for sample_idx in touched:
                    if pending_counts.get(sample_idx) == 0:
                        _finalize_sample(sample_idx)
                continue
            if msg_type == "task_failed":
                failed_segments += 1
                sample_idx = int(message["sample_idx"])
                sample_failures[sample_idx].append(message)
                if sample_idx in pending_counts:
                    pending_counts[sample_idx] -= 1
                elif sample_idx in sample_task_counts:
                    pending_counts[sample_idx] = int(sample_task_counts[sample_idx]) - 1
                if pending_counts.get(sample_idx) == 0:
                    _finalize_sample(sample_idx)
                continue
            raise RuntimeError(f"Unknown worker message type: {msg_type}")

        if reader_error:
            raise reader_error[0]
        for sample_idx in list(pending_counts.keys()):
            if pending_counts[sample_idx] == 0:
                _finalize_sample(sample_idx)

        if pending_rows:
            write_queue.put(pending_rows)
            pending_rows = []

        write_queue.put(None)
        writer_done.wait()
        if writer_error:
            raise writer_error[0]
    finally:
        if not writer_done.is_set():
            try:
                write_queue.put_nowait(None)
            except Exception:
                pass
            writer_done.wait(timeout=10)

    incomplete = sum(1 for v in pending_counts.values() if v > 0)
    if incomplete > 0:
        raise RuntimeError(f"There are {incomplete} samples with unfinished segment tasks after all workers exited.")
    if failed_segments > 0:
        logger.warning("Mimi encode: %d segments failed and were skipped.", failed_segments)
    return {
        "completed_samples": completed_samples,
        "completed_rows": completed_rows,
        "failed_segments": failed_segments,
    }


def _read_parquet_tasks(
    parquet_files: List[Path],
    done_sessions: set,
    max_samples: Optional[int],
    batch_rows: int,
    task_queues: List[Any],
    sample_task_counts: Dict[int, int],
    sample_meta: Dict[int, Dict[str, Any]],
    reader_done: threading.Event,
    reader_error: List[BaseException],
) -> None:
    try:
        sample_idx = 0
        task_idx = 0
        skip_done = 0
        skip_no_audio = 0
        for pfile in parquet_files:
            logger.info("读取输入 parquet: %s", pfile)
            for row in _iter_parquet_rows(pfile, batch_rows=batch_rows):
                row = _row_to_plain_dict(row)
                sid = row.get("session_id")
                if sid in done_sessions:
                    skip_done += 1
                    continue
                audios = list(row.get("audios") or [])
                collected = collect_supervised_assistant_audio_entries_from_parquet(audios)
                entries = collected["entries"]
                if not entries:
                    skip_no_audio += 1
                    continue
                current_idx = sample_idx
                sample_idx += 1
                sample_meta[current_idx] = {
                    "session_id": sid,
                    "row": row,
                }
                sample_task_counts[current_idx] = len(entries)
                for entry_order, entry in enumerate(entries):
                    task = {
                        "sample_idx": current_idx,
                        "entry_order": entry_order,
                        "assistant_turn_idx": int(entry["assistant_turn_idx"]),
                        "audio_idx": int(entry["audio_idx"]),
                        "audio_bytes": entry["audio_bytes"],
                    }
                    task_queues[task_idx % len(task_queues)].put(task)
                    task_idx += 1
                if max_samples and sample_idx >= max_samples:
                    logger.info("已达 max_samples=%d，停止读取", max_samples)
                    raise StopIteration
    except StopIteration:
        pass
    except BaseException as exc:
        reader_error.append(exc)
    finally:
        for q in task_queues:
            q.put(None)
        logger.info(
            "reader done: samples=%d tasks=%d skip_done=%d skip_no_audio=%d",
            len(sample_meta),
            sum(sample_task_counts.values()),
            skip_done if "skip_done" in locals() else 0,
            skip_no_audio if "skip_no_audio" in locals() else 0,
        )
        reader_done.set()


def _read_parquet_tasks_prefetch(
    input_path: str,
    done_sessions: set,
    max_samples: Optional[int],
    batch_rows: int,
    prefetch_shards: int,
    input_poll_seconds: float,
    input_settle_seconds: float,
    system_prompt_check_seconds: float,
    upstream_pattern: str,
    task_queues: List[Any],
    sample_task_counts: Dict[int, int],
    sample_meta: Dict[int, Dict[str, Any]],
    reader_done: threading.Event,
    reader_error: List[BaseException],
) -> None:
    reader_threads = max(1, int(prefetch_shards))
    file_queue: "queue.Queue[Optional[Path]]" = queue.Queue(maxsize=reader_threads)
    stop_prefetch = threading.Event()
    manager_error: List[BaseException] = []
    file_errors: List[BaseException] = []
    seen_paths: set = set()
    seen_lock = threading.Lock()
    sample_lock = threading.Lock()
    task_lock = threading.Lock()
    sample_idx_box = [0]
    task_idx_box = [0]
    processed_files = []

    def _next_sample_idx() -> int:
        with sample_lock:
            sample_idx = sample_idx_box[0]
            sample_idx_box[0] += 1
            return sample_idx

    def _dispatch_task(task: Dict[str, Any]) -> None:
        with task_lock:
            task_queue_idx = task_idx_box[0] % len(task_queues)
            task_idx_box[0] += 1
        task_queues[task_queue_idx].put(task)

    def _manager_loop() -> None:
        try:
            while not stop_prefetch.is_set():
                parquet_files = _discover_input_parquets(input_path, settle_seconds=input_settle_seconds)
                found_new = False
                for parquet_path in parquet_files:
                    with seen_lock:
                        if parquet_path in seen_paths:
                            continue
                        seen_paths.add(parquet_path)
                    found_new = True
                    file_queue.put(parquet_path)
                if Path(input_path).is_file():
                    break
                if not _input_has_unfinished_upstream(upstream_pattern):
                    break
                if not found_new:
                    logger.info("No new input parquet; upstream still running; sleep %.0fs", input_poll_seconds)
                time.sleep(max(1.0, float(input_poll_seconds)))
        except BaseException as exc:
            manager_error.append(exc)
        finally:
            for _ in range(reader_threads):
                while True:
                    try:
                        file_queue.put(None, timeout=1)
                        break
                    except queue.Full:
                        if stop_prefetch.is_set():
                            try:
                                file_queue.get_nowait()
                            except queue.Empty:
                                pass

    def _file_reader_worker(worker_id: int) -> None:
        local_files = 0
        try:
            while not stop_prefetch.is_set():
                parquet_path = file_queue.get()
                if parquet_path is None:
                    break
                local_files += 1
                processed_files.append(str(parquet_path))
                logger.info("读取输入 parquet[%d]: %s", worker_id, parquet_path)
                for row, entries in _iter_parquet_ready_rows(
                    parquet_path,
                    done_sessions,
                    batch_rows,
                    system_prompt_check_seconds,
                ):
                    if stop_prefetch.is_set():
                        break
                    current_idx = _next_sample_idx()
                    if max_samples and current_idx >= max_samples:
                        logger.info("已达 max_samples=%d，停止读取", max_samples)
                        stop_prefetch.set()
                        break
                    sample_meta[current_idx] = {
                        "session_id": row.get("session_id"),
                        "row": row,
                    }
                    sample_task_counts[current_idx] = len(entries)
                    for entry_order, entry in enumerate(entries):
                        task = {
                            "sample_idx": current_idx,
                            "entry_order": entry_order,
                            "assistant_turn_idx": int(entry["assistant_turn_idx"]),
                            "audio_idx": int(entry["audio_idx"]),
                            "audio_bytes": entry["audio_bytes"],
                        }
                        _dispatch_task(task)
                if stop_prefetch.is_set():
                    break
        except BaseException as exc:
            file_errors.append(exc)
            stop_prefetch.set()
        finally:
            logger.info("file reader %d done, files=%d", worker_id, local_files)

    manager_thread = threading.Thread(target=_manager_loop, name="codec-file-manager", daemon=True)
    manager_thread.start()
    readers = [
        threading.Thread(target=_file_reader_worker, args=(idx,), name=f"codec-file-reader-{idx}", daemon=True)
        for idx in range(reader_threads)
    ]
    for thread in readers:
        thread.start()

    try:
        manager_thread.join()
        for thread in readers:
            thread.join()
    except BaseException as exc:
        reader_error.append(exc)
        stop_prefetch.set()
    finally:
        stop_prefetch.set()
        if manager_thread.is_alive():
            manager_thread.join(timeout=5)
        for thread in readers:
            if thread.is_alive():
                thread.join(timeout=5)
        if manager_error:
            reader_error.append(manager_error[0])
        if file_errors:
            reader_error.append(file_errors[0])
        for q in task_queues:
            q.put(None)
        logger.info(
            "reader done: samples=%d tasks=%d files=%d",
            len(sample_meta),
            sum(sample_task_counts.values()),
            len(processed_files),
        )
        reader_done.set()


def run_streaming_parquet_pipeline(args, output_dir: Path, mimi_path: Path) -> None:
    import multiprocessing as mp

    parquet_files = _discover_input_parquets(args.input_parquet_dir, settle_seconds=args.input_settle_seconds)
    while not parquet_files:
        if not _input_has_unfinished_upstream(args.upstream_pattern):
            raise FileNotFoundError(f"在 {args.input_parquet_dir} 找不到稳定可读的 *.parquet")
        logger.info("No input parquet yet; upstream still running; sleep %.0fs", args.input_poll_seconds)
        time.sleep(max(1.0, float(args.input_poll_seconds)))
        parquet_files = _discover_input_parquets(args.input_parquet_dir, settle_seconds=args.input_settle_seconds)

    has_images = _detect_has_images(args.input_parquet_dir)
    output_schema = _build_output_schema(has_images)
    logger.info("输入 parquet %s images 列，输出 schema 字段: %s",
                "含" if has_images else "不含", output_schema.names)

    done_sessions: set = set()
    if args.resume:
        done_sessions = _load_done_sessions(output_dir, args.output_filename)
        if done_sessions:
            logger.info("断点续传：已跳过 %d 个已处理 session", len(done_sessions))

    available_gpus = torch.cuda.device_count()
    if available_gpus <= 0:
        n_gpus = 1
        logger.warning("未检测到 CUDA GPU，退回 CPU，速度会很慢。")
    else:
        n_gpus = min(max(int(args.n_gpus), 1), available_gpus)
        if available_gpus < args.n_gpus:
            logger.warning("可见 GPU 数不足 %d，仅检测到 %d 张，将使用 %d 张。", args.n_gpus, available_gpus, n_gpus)
    logger.info(
        "streaming pipeline: files=%d n_gpus=%d batch_size=%d output_batch_sessions=%d row_group_sessions=%d prefetch_shards=%d result_queue_batches=%d",
        len(parquet_files),
        n_gpus,
        args.batch_size,
        args.sessions_per_output_batch,
        args.row_group_sessions,
        args.prefetch_shards,
        args.result_queue_batches,
    )

    global SESSIONS_PER_OUTPUT_BATCH
    SESSIONS_PER_OUTPUT_BATCH = max(1, int(args.sessions_per_output_batch))

    ctx = mp.get_context("spawn")
    task_queues = [
        ctx.Queue(maxsize=max(1, int(args.task_queue_size)))
        for _ in range(n_gpus)
    ]
    result_queue = ctx.Queue(maxsize=max(1, int(args.result_queue_batches)))
    sample_task_counts: Dict[int, int] = {}
    sample_meta: Dict[int, Dict[str, Any]] = {}
    reader_done = threading.Event()
    reader_error: List[BaseException] = []
    consume_result: Dict[str, int] = {}

    workers = []
    for rank in range(n_gpus):
        proc = ctx.Process(
            target=worker_encode_segments_streaming,
            args=(rank, task_queues[rank], mimi_path, args.batch_size, result_queue),
            daemon=False,
        )
        proc.start()
        workers.append(proc)

    def _consume_wrapper():
        try:
            consume_result.update(
                _consume_segment_results_streaming(
                    result_queue,
                    n_gpus,
                    sample_task_counts,
                    sample_meta,
                    reader_done,
                    reader_error,
                    output_dir,
                    args.output_filename,
                    args.row_group_sessions,
                    args.write_queue_batches,
                    schema=output_schema,
                )
            )
        except BaseException as exc:
            consume_result["error"] = exc

    consumer_thread = threading.Thread(target=_consume_wrapper, name="codec-consumer", daemon=True)
    consumer_thread.start()

    reader_thread = threading.Thread(
        target=_read_parquet_tasks_prefetch,
        args=(
            args.input_parquet_dir,
            done_sessions,
            args.max_samples,
            args.read_batch_rows,
            args.prefetch_shards,
            args.input_poll_seconds,
            args.input_settle_seconds,
            args.system_prompt_check_seconds,
            args.upstream_pattern,
            task_queues,
            sample_task_counts,
            sample_meta,
            reader_done,
            reader_error,
        ),
        name="codec-reader",
        daemon=True,
    )
    reader_thread.start()

    stats = None
    try:
        reader_thread.join()
        consumer_thread.join()
    finally:
        for proc in workers:
            proc.join()
    for proc in workers:
        if proc.exitcode != 0:
            raise RuntimeError(f"Worker pid={proc.pid} exitcode={proc.exitcode}")
    if "error" in consume_result:
        raise consume_result["error"]
    stats = consume_result
    if reader_error:
        raise reader_error[0]
    if stats:
        logger.info(
            "全部完成：写入 %d 个 session, writer rows=%d, failed_segments=%d",
            stats.get("completed_samples", 0),
            stats.get("completed_rows", 0),
            stats.get("failed_segments", 0),
        )


def main():
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Interleaved E2E: 从 session 级 parquet 提取 Mimi codec，输出带 codec 字段的 parquet")
    parser.add_argument("--input-parquet-dir",  default=DEFAULT_INPUT_PARQUET_DIR,  help="输入 session 级 parquet 目录")
    parser.add_argument("--output-dir",         default=DEFAULT_OUTPUT_PARQUET_DIR, help="输出 parquet 目录")
    parser.add_argument("--output-filename",    default=DEFAULT_OUTPUT_FILENAME,    help="输出 parquet 文件名前缀")
    parser.add_argument("--mimi-path",          default=str(DEFAULT_MIMI_PATH),     help="Mimi 模型路径")
    parser.add_argument("--n-gpus",             type=int, default=N_GPUS,           help="编码使用的 GPU 数")
    parser.add_argument("--batch-size",         type=int, default=DEFAULT_BATCH_SIZE, help="单卡同长度音频的 Mimi batch size")
    parser.add_argument("--max-samples",        type=int, default=None,             help="仅处理前 N 条（烟测用）")
    parser.add_argument("--resume",             action="store_true",                help="断点续传：跳过已有输出 parquet 里的 session")
    parser.add_argument("--sessions-per-output-batch", type=int, default=SESSIONS_PER_OUTPUT_BATCH,
                        help="每个输出 parquet 的 session 数，默认 10000")
    parser.add_argument("--row-group-sessions", type=int, default=DEFAULT_WRITE_ROW_GROUP_SESSIONS,
                        help="输出 parquet row group 的 session 数")
    parser.add_argument("--read-batch-rows", type=int, default=DEFAULT_READ_BATCH_ROWS,
                        help="输入 parquet iter_batches 的行数")
    parser.add_argument("--write-queue-batches", type=int, default=DEFAULT_WRITE_QUEUE_BATCHES,
                        help="writer 队列最多缓存多少个输出 parquet batch")
    parser.add_argument("--task-queue-size", type=int, default=DEFAULT_TASK_QUEUE_SIZE,
                        help="每张 GPU 的待编码任务队列大小")
    parser.add_argument("--result-queue-batches", type=int, default=DEFAULT_RESULT_QUEUE_BATCHES,
                        help="worker 到 consumer 的结果队列最多缓存多少个 batch")
    parser.add_argument("--prefetch-shards", type=int, default=DEFAULT_PREFETCH_SHARDS,
                        help="提前预读多少个输入 parquet shard")
    parser.add_argument("--input-poll-seconds", type=float, default=DEFAULT_INPUT_POLL_SECONDS,
                        help="输入目录没有新 parquet 时的轮询间隔")
    parser.add_argument("--input-settle-seconds", type=float, default=0.0,
                        help="输入目录轮询时确认 parquet 文件大小稳定的等待秒数")
    parser.add_argument("--system-prompt-check-seconds", type=float, default=DEFAULT_SYSTEM_PROMPT_CHECK_SECONDS,
                        help="输入 parquet 第一条 system_prompt 为空时的重试间隔")
    parser.add_argument("--upstream-pattern", default="[b]uild_training_parquet.py",
                        help="用于判断上游是否仍在运行的 pgrep pattern")
    parser.add_argument("--legacy-buffered", action="store_true",
                        help="使用旧版整 shard 缓冲流程，仅用于对照/回滚")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mimi_path = Path(args.mimi_path)

    logger.info("input_parquet_dir=%s, output_dir=%s, max_samples=%s, batch_size=%s, streaming=%s",
                args.input_parquet_dir, output_dir, args.max_samples, args.batch_size, not args.legacy_buffered)
    if not args.legacy_buffered:
        run_streaming_parquet_pipeline(args, output_dir, mimi_path)
        return

    # 断点续传：读已有输出 parquet 的 session_id 列
    done_sessions: set = set()
    if args.resume:
        done_sessions = _load_done_sessions(output_dir, args.output_filename)
        if done_sessions:
            logger.info("断点续传：已跳过 %d 个已处理 session", len(done_sessions))

    sample_list = build_sample_list_from_parquet(
        args.input_parquet_dir, done_sessions, max_samples=args.max_samples,
        batch_rows=args.read_batch_rows)
    if not sample_list:
        logger.warning("无新样本，退出")
        return

    pending_tasks, sample_task_counts = _build_pending_segment_tasks_parquet(sample_list)
    logger.info("总样本=%d, 待编码样本=%d, 待编码音频段=%d",
                len(sample_list), len(sample_task_counts), len(pending_tasks))

    # GPU 数量
    available_gpus = torch.cuda.device_count()
    if available_gpus <= 0:
        n_gpus = 1
        logger.warning("未检测到 CUDA GPU，退回 CPU，速度会很慢。")
    else:
        n_gpus = min(max(int(args.n_gpus), 1), available_gpus)
    logger.info("使用 %d 个 GPU 进行编码", n_gpus)

    # 多进程 Mimi encode
    codec_bytes_map: Dict[int, bytes] = {}
    if pending_tasks:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue(maxsize=max(4096, n_gpus * args.batch_size * 32))
        task_chunks, loads = _distribute_segment_tasks(pending_tasks, n_gpus)
        logger.info("启动 %d 个 GPU 进程，负载: %s", n_gpus, loads)
        workers = []
        consume_error = None
        for rank in range(n_gpus):
            proc = ctx.Process(
                target=worker_encode_segments,
                args=(rank, task_chunks[rank], mimi_path, args.batch_size, result_queue),
                daemon=False,
            )
            proc.start()
            workers.append(proc)
        try:
            codec_bytes_map = _consume_segment_results_parquet(
                result_queue, n_gpus, sample_task_counts)
        except Exception as exc:
            consume_error = exc
        finally:
            for proc in workers:
                proc.join()
        for proc in workers:
            if proc.exitcode != 0:
                raise RuntimeError(f"Worker pid={proc.pid} exitcode={proc.exitcode}")
        if consume_error is not None:
            raise consume_error
        logger.info("编码完成，成功 session=%d", len(codec_bytes_map))

    # 写输出 parquet：按 SESSIONS_PER_OUTPUT_BATCH 落盘
    batch_idx = len(sorted(output_dir.glob(f"{args.output_filename}_*.parquet")))
    pending_rows: List[Dict] = []
    output_schema = _build_output_schema(_detect_has_images(args.input_parquet_dir))
    written = 0
    sample_by_idx = {int(s["idx"]): s for s in sample_list}

    for idx, codec_bytes in codec_bytes_map.items():
        sample = sample_by_idx.get(idx)
        if sample is None:
            continue
        row = dict(sample["row"])
        # trim 最后一轮（同原版逻辑）——更新 messages 和 audios
        audios   = list(row["audios"])   if row.get("audios")   is not None else []
        messages = list(row["messages"]) if row.get("messages") is not None else []
        if len(audios) >= 2 and len(messages) >= 2:
            audios   = audios[:-2]
            messages = messages[:-2]
        row["audios"]   = audios
        row["messages"] = messages
        row["codec"]    = codec_bytes
        pending_rows.append(row)
        written += 1

        if len(pending_rows) >= SESSIONS_PER_OUTPUT_BATCH:
            out_path = output_dir / f"{args.output_filename}_{batch_idx:06d}.parquet"
            pq.write_table(pa.Table.from_pylist(pending_rows, schema=output_schema),
                           str(out_path), compression="snappy")
            logger.info("写出 batch_%06d: %d session → %s", batch_idx, len(pending_rows), out_path)
            batch_idx += 1
            pending_rows = []

    if pending_rows:
        out_path = output_dir / f"{args.output_filename}_{batch_idx:06d}.parquet"
        pq.write_table(pa.Table.from_pylist(pending_rows, schema=output_schema),
                       str(out_path), compression="snappy")
        logger.info("写出 batch_%06d: %d session → %s", batch_idx, len(pending_rows), out_path)
        batch_idx += 1

    logger.info("全部完成：写入 %d 个 session，共 %d 个 parquet 文件", written, batch_idx)


if __name__ == "__main__":
    main()
