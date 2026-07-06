import argparse
import concurrent.futures
import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
try:
    from tqdm import tqdm as _tqdm_cls
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Parquet 输出 schema（与 API 流水线 _PARQUET_SCHEMA 完全一致）────────────
_PARQUET_SCHEMA = pa.schema([
    pa.field("session_id",       pa.string()),
    pa.field("sample_id",        pa.string()),
    pa.field("split",            pa.string()),
    pa.field("chunk_id",         pa.int32()),
    pa.field("time_start",       pa.float32()),
    pa.field("time_end",         pa.float32()),
    pa.field("r1_audio_bytes",   pa.large_binary()),
    pa.field("r2_audio_bytes",   pa.large_binary()),
    pa.field("from_s2",          pa.string()),
    pa.field("asr_context",      pa.string()),
    pa.field("asr",              pa.string()),
    pa.field("tts",              pa.string()),
    pa.field("tts_control",      pa.string()),
    pa.field("system2_control",  pa.string()),
    pa.field("voice_plan",       pa.string()),
    pa.field("agent_speaker",    pa.string()),
    pa.field("user_speaker",     pa.string()),
])


class BatchParquetWriter:
    """线程安全的批量 parquet 写入器，每攒满 sessions_per_batch 个 session 原子落盘一次。"""

    def __init__(self, batches_dir: str, sessions_per_batch: int = 100):
        self._dir = batches_dir
        self._sessions_per_batch = sessions_per_batch
        self._lock = threading.Lock()
        self._pending_rows: list = []
        self._pending_sessions: list = []
        self._manifest_path = os.path.join(batches_dir, "manifest.txt")
        os.makedirs(batches_dir, exist_ok=True)
        existing = sorted(Path(batches_dir).glob("batch_*.parquet"))
        self._batch_idx = len(existing)

    def load_done_sessions(self) -> set:
        done = set()
        base = Path(self._manifest_path).parent.parent
        for manifest in sorted(base.glob("rank_*/manifest.txt")):
            with open(manifest, encoding="utf-8") as f:
                done |= {line.strip() for line in f if line.strip()}
        return done

    def add_session(self, session_name: str, rows: list):
        with self._lock:
            self._pending_rows.extend(rows)
            self._pending_sessions.append(session_name)
            if len(self._pending_sessions) >= self._sessions_per_batch:
                self._flush_locked()

    def flush_all(self):
        with self._lock:
            if self._pending_sessions:
                self._flush_locked()

    def _flush_locked(self):
        if not self._pending_rows:
            return
        tmp   = os.path.join(self._dir, f"batch_{self._batch_idx:06d}.parquet.tmp")
        final = os.path.join(self._dir, f"batch_{self._batch_idx:06d}.parquet")
        tbl = pa.Table.from_pylist(self._pending_rows, schema=_PARQUET_SCHEMA)
        pq.write_table(tbl, tmp, compression="snappy")
        os.rename(tmp, final)
        with open(self._manifest_path, "a", encoding="utf-8") as f:
            for s in self._pending_sessions:
                f.write(s + "\n")
        logging.info("[BatchParquetWriter] batch_%06d: %d rows, %d sessions → %s",
                     self._batch_idx, len(self._pending_rows), len(self._pending_sessions), final)
        self._batch_idx += 1
        self._pending_rows = []
        self._pending_sessions = []


# 模块级单例，由 main_sync 初始化
_BATCH_WRITER: "BatchParquetWriter | None" = None

# 哨兵对象，表示从 pending_queue get 超时（区别于 None sentinel）
_EMPTY = object()

DEFAULT_CUSTOM_MODEL_PATH = os.environ.get("QWEN3_TTS_CUSTOM_MODEL", "models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEFAULT_BASE_MODEL_PATH = os.environ.get("QWEN3_TTS_BASE_MODEL", "models/Qwen3-TTS-12Hz-1.7B-Base")
DEFAULT_ALIGNER_MODEL_PATH = os.environ.get("QWEN3_ALIGNER_MODEL", "models/Qwen3-ForcedAligner-0.6B")
DEFAULT_INPUT_PATH = os.environ.get("TTS_INPUT_PATH", "data/chat_data")
DEFAULT_OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", "outputs/tts/output_dialogue_e2e")
DEFAULT_CUT_DIR = os.environ.get("TTS_CUT_DIR", "outputs/tts/finalcut_sessions")
DEFAULT_CLONE_PROMPT_DIR = os.environ.get("TTS_CLONE_PROMPT_DIR", "data/user_clone_prompts")
DEFAULT_CLONE_PROMPT_TEXT = "这句话带着温柔的期许，像风轻轻吹过耳边，让人心里暖暖的。你是在鼓励我，还是在悄悄给自己打气呢？不管怎样，我会记住这份心意，努力走得更远，不负你的期待。"
DEFAULT_AGENT_SPEAKER = "Vivian"
DEFAULT_AGENT_INSTRUCT = "略快语速，干练且利落地说"
# Agent 声线 Voice Clone 参考音频（空字符串表示不启用，仍走 custom_voice）
DEFAULT_AGENT_REF_AUDIO = os.environ.get("TTS_AGENT_REF_AUDIO", "")
DEFAULT_AGENT_REF_TEXT  = "The weather is nice, and I speak calmly and clearly. 今天天气很好，我平静清楚地说话。"
DEFAULT_USER_CUSTOM_SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan"]
DEFAULT_USER_CUSTOM_RATIO = 1.0  # User 全部走 custom_voice 内置音色，不走 clone

# 按语言分开的内置音色池（session 内稳定，跨语言隔离）
_SPEAKERS_ZH = ["Vivian", "Serena", "Uncle_Fu", "Dylan"]          # 中文：不含英语母语音色
_SPEAKERS_EN = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Ryan", "Aiden"]  # 英文：全部可用

# PT 推理默认参数
DEFAULT_TTS_GPU_IDS = "0,1,2,3,4,5,6,7"
DEFAULT_TTS_BATCH_SIZE = 40           # 每张卡的 batch size
DEFAULT_TTS_ACCUMULATE_TIMEOUT = 5.0  # 攒批超时秒数
DEFAULT_TTS_MAX_RETRIES = 3           # 每条请求最大重试次数

SAMPLE_RATE = 24000
ALIGNER_TARGET_SR = 16000

CHANNELS = 1
SAMPWIDTH = 2
CHUNK_MS = 480
CHUNK_SEC = CHUNK_MS / 1000.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC) * CHANNELS
CHARS_PER_CHUNK = 5
MAX_NEW_TOKENS = 2048
PUNCTUATION = set("。！？，、；：.!?,;:")
ALIGNER_PUNCT_SET = set("，。！？、；：""''（）【】《》…—· \t\n" + ".,!?;:'\"()[]{}-–—/\\@#$%^&*+=<>|~`")
PEND_TO_EXTRA_CHUNKS = {
    "PEND1S": 2, "PEND2S": 4, "PEND3S": 6, "PEND4S": 8, "PEND5S": 10,
    "PEND6S": 12, "PEND7S": 14, "PEND8S": 16, "PEND9S": 18, "PEND10S": 20,
    "PEND11S": 22, "PEND12S": 24, "PEND13S": 26, "PEND14S": 28, "PEND15S": 30,
    "PEND16S": 32, "PEND17S": 34, "PEND18S": 36, "PEND19S": 38, "PEND20S": 40,
    "PEND21S": 42, "PEND22S": 44, "PEND23S": 46, "PEND24S": 48, "PEND25S": 50,
    "PEND26S": 52, "PEND27S": 54, "PEND28S": 56, "PEND29S": 58, "PEND30S": 60,
    "PEND31S": 62, "PEND32S": 64, "PEND33S": 66, "PEND34S": 68, "PEND35S": 70,
}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


def stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def safe_name(value: str, fallback: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", value or "").strip("_")
    return text or fallback


def detect_language(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"
    if re.search(r"[A-Za-z]", text):
        return "English"
    return "Auto"


def _largest_power_of_two_leq(value: int) -> int:
    if value <= 0:
        return 0
    return 1 << (value.bit_length() - 1)


class RuntimeConfig:
    def __init__(self, args: argparse.Namespace):
        self.output_dir = args.output_dir
        self.cut_dir = args.cut_dir
        self.custom_model_path = args.custom_model_path
        self.base_model_path = args.base_model_path
        self.aligner_model_path = args.aligner_model_path
        self.aligner_gpu_ids = tuple(int(x) for x in args.aligner_gpu_ids.split(",") if x.strip())
        if not self.aligner_gpu_ids:
            raise ValueError("--aligner-gpu-ids 不能为空")
        self.tts_gpu_ids = tuple(int(x) for x in args.tts_gpu_ids.split(",") if x.strip())
        if not self.tts_gpu_ids:
            raise ValueError("--tts-gpu-ids 不能为空")
        self.tts_batch_size = args.tts_batch_size
        if self.tts_batch_size <= 0:
            raise ValueError("--tts-batch-size 必须为正整数")
        self.tts_accumulate_timeout = args.tts_accumulate_timeout
        self.tts_max_retries = args.tts_max_retries
        self.session_concurrency = args.session_concurrency
        self.finalize_workers = args.finalize_workers
        self.user_custom_ratio = args.user_custom_ratio
        self.user_clone_prompt_dir = args.user_clone_prompt_dir
        self.user_clone_prompt_text = args.user_clone_prompt_text
        self.agent_speaker = args.agent_speaker
        self.agent_instruct = args.agent_instruct
        self.agent_ref_audio = args.agent_ref_audio  # 空字符串 → 不启用 clone
        self.agent_ref_text  = args.agent_ref_text
        speakers = [item.strip() for item in args.user_custom_speakers.split(",") if item.strip()]
        self.user_custom_speakers = speakers or list(DEFAULT_USER_CUSTOM_SPEAKERS)
        self.resume = args.resume
        self.max_samples = args.max_samples
        self.node_rank = args.node_rank
        self.nnodes = args.nnodes


def _aligner_is_punct(char: str) -> bool:
    return char in ALIGNER_PUNCT_SET or (not char.isalnum() and not ("\u4e00" <= char <= "\u9fff"))


def _aligner_build_o_clean_and_positions(text: str):
    clean_chars = []
    clean_positions = []
    for idx, char in enumerate(text):
        if not _aligner_is_punct(char):
            clean_chars.append(char)
            clean_positions.append(idx)
    clean_positions.append(len(text))
    return "".join(clean_chars), clean_positions


def pcm_24k_to_float16k(pcm_bytes: bytes):
    if not pcm_bytes:
        raise ValueError("aligner 收到空音频")
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64) / 32768.0
    if SAMPLE_RATE == ALIGNER_TARGET_SR:
        return audio.astype(np.float32), ALIGNER_TARGET_SR
    duration = len(audio) / SAMPLE_RATE
    new_len = int(duration * ALIGNER_TARGET_SR)
    if new_len <= 0:
        raise ValueError("aligner 音频长度非法")
    audio_16k = np.interp(
        np.linspace(0, len(audio) - 1, new_len),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)
    return audio_16k, ALIGNER_TARGET_SR


def tokenize_text(text: str):
    tokens = []
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char in PUNCTUATION:
            if tokens:
                tokens[-1] = (tokens[-1][0] + char, False)
            else:
                tokens.append((char, True))
            idx += 1
            continue
        if "\u4e00" <= char <= "\u9fff":
            tokens.append((char, False))
            idx += 1
            continue
        if char.isalpha():
            word = ""
            while idx < len(text) and (text[idx].isalpha() or text[idx] == "'") and not ("\u4e00" <= text[idx] <= "\u9fff"):
                word += text[idx]
                idx += 1
            tokens.append((word, False))
            continue
        if char.isdigit():
            num = ""
            while idx < len(text) and text[idx].isdigit():
                num += text[idx]
                idx += 1
            tokens.append((num, False))
            continue
        if tokens:
            prev_text, prev_is_punct = tokens[-1]
            tokens[-1] = (prev_text + char, prev_is_punct)
        else:
            tokens.append((char, True))
        idx += 1
    return tokens


def count_non_punctuation_chars(text: str) -> int:
    return sum(1 for token, is_punct in tokenize_text(text) if not is_punct)


def _extract_chunk_bytes(audio_track: np.ndarray, chunk_idx: int,
                          chunk_start_sec: float, chunk_end_sec: float) -> bytes:
    """从整轨 PCM int16 数组中截取一个 chunk 并 pad 到 CHUNK_SAMPLES，返回 bytes。"""
    start = int(chunk_start_sec * SAMPLE_RATE)
    end   = int(chunk_end_sec   * SAMPLE_RATE)
    chunk = audio_track[start:end]
    if len(chunk) < CHUNK_SAMPLES:
        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
    elif len(chunk) > CHUNK_SAMPLES:
        chunk = chunk[:CHUNK_SAMPLES]
    return chunk.tobytes()


def save_wav(filepath: str, audio_bytes: bytes):
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_bytes)


def save_chunk_audio(audio_track, chunk_idx, chunk_start_sec, chunk_end_sec, audio_chunks_dir, prefix):
    start_sample = int(chunk_start_sec * SAMPLE_RATE)
    end_sample = int(chunk_end_sec * SAMPLE_RATE)
    chunk_audio = audio_track[start_sample:end_sample]
    audio_filename = f"{prefix}_chunk_{chunk_idx:04d}.wav"
    audio_path = os.path.join(audio_chunks_dir, audio_filename)
    save_wav(audio_path, chunk_audio.tobytes())
    return f"audio_chunks/{audio_filename}"


def pad_audio_bytes_to_chunk_multiple(audio_bytes: bytes):
    """将 PCM 字节补零到 CHUNK_SAMPLES 的整数倍，返回 (padded_bytes, duration_sec)。"""
    pcm = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(pcm) == 0:
        return audio_bytes, 0.0
    remainder = len(pcm) % CHUNK_SAMPLES
    if remainder:
        pad = np.zeros(CHUNK_SAMPLES - remainder, dtype=np.int16)
        pcm = np.concatenate([pcm, pad])
    padded_duration = len(pcm) / (SAMPLE_RATE * CHANNELS)
    return pcm.tobytes(), padded_duration


def pcm_duration_sec(audio_bytes: bytes) -> float:
    return len(np.frombuffer(audio_bytes, dtype=np.int16)) / (SAMPLE_RATE * CHANNELS)


def parse_special_tags(text: str):
    pend_chunks = 0

    def _replace_pend(match):
        nonlocal pend_chunks
        pend_chunks += PEND_TO_EXTRA_CHUNKS[match.group(1)]
        return ""

    stripped = re.sub(r"\[(PEND(?:10|[1-9])S)\]", _replace_pend, text or "")
    s2_pattern = r"「([^」]+)」"
    s2_matches = list(re.finditer(s2_pattern, stripped))
    control_pattern = r"\[(THINK|WAIT)\]"
    control_matches = list(re.finditer(control_pattern, stripped))

    temp_text = stripped
    for match in reversed(s2_matches):
        temp_text = temp_text[: match.start()] + temp_text[match.end() :]
    for match in reversed(control_matches):
        temp_text = temp_text[: match.start()] + temp_text[match.end() :]

    clean_chars = []
    trigger_positions = []
    cut_pos = None
    ghost_start = None
    idx = 0
    while idx < len(temp_text):
        if temp_text[idx] == "^":
            trigger_positions.append(len(clean_chars))
            idx += 1
        elif temp_text[idx : idx + 5] == "[CUT]":
            cut_pos = len(clean_chars)
            ghost_start = idx + 5
            idx += 5
        else:
            clean_chars.append(temp_text[idx])
            idx += 1

    clean_text = "".join(clean_chars).strip()
    result = {
        "clean_text": clean_text,
        "display_text": clean_text,
        "pend_chunks": pend_chunks,
        "triggers": trigger_positions,
        "cut_position": cut_pos,
        "s2_messages": [],
        "s2_controls": [],
        "ghost_text": temp_text[ghost_start:].strip() if ghost_start is not None else "",
    }

    for match in s2_matches:
        original_pos = match.start()
        offset = 0
        for item in s2_matches:
            if item.start() < original_pos:
                offset += len(item.group(0))
        for item in control_matches:
            if item.start() < original_pos:
                offset += len(item.group(0))
        clean_pos = original_pos - offset
        result["s2_messages"].append((clean_pos, match.group(1)))

    for match in control_matches:
        original_pos = match.start()
        offset = 0
        for item in s2_matches:
            if item.start() < original_pos:
                offset += len(item.group(0))
        for item in control_matches:
            if item.start() < original_pos:
                offset += len(item.group(0))
        clean_pos = original_pos - offset
        result["s2_controls"].append((clean_pos, match.group(1)))

    return result


def run_align_for_utterance(aligner_model, audio_pcm_bytes, text):
    """
    对单条语音执行强制对齐，返回 (aligned_char_timestamps, total_duration)。
    任何异常（空结果、时长非法、重建不一致）均直接 raise，不做 fallback。
    audio_pcm_bytes 应已经 pad 到 CHUNK_SAMPLES 整数倍。
    """
    if not text:
        raise ValueError("aligner 收到空文本")
    audio_float, sr = pcm_24k_to_float16k(audio_pcm_bytes)
    results = aligner_model.align(audio=(audio_float, sr), text=text, language=detect_language(text))
    if not results or not results[0]:
        raise RuntimeError(f"aligner 返回空结果: text={text[:40]!r}")

    original_text = text
    segments = results[0]
    clean_text, clean_positions = _aligner_build_o_clean_and_positions(original_text)
    offset = 0
    segment_spans = []
    for segment in segments:
        seg_text = (getattr(segment, "text", "") or "").strip()
        if not seg_text:
            pos = clean_positions[offset] if offset < len(clean_positions) else len(original_text)
            segment_spans.append((pos, pos))
            continue
        seg_len = len(seg_text)
        end_clean = min(offset + seg_len, len(clean_text))
        start_o = clean_positions[offset] if offset < len(clean_positions) else len(original_text)
        end_o = clean_positions[end_clean] if end_clean < len(clean_positions) else len(original_text)
        segment_spans.append((start_o, end_o))
        offset += seg_len

    total_duration = max((getattr(segment, "end_time", 0) or 0) for segment in segments)
    if total_duration <= 0:
        raise RuntimeError(f"aligner 总时长非法: {total_duration}, text={text[:40]!r}")

    char_start = [0.0] * len(original_text)
    char_end = [0.0] * len(original_text)
    for seg_idx, segment in enumerate(segments):
        start_time = getattr(segment, "start_time", 0) or 0
        end_time = getattr(segment, "end_time", 0) or 0
        span_start, span_end = segment_spans[seg_idx]
        if span_end <= span_start:
            continue
        seg_duration = end_time - start_time
        for pos in range(span_start, span_end):
            frac = (pos - span_start) / (span_end - span_start)
            char_start[pos] = start_time + seg_duration * frac
            char_end[pos] = start_time + seg_duration * (frac + 1.0 / (span_end - span_start))

    tokens = tokenize_text(original_text)
    aligned_char_timestamps = []
    pos = 0
    for token, is_punct in tokens:
        if is_punct:
            if aligned_char_timestamps:
                aligned_char_timestamps[-1]["text"] += token
            continue
        token_len = len(token)
        end_pos = min(pos + token_len, len(original_text))
        if pos >= len(original_text):
            raise RuntimeError(f"aligner 重建位置越界: pos={pos}, text={text[:40]!r}")
        t_start = char_start[pos]
        t_end = char_end[end_pos - 1] if end_pos > pos else char_start[pos]
        aligned_char_timestamps.append(
            {
                "text": token,
                "start_ms": t_start * 1000,
                "end_ms": t_end * 1000,
                "start_sec": t_start,
                "end_sec": t_end,
                "is_punctuation": False,
            }
        )
        pos = end_pos

    reconstructed = "".join(item["text"] for item in aligned_char_timestamps)
    if reconstructed != original_text:
        raise RuntimeError(
            f"aligner 重建文本与原文不一致: reconstructed={reconstructed[:40]!r} vs original={original_text[:40]!r}"
        )
    return aligned_char_timestamps, total_duration


class ThreadedAlignerPool:
    """
    每个 aligner worker 有独立的 task_queue，submit 时按 session_name hash 路由。
    同一 session 的所有 align 任务发到同一个 worker，保证该 session 的所有
    utterance align 完毕后立即触发落盘，不被其他 session 的任务插队。
    
    使用纯线程阻塞方式：submit_sync(...) 会在调用线程中阻塞，直到 align 完成并返回结果。
    """
    def __init__(self, gpu_ids, model_path):
        self.gpu_ids = tuple(gpu_ids)
        self.num_workers = len(self.gpu_ids)
        self.model_path = model_path
        self.ctx = mp.get_context("spawn")
        # 每个 worker 一个独立队列，按 session hash 路由
        self.task_queues = [self.ctx.Queue() for _ in self.gpu_ids]
        self.result_queue = self.ctx.Queue()
        self.processes = []
        self.pending = {}
        self.pending_lock = threading.Lock()
        self.listener_thread = None
        self.closed = False

    def _worker_idx_for_session(self, session_name: str) -> int:
        """同一 session 的所有 utterance 固定路由到同一个 aligner worker。"""
        return stable_int(session_name) % self.num_workers

    def start(self):
        if self.processes:
            return
        for worker_id, (gpu_id, tq) in enumerate(zip(self.gpu_ids, self.task_queues)):
            proc = self.ctx.Process(
                target=_aligner_worker_main,
                args=(worker_id, gpu_id, self.model_path, tq, self.result_queue),
                daemon=True,
            )
            proc.start()
            self.processes.append(proc)

        ready_workers = {}
        while len(ready_workers) < len(self.gpu_ids):
            try:
                msg = self.result_queue.get(timeout=600)
            except queue.Empty as exc:
                raise RuntimeError("等待 aligner worker 启动超时。") from exc
            if not msg or msg[0] != "ready":
                continue
            _, worker_id, gpu_id, error = msg
            if error:
                self.close()
                raise RuntimeError(f"Aligner worker {worker_id} (GPU {gpu_id}) 启动失败: {error}")
            ready_workers[worker_id] = gpu_id
            logging.info(f"✅ Aligner worker {worker_id} 已就绪，绑定 GPU {gpu_id}")

        self.listener_thread = threading.Thread(target=self._listen_results, daemon=True)
        self.listener_thread.start()
        logging.info(f"Aligner worker pool 已就绪，共 {len(self.gpu_ids)} 个 worker。")

    def _listen_results(self):
        while True:
            msg = self.result_queue.get()
            if msg is None:
                return
            if not msg or msg[0] != "result":
                continue
            _, task_id, ts, dur, error = msg
            with self.pending_lock:
                evt, result_holder = self.pending.pop(task_id, (None, None))
            if evt is None:
                continue
            result_holder["ts"] = ts
            result_holder["dur"] = dur
            result_holder["error"] = error
            evt.set()

    def submit_sync(self, audio_pcm_bytes, text, session_name, item_index):
        """同步阻塞调用：等待 align 完成并返回 (ts, dur)。"""
        if self.closed:
            raise RuntimeError("Aligner pool 已关闭，无法继续提交任务。")
        task_id = f"{session_name}:{item_index}:{time.time_ns()}"
        evt = threading.Event()
        result_holder = {}
        with self.pending_lock:
            self.pending[task_id] = (evt, result_holder)
        # 按 session_name hash 路由到固定 worker 的专属队列
        worker_idx = self._worker_idx_for_session(session_name)
        self.task_queues[worker_idx].put((task_id, audio_pcm_bytes, text))
        evt.wait()
        if result_holder.get("error"):
            raise RuntimeError(f"Aligner task failed: {result_holder['error']}")
        return result_holder["ts"], result_holder["dur"]

    def close(self):
        if self.closed:
            return
        self.closed = True
        for tq in self.task_queues:
            tq.put(None)
        for proc in self.processes:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        self.result_queue.put(None)
        if self.listener_thread is not None:
            self.listener_thread.join(timeout=5)


def _aligner_worker_main(worker_id, gpu_id, model_path, task_queue, result_queue):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logging.info(f"[AlignerWorker {worker_id}] 正在加载 GPU {gpu_id} 上的模型...")
        import torch as _torch
        from qwen_asr import Qwen3ForcedAligner

        aligner_model = Qwen3ForcedAligner.from_pretrained(
            model_path,
            dtype=_torch.bfloat16,
            device_map="cuda:0",
        )
        result_queue.put(("ready", worker_id, gpu_id, None))
        logging.info(f"[AlignerWorker {worker_id}] GPU {gpu_id} 模型加载完成。")
    except Exception as exc:
        result_queue.put(("ready", worker_id, gpu_id, repr(exc)))
        logging.exception(f"[AlignerWorker {worker_id}] GPU {gpu_id} 启动失败: {exc}")
        return

    while True:
        task = task_queue.get()
        if task is None:
            break
        task_id, audio_pcm_bytes, text = task
        try:
            ts, dur = run_align_for_utterance(aligner_model, audio_pcm_bytes, text)
            result_queue.put(("result", task_id, ts, dur, None))
        except Exception as exc:
            logging.exception(f"[AlignerWorker {worker_id}] 处理任务失败: {exc}")
            result_queue.put(("result", task_id, None, None, repr(exc)))


# ─────────────────────────────────────────────────────────────
#  PT 多卡并行 TTS 推理器
# ─────────────────────────────────────────────────────────────

# 子进程任务消息格式：
#   ("batch", batch_id, jobs_serializable)   -> 执行推理
#   None                                      -> 终止 worker
#
# 子进程结果消息格式：
#   ("result", batch_id, results_list, error_str_or_None)
#
# jobs_serializable 中每条 job 为 dict，包含：
#   request_id, tts_type("custom"), text, language, speaker, instruct
#
# results_list 中每条为 dict：
#   request_id, audio_bytes(bytes, 已 pad 到 CHUNK_SAMPLES 整数倍) or None, duration_sec, error(str or None)


def _pt_tts_worker_main(worker_id: int, gpu_id: int, custom_model_path: str,
                         base_model_path: str,
                         agent_ref_audio: str, agent_ref_text: str,
                         user_clone_prompts_data: list,
                         task_queue: mp.Queue, result_queue: mp.Queue):
    """
    子进程：每张 GPU 上运行一个 Qwen3TTSModel，从 task_queue 中取 batch 进行推理。

    - Agent job（tts_type="agent_clone"）：Base 模型 + generate_voice_clone（音色由 agent_ref_audio 锁定）
    - User job（tts_type="user_clone"）：Base 模型 + generate_voice_clone（音色由 user clone profile 决定）
    - User job（tts_type="custom"）：CustomVoice 模型 + generate_custom_voice

    推理完成后立即对每条音频执行 pad 到 CHUNK_SAMPLES 整数倍，再放入结果队列。
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [TTS-W{worker_id}/GPU{gpu_id}] %(levelname)s %(message)s")
        import torch as _torch
        from qwen_tts import Qwen3TTSModel

        # ── 加载 CustomVoice 模型（用于 User custom 句子）────────────
        logging.info(f"[TTSWorker {worker_id}] GPU {gpu_id}: 正在加载 CustomVoice 模型 {custom_model_path} ...")
        model = Qwen3TTSModel.from_pretrained(
            custom_model_path,
            device_map="cuda:0",
            dtype=_torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        logging.info(f"[TTSWorker {worker_id}] GPU {gpu_id}: CustomVoice 模型加载完成。")

        # ── 判断是否需要 Base 模型（agent clone 或 user clone 任一需要就加载）──
        need_base = bool(agent_ref_audio) or bool(user_clone_prompts_data)
        base_model = None
        agent_clone_prompt = None
        # profile_id -> clone_prompt 对象
        user_clone_prompts: dict = {}

        if need_base:
            logging.info(f"[TTSWorker {worker_id}] GPU {gpu_id}: 正在加载 Base 模型 {base_model_path} ...")
            base_model = Qwen3TTSModel.from_pretrained(
                base_model_path,
                device_map="cuda:0",
                dtype=_torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            logging.info(f"[TTSWorker {worker_id}] GPU {gpu_id}: Base 模型加载完成。")

            if agent_ref_audio:
                logging.info(f"[TTSWorker {worker_id}] GPU {gpu_id}: 提取 Agent clone prompt ...")
                agent_clone_prompt = base_model.create_voice_clone_prompt(
                    ref_audio=agent_ref_audio,
                    ref_text=agent_ref_text,
                    x_vector_only_mode=False,
                )
                logging.info(f"[TTSWorker {worker_id}] GPU {gpu_id}: Agent clone prompt 提取完成。")

            if user_clone_prompts_data:
                logging.info(f"[TTSWorker {worker_id}] GPU {gpu_id}: 提取 {len(user_clone_prompts_data)} 个 User clone prompt ...")
                for profile in user_clone_prompts_data:
                    pid = profile["profile_id"]
                    try:
                        prompt = base_model.create_voice_clone_prompt(
                            ref_audio=profile["prompt_audio_path"],
                            ref_text=profile["prompt_text"],
                            x_vector_only_mode=False,
                        )
                        user_clone_prompts[pid] = prompt
                        logging.info(f"[TTSWorker {worker_id}] GPU {gpu_id}: User clone profile '{pid}' 提取完成。")
                    except Exception as exc:
                        logging.warning(f"[TTSWorker {worker_id}] GPU {gpu_id}: User clone profile '{pid}' 提取失败: {exc}")
                logging.info(f"[TTSWorker {worker_id}] GPU {gpu_id}: User clone prompts 提取完成，共 {len(user_clone_prompts)} 个。")

        result_queue.put(("ready", worker_id, gpu_id, None))
    except Exception as exc:
        logging.exception(f"[TTSWorker {worker_id}] GPU {gpu_id}: 启动失败: {exc}")
        result_queue.put(("ready", worker_id, gpu_id, repr(exc)))
        return

    while True:
        msg = task_queue.get()
        if msg is None:
            break
        msg_type, batch_id, jobs = msg
        if msg_type != "batch":
            continue

        # 预先填充结果列表
        results = [{"request_id": j["request_id"], "audio_bytes": None, "duration_sec": 0.0, "error": None} for j in jobs]

        if not jobs:
            result_queue.put(("result", batch_id, results, None))
            continue

        try:
            # 按 tts_type 拆分
            agent_indices = [i for i, j in enumerate(jobs) if j.get("tts_type") == "agent_clone"]
            user_clone_indices = [i for i, j in enumerate(jobs) if j.get("tts_type") == "user_clone"]
            custom_indices = [i for i, j in enumerate(jobs) if j.get("tts_type") == "custom"]

            t0 = time.perf_counter()

            # ── Agent clone batch ──────────────────────────────
            if agent_indices and agent_clone_prompt is not None:
                agent_texts = [jobs[i]["text"] for i in agent_indices]
                agent_langs = [jobs[i]["language"] for i in agent_indices]
                agent_wavs, sr = base_model.generate_voice_clone(
                    text=agent_texts,
                    language=agent_langs,
                    voice_clone_prompt=agent_clone_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                for list_pos, orig_idx in enumerate(agent_indices):
                    results[orig_idx]["_wav"] = agent_wavs[list_pos]
                    results[orig_idx]["_sr"]  = sr

            # ── User clone batch（按 profile_id 分组，每组各自调一次 generate_voice_clone）──
            if user_clone_indices and base_model is not None:
                # 按 profile_id 分组
                profile_groups: dict = {}
                for i in user_clone_indices:
                    pid = jobs[i].get("clone_profile_id", "")
                    profile_groups.setdefault(pid, []).append(i)
                for pid, indices in profile_groups.items():
                    clone_prompt = user_clone_prompts.get(pid)
                    if clone_prompt is None:
                        # profile 不可用，fallback 到 custom_voice
                        custom_indices = custom_indices + indices
                        continue
                    texts = [jobs[i]["text"] for i in indices]
                    langs = [jobs[i]["language"] for i in indices]
                    wavs, sr = base_model.generate_voice_clone(
                        text=texts,
                        language=langs,
                        voice_clone_prompt=clone_prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    for list_pos, orig_idx in enumerate(indices):
                        results[orig_idx]["_wav"] = wavs[list_pos]
                        results[orig_idx]["_sr"]  = sr
            elif user_clone_indices:
                # base_model 不可用，fallback
                custom_indices = custom_indices + user_clone_indices

            # ── Custom voice batch（User custom_voice 或 fallback）──
            # 无 clone prompt 的 agent 也 fallback 到 custom
            fallback_agent = agent_indices if agent_clone_prompt is None else []
            all_custom = custom_indices + fallback_agent
            if all_custom:
                texts     = [jobs[i]["text"]              for i in all_custom]
                languages = [jobs[i]["language"]          for i in all_custom]
                speakers  = [jobs[i]["speaker"]           for i in all_custom]
                instructs = [jobs[i].get("instruct", "")  for i in all_custom]
                custom_wavs, sr = model.generate_custom_voice(
                    text=texts,
                    language=languages,
                    speaker=speakers,
                    instruct=instructs,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                for list_pos, orig_idx in enumerate(all_custom):
                    results[orig_idx]["_wav"] = custom_wavs[list_pos]
                    results[orig_idx]["_sr"]  = sr

            elapsed = time.perf_counter() - t0

            total_audio_sec = 0.0
            for idx in range(len(jobs)):
                wav = results[idx].pop("_wav", None)
                if wav is None:
                    results[idx]["error"] = "未生成音频"
                    continue
                # wav -> float32 numpy
                if hasattr(wav, "numpy"):
                    wav_np = wav.float().cpu().numpy().flatten()
                elif isinstance(wav, np.ndarray):
                    wav_np = wav.flatten().astype(np.float32)
                else:
                    wav_np = np.asarray(wav, dtype=np.float32).flatten()
                wav_np = np.clip(wav_np, -1.0, 1.0)
                raw_bytes = (wav_np * 32767.0).astype(np.int16).tobytes()
                # 立即 pad 到 CHUNK_SAMPLES 整数倍
                padded_bytes, duration_sec = _pad_pcm_to_chunk_multiple(raw_bytes)
                results[idx]["audio_bytes"] = padded_bytes
                results[idx]["duration_sec"] = duration_sec
                total_audio_sec += duration_sec

            rtf = elapsed / total_audio_sec if total_audio_sec > 0 else 0.0
            logging.info(
                "[TTSWorker %d][GPU %d] batch_id=%s size=%d audio=%.2fs wall=%.2fs rtf=%.3f",
                worker_id, gpu_id, batch_id, len(jobs), total_audio_sec, elapsed, rtf,
            )
            # 结果消息：("result", batch_id, results, error)
            result_queue.put(("result", batch_id, results, None))

        except Exception as exc:
            logging.exception(f"[TTSWorker {worker_id}] GPU {gpu_id}: batch {batch_id} 推理失败: {exc}")
            for r in results:
                if r["audio_bytes"] is None:
                    r["error"] = repr(exc)
            result_queue.put(("result", batch_id, results, repr(exc)))


def _pad_pcm_to_chunk_multiple(raw_bytes: bytes):
    """独立函数，可在子进程中调用，避免 import 问题。"""
    pcm = np.frombuffer(raw_bytes, dtype=np.int16)
    if len(pcm) == 0:
        return raw_bytes, 0.0
    remainder = len(pcm) % CHUNK_SAMPLES
    if remainder:
        pad = np.zeros(CHUNK_SAMPLES - remainder, dtype=np.int16)
        pcm = np.concatenate([pcm, pad])
    duration = len(pcm) / (SAMPLE_RATE * CHANNELS)
    return pcm.tobytes(), duration


class MultiGPUPTSynthesizer:
    """
    多卡 PT 批量 TTS 合成器（V3：每卡独立 dispatcher，完成即取下一批）。

    工作方式：
    - 调用方通过 submit_job() 提交单条 TTS job（非阻塞），返回 Future。
    - 每张 GPU 有一个专属的 per-device pending queue（按 session_name hash 路由）：
        * 同一 session 的所有 utterance 始终路由到同一张卡；
        * 这样 session 完成后可立刻落盘，无需等待其他卡；
        * 每卡有独立的双缓冲 Semaphore（值=2），允许最多 2 个 batch 同时在途；
        * 该 GPU 完成一批后立即释放 Semaphore，dispatcher 立刻取下一批，
          不需等待其他 GPU，彻底消除快卡等慢卡的问题。
    - 每条 job 失败时在结果处理层重试，最多 max_retries 次。
    - wait_all() 等待所有已提交任务完成（发 sentinel，等所有 dispatcher 退出）。
    - Future 结果为 (audio_bytes: bytes, duration_sec: float)，
      audio_bytes 已经 pad 到 CHUNK_SAMPLES 整数倍。
    """

    def __init__(self, gpu_ids, custom_model_path: str, base_model_path: str,
                 agent_ref_audio: str, agent_ref_text: str,
                 user_clone_prompts_data: list,
                 batch_size: int, accumulate_timeout: float, max_retries: int):
        self.gpu_ids = list(gpu_ids)
        self.num_devices = len(self.gpu_ids)
        self.custom_model_path = custom_model_path
        self.base_model_path   = base_model_path
        self.agent_ref_audio   = agent_ref_audio
        self.agent_ref_text    = agent_ref_text
        self.user_clone_prompts_data = user_clone_prompts_data
        self.per_device_batch = batch_size
        self.accumulate_timeout = accumulate_timeout
        self.max_retries = max_retries

        self._ctx = mp.get_context("spawn")
        self._worker_task_queues = [self._ctx.Queue() for _ in self.gpu_ids]
        self._result_queue = self._ctx.Queue()
        self._workers = []

        # 每卡专属 pending queue，同一 session 的 utterance 按 hash 路由到固定卡
        # 这样同一 session 的所有 utterance 在同一卡上完成，完成即可落盘
        self._device_pending_queues: list = [queue.Queue() for _ in self.gpu_ids]

        self._futures: dict = {}          # req_id -> (Future, retry_count)
        self._futures_lock = threading.Lock()
        self._inflight_batches: dict = {} # batch_id -> batch_info
        self._inflight_lock = threading.Lock()

        self._stopped = False
        self._dispatcher_threads: list = []
        self._result_listener_thread = None
        self._batch_id_counter = 0
        self._batch_id_lock = threading.Lock()

        # 用于 wait_all: 跟踪还未完成（未 resolve）的 Future 数量
        self._pending_futures_count = 0
        self._pending_futures_lock = threading.Lock()
        self._all_futures_done_event = threading.Event()

        # worker 重启锁（避免多线程同时重启同一个 worker）
        self._worker_restart_lock = threading.Lock()

    # ── 启动 ──────────────────────────────────────────────────

    def start(self):
        for worker_id, (gpu_id, tq) in enumerate(zip(self.gpu_ids, self._worker_task_queues)):
            proc = self._ctx.Process(
                target=_pt_tts_worker_main,
                args=(worker_id, gpu_id, self.custom_model_path,
                      self.base_model_path,
                      self.agent_ref_audio, self.agent_ref_text,
                      self.user_clone_prompts_data,
                      tq, self._result_queue),
                daemon=True,
            )
            proc.start()
            self._workers.append(proc)

        ready_count = 0
        while ready_count < self.num_devices:
            try:
                msg = self._result_queue.get(timeout=600)
            except queue.Empty as exc:
                raise RuntimeError("等待 TTS worker 启动超时。") from exc
            if not msg or msg[0] != "ready":
                continue
            _, worker_id, gpu_id, error = msg
            if error:
                self.close()
                raise RuntimeError(f"TTS worker {worker_id} (GPU {gpu_id}) 启动失败: {error}")
            ready_count += 1
            logging.info(f"✅ TTS worker {worker_id} 已就绪，绑定 GPU {gpu_id}")

        logging.info(
            f"MultiGPUPTSynthesizer 就绪，共 {self.num_devices} 张 GPU，每卡 batch={self.per_device_batch}"
        )

        self._result_listener_thread = threading.Thread(target=self._result_listener, daemon=True)
        self._result_listener_thread.start()

        # 每张卡一个独立 dispatcher 线程
        for dev_idx in range(self.num_devices):
            t = threading.Thread(
                target=self._per_device_dispatcher,
                args=(dev_idx,),
                daemon=True,
            )
            t.start()
            self._dispatcher_threads.append(t)

    # ── 提交 ─────────────────────────────────────────────────

    def _dev_idx_for_session(self, session_name: str) -> int:
        """根据 session_name 的 hash 值，将该 session 固定路由到某张卡。"""
        return stable_int(session_name) % self.num_devices

    def submit_job(self, job: dict) -> concurrent.futures.Future:
        """
        提交一条 TTS job，返回 Future，结果为 (audio_bytes, duration_sec)。
        job 字段：request_id, text, language, speaker, instruct, session_name
        同一 session 的所有 utterance 路由到同一张卡（按 session_name hash）。
        """
        if self._stopped:
            raise RuntimeError("MultiGPUPTSynthesizer 已关闭，无法提交任务。")
        fut = concurrent.futures.Future()
        with self._futures_lock:
            self._futures[job["request_id"]] = (fut, 0)
        with self._pending_futures_lock:
            self._pending_futures_count += 1
            self._all_futures_done_event.clear()
        # 按 session_name hash 路由到固定卡的专属队列
        session_name = job.get("session_name", job["request_id"])
        dev_idx = self._dev_idx_for_session(session_name)
        self._device_pending_queues[dev_idx].put(job)
        return fut

    def wait_all(self):
        """
        等待所有已提交 job 全部完成（含重试），然后通知 dispatcher 退出。
        必须先等所有 Future done，再发 sentinel，以防 sentinel 把重试 job 截断。
        """
        # 1. 等所有 Future 都被 resolve（成功/失败均算完成）
        with self._pending_futures_lock:
            needs_wait = self._pending_futures_count > 0
            logging.info("[wait_all] 等待 %d 个 Future 完成", self._pending_futures_count)
        if needs_wait:
            self._all_futures_done_event.wait()
        logging.info("[wait_all] 所有 Future 已完成，发送 sentinel...")
        # 2. 向每张卡的专属队列各发一个 sentinel，各自的 dispatcher 消费后退出
        for dq in self._device_pending_queues:
            dq.put(None)
        for t in self._dispatcher_threads:
            t.join()
        logging.info("[wait_all] 所有 dispatcher 已退出。")

    # ── 关闭 ─────────────────────────────────────────────────

    def close(self):
        if self._stopped:
            return
        self._stopped = True
        # 向每卡专属队列各发 sentinel，唤醒阻塞中的 dispatcher
        for dq in self._device_pending_queues:
            try:
                dq.put_nowait(None)
            except Exception:
                pass
        for tq in self._worker_task_queues:
            try:
                tq.put_nowait(None)
            except Exception:
                pass
        for proc in self._workers:
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()
        try:
            self._result_queue.put_nowait(None)
        except Exception:
            pass
        if self._result_listener_thread:
            self._result_listener_thread.join(timeout=5)

    # ── 内部：计数辅助 ───────────────────────────────────────────

    def _decrement_pending_futures(self):
        """每次 Future 最终 resolve（成功/失败）时调用，递减计数并在归零时设置 Event。"""
        with self._pending_futures_lock:
            self._pending_futures_count -= 1
            remaining = self._pending_futures_count
            if self._pending_futures_count <= 0:
                self._all_futures_done_event.set()
        logging.debug("[Decrement] 剩余 Future: %d", remaining)

    # ── 内部：per-device 调度线程 ─────────────────────────────
    # 每张 GPU 各一个线程，独立从 pending_queue 攒批，各自管控双缓冲 Semaphore

    def _next_batch_id(self) -> str:
        with self._batch_id_lock:
            self._batch_id_counter += 1
            return f"batch_{self._batch_id_counter:06d}"

    def _per_device_dispatcher(self, dev_idx: int):
        """每张卡专属的 dispatcher 线程，只消费本卡的专属 pending 队列。"""
        gpu_id = self.gpu_ids[dev_idx]
        tq = self._worker_task_queues[dev_idx]
        my_queue = self._device_pending_queues[dev_idx]
        # 双缓冲：允许同时有 2 个 batch 在推理（一个推理中，一个已下发等推理槽释放）
        inflight_sem = threading.Semaphore(2)
        # 该卡所有在途 batch 的 saved_jobs（用于重试）
        local_inflight: dict = {}  # batch_id -> jobs list

        # 从本卡专属队列中取一个 job，返回 _EMPTY 表示超时，None 表示 sentinel
        def _get_next_job(timeout):
            """从本卡专属队列取 job，无锁竞争。"""
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return _EMPTY
                try:
                    return my_queue.get_nowait()
                except queue.Empty:
                    pass
                time.sleep(0.005)

        def _flush(jobs):
            """将一批 jobs 下发给本卡，使用双缓冲 Semaphore 控制在途数量。"""
            if not jobs:
                return
            batch_id = self._next_batch_id()
            with self._inflight_lock:
                self._inflight_batches[batch_id] = {
                    "jobs": jobs,
                    "dev_idx": dev_idx,
                    "sem": inflight_sem,
                }
            local_inflight[batch_id] = jobs
            inflight_sem.acquire()  # 双缓冲：最多 2 个 batch 同时在途
            logging.info(
                "[Dispatcher-GPU%d] 下发 batch_id=%s, jobs=%d",
                gpu_id, batch_id, len(jobs),
            )
            tq.put(("batch", batch_id, jobs))

        accumulated: list = []
        last_arrival = time.monotonic()
        sentinel_received = False

        while not sentinel_received and not self._stopped:
            job = _get_next_job(timeout=0.05)
            if job is _EMPTY:
                # 超时检查
                now = time.monotonic()
                if accumulated and (now - last_arrival) >= self.accumulate_timeout:
                    logging.info(
                        "[Dispatcher-GPU%d] 超时 flush，jobs=%d", gpu_id, len(accumulated)
                    )
                    _flush(list(accumulated))
                    accumulated = []
                continue

            if job is None:
                # sentinel
                sentinel_received = True
                break

            req_id = job["request_id"]
            with self._futures_lock:
                entry = self._futures.get(req_id)
            if entry is None:
                continue
            fut, _ = entry
            if fut.cancelled():
                with self._futures_lock:
                    self._futures.pop(req_id, None)
                continue

            accumulated.append(job)
            last_arrival = time.monotonic()

            if len(accumulated) >= self.per_device_batch:
                logging.info(
                    "[Dispatcher-GPU%d] 攒满 %d 条，flush", gpu_id, len(accumulated)
                )
                _flush(list(accumulated))
                accumulated = []

        # sentinel 收到后，把剩余攒批也 flush 出去
        if accumulated:
            logging.info(
                "[Dispatcher-GPU%d] sentinel 后 flush 剩余 jobs=%d", gpu_id, len(accumulated)
            )
            _flush(list(accumulated))

        # 等待本卡所有 inflight batch 完成（获取 2 次信号量代表都完成）
        for _ in range(2):
            inflight_sem.acquire()
        logging.info("[Dispatcher-GPU%d] 所有批次已完成，dispatcher 退出。", gpu_id)

    # ── 内部：worker 重启 ──────────────────────────────────────

    def _restart_worker(self, dev_idx: int):
        """检测到 dev_idx 号 worker 死亡时，重启该进程并将其在途 batch 重新入队。"""
        with self._worker_restart_lock:
            old_proc = self._workers[dev_idx]
            if old_proc.is_alive():
                return  # 已被其他线程重启
            gpu_id = self.gpu_ids[dev_idx]
            logging.error(
                "[WatchDog] GPU %d worker (pid=%s) 已死亡 (exitcode=%s)，准备重启。",
                gpu_id, old_proc.pid, old_proc.exitcode,
            )
            # 清空旧的 task_queue，防止遗留消息
            old_tq = self._worker_task_queues[dev_idx]
            while True:
                try:
                    old_tq.get_nowait()
                except Exception:
                    break

            # 将该 GPU 所有在途 batch 的 job 重新推回对应卡的专属队列
            rescued = 0
            with self._inflight_lock:
                dead_batch_ids = [
                    bid for bid, info in self._inflight_batches.items()
                    if info["dev_idx"] == dev_idx
                ]
                for bid in dead_batch_ids:
                    info = self._inflight_batches.pop(bid)
                    for job in info["jobs"]:
                        # 仍然路由到原卡（session_name hash 不变）
                        sn = job.get("session_name", job["request_id"])
                        target_dev = self._dev_idx_for_session(sn)
                        self._device_pending_queues[target_dev].put(job)
                        rescued += 1
                    # 释放 Semaphore，让 dispatcher 继续（重启后会重新 acquire）
                    info["sem"].release()
            logging.warning(
                "[WatchDog] GPU %d: 已将 %d 条在途 job 重新入队。", gpu_id, rescued
            )

            # 启动新进程
            new_proc = self._ctx.Process(
                target=_pt_tts_worker_main,
                args=(dev_idx, gpu_id, self.custom_model_path,
                      self.base_model_path,
                      self.agent_ref_audio, self.agent_ref_text,
                      old_tq, self._result_queue),
                daemon=True,
            )
            new_proc.start()
            self._workers[dev_idx] = new_proc
            logging.info("[WatchDog] GPU %d: 新 worker 已启动 (pid=%s)，等待就绪...", gpu_id, new_proc.pid)

    # ── 内部：结果监听线程 ────────────────────────────────────

    def _result_listener(self):
        # 上次巡检时间（每 5 秒检查一次所有 worker 进程是否存活）
        last_watchdog_time = time.monotonic()

        while True:
            try:
                msg = self._result_queue.get(timeout=1.0)
            except queue.Empty:
                if self._stopped:
                    break
                # ── watchdog：检测 worker 进程是否存活 ──
                now = time.monotonic()
                if now - last_watchdog_time >= 5.0:
                    last_watchdog_time = now
                    for dev_idx, proc in enumerate(self._workers):
                        if not proc.is_alive() and not self._stopped:
                            threading.Thread(
                                target=self._restart_worker,
                                args=(dev_idx,),
                                daemon=True,
                            ).start()
                continue
            if msg is None:
                break
            msg_type = msg[0]
            # 新 worker 就绪通知（重启后发来）
            if msg_type == "ready":
                _, worker_id, gpu_id, error = msg
                if error:
                    logging.error("[WatchDog] 重启的 worker %d (GPU %d) 启动失败: %s", worker_id, gpu_id, error)
                else:
                    logging.info("[WatchDog] ✅ 重启的 worker %d (GPU %d) 已就绪。", worker_id, gpu_id)
                continue
            if msg_type != "result":
                continue
            msg_type, batch_id, results, batch_error = msg

            with self._inflight_lock:
                batch_info = self._inflight_batches.pop(batch_id, None)
            if batch_info is None:
                logging.warning("[ResultListener] 未知 batch_id: %s", batch_id)
                continue

            saved_jobs = batch_info["jobs"]
            sem = batch_info["sem"]

            logging.debug("[ResultListener] 处理 batch_id=%s, results=%d", batch_id, len(results))
            for res in results:
                req_id = res["request_id"]
                audio_bytes = res.get("audio_bytes")
                duration_sec = res.get("duration_sec", 0.0)
                error_str = res.get("error")

                with self._futures_lock:
                    entry = self._futures.get(req_id)
                if entry is None:
                    logging.warning("[ResultListener] req_id=%s 不在 futures 字典中，跳过", req_id)
                    continue
                fut, retry_count = entry

                if error_str and audio_bytes is None:
                    if retry_count < self.max_retries:
                        logging.warning(
                            "[ResultListener] req %s 失败（%s），重试 %d/%d",
                            req_id, error_str, retry_count + 1, self.max_retries,
                        )
                        with self._futures_lock:
                            self._futures[req_id] = (fut, retry_count + 1)
                        retry_job = next((j for j in saved_jobs if j["request_id"] == req_id), None)
                        if retry_job:
                            # 仍然路由到同一张卡（session_name hash 不变）
                            sn = retry_job.get("session_name", retry_job["request_id"])
                            target_dev = self._dev_idx_for_session(sn)
                            self._device_pending_queues[target_dev].put(retry_job)
                            # 重试不算完成，不递减计数
                        else:
                            logging.error(
                                "[ResultListener] 无法找到 req %s 的原始 job，标记失败", req_id
                            )
                            with self._futures_lock:
                                self._futures.pop(req_id, None)
                            if not fut.done():
                                fut.set_exception(
                                    RuntimeError(f"TTS 失败且无法重试: {error_str}")
                                )
                            self._decrement_pending_futures()
                    else:
                        logging.error(
                            "[ResultListener] req %s 重试耗尽，标记失败: %s", req_id, error_str
                        )
                        with self._futures_lock:
                            self._futures.pop(req_id, None)
                        if not fut.done():
                            fut.set_exception(RuntimeError(f"TTS 失败: {error_str}"))
                        self._decrement_pending_futures()
                else:
                    with self._futures_lock:
                        self._futures.pop(req_id, None)
                    if not fut.done():
                        fut.set_result((audio_bytes, duration_sec))
                    else:
                        logging.warning("[ResultListener] req_id=%s 已 done，跳过设置结果", req_id)
                    self._decrement_pending_futures()

            # 本批完成，释放该卡的双缓冲 Semaphore，dispatcher 可立即下发下一批
            sem.release()


# ─────────────────────────────────────────────────────────────
#  业务逻辑
# ─────────────────────────────────────────────────────────────

def scan_clone_profiles(prompt_dir: str, prompt_text: str):
    root = Path(prompt_dir)
    profiles = []
    if not root.exists():
        return profiles

    def first_audio(path: Path):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            return path
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS:
                    return candidate
        return None

    for child in sorted(root.iterdir()):
        audio = first_audio(child)
        if audio is None:
            continue
        profile_id = child.stem if child.is_file() else child.name
        profiles.append(
            {
                "profile_id": safe_name(profile_id, "clone_profile"),
                "prompt_audio_path": str(audio.resolve()),
                "prompt_text": prompt_text,
            }
        )
    return profiles


def build_voice_plan(sample, config: RuntimeConfig, clone_profiles):
    sample_key = sample.get("sample_id") or sample.get("session_name") or "sample"
    rng_value = stable_int(sample_key)

    if config.user_custom_ratio < 1.0 and not clone_profiles:
        raise RuntimeError(
            f"需要 Base clone 声线配比，但目录 {config.user_clone_prompt_dir} 未提供任何录音。"
        )

    agent_plan = {
        "backend": "custom_voice",
        "speaker": config.agent_speaker,
        "instruct": config.agent_instruct,
        "model_path": config.custom_model_path,
        "profile_type": "custom_preset",
    }

    use_custom = (rng_value % 10000) / 10000.0 < config.user_custom_ratio
    if use_custom:
        # 检测 session 语言：取 prompt 或前几条 dialogue 的文字
        _lang_text = sample.get("prompt") or ""
        if not _lang_text:
            _msgs = sample.get("dialogue") or []
            _lang_text = " ".join(
                (m.get("content") or "") for m in _msgs[:4] if isinstance(m, dict)
            )
        _pool = _SPEAKERS_EN if detect_language(_lang_text) == "English" else _SPEAKERS_ZH
        speaker = _pool[rng_value % len(_pool)]
        user_plan = {
            "backend": "custom_voice",
            "profile_type": "custom_preset",
            "speaker": speaker,
            "model_path": config.custom_model_path,
        }
    else:
        choice = clone_profiles[rng_value % len(clone_profiles)]
        user_plan = {
            "backend": "base_clone",
            "profile_type": "base_real_clone",
            "profile_id": choice["profile_id"],
            "prompt_audio_path": choice["prompt_audio_path"],
            "prompt_text": choice["prompt_text"],
            "model_path": config.base_model_path,
        }
    return {"agent": agent_plan, "user": user_plan}


def get_system_prompt(sample):
    metadata = sample.get("metadata") or {}
    if sample.get("system_prompt"):
        return sample["system_prompt"]
    if sample.get("writer_system"):
        return sample["writer_system"]
    if metadata.get("writer_system"):
        return metadata["writer_system"]
    messages = sample.get("messages") or []
    for message in messages:
        if message.get("role") == "system":
            return message.get("content", "")
    return ""


def get_original_system(sample):
    metadata = sample.get("metadata") or {}
    return sample.get("original_system") or metadata.get("original_system") or ""


def get_seed_summary(sample):
    metadata = sample.get("metadata") or {}
    return sample.get("seed_summary") or metadata.get("seed_summary") or {}


def normalize_dialogue_messages(sample):
    if sample.get("messages"):
        dialogue = []
        for message in sample["messages"]:
            role = message.get("role")
            if role == "system":
                continue
            if role == "assistant":
                dialogue.append({"role": "Agent", "text": message.get("content", "")})
            elif role == "user":
                dialogue.append({"role": "User", "text": message.get("content", "")})
        return dialogue
    if sample.get("labeled_dialogue"):
        dialogue = []
        for message in sample["labeled_dialogue"]:
            role = message.get("role")
            text = message.get("text", "")
            if role in {"Agent", "assistant"}:
                dialogue.append({"role": "Agent", "text": text})
            elif role in {"User", "user"}:
                dialogue.append({"role": "User", "text": text})
        return dialogue
    raise RuntimeError("样本中没有 messages 或 labeled_dialogue")


def normalize_sample(raw_item, index, config: RuntimeConfig, clone_profiles):
    sample_id = raw_item.get("sample_id") or f"sample_{index:08d}"
    split = raw_item.get("split") or raw_item.get("dataset_split") or "unknown"
    session_name = safe_name(sample_id.replace(":", "_"), f"session_{index:08d}")
    sample = {
        "sample_id": sample_id,
        "split": split,
        "prompt_id": raw_item.get("prompt_id") or "",
        "prompt": raw_item.get("prompt") or "",
        "system_prompt": get_system_prompt(raw_item),
        "original_system": get_original_system(raw_item),
        "seed_summary": get_seed_summary(raw_item),
        "dialogue": normalize_dialogue_messages(raw_item),
        "session_name": session_name,
    }
    sample["voice_plan"] = build_voice_plan(sample, config, clone_profiles)
    return sample


def resolve_input_paths(input_path: str):
    raw_parts = [part.strip() for part in str(input_path).split(",") if part.strip()]
    if not raw_parts:
        raise ValueError("--input-path 不能为空")

    resolved_paths = []
    for raw_part in raw_parts:
        path = Path(raw_part)
        if not path.exists():
            raise FileNotFoundError(f"输入路径不存在: {raw_part}")
        if path.is_dir():
            found_in_dir = False
            for filename in ("inbound.director.jsonl", "outbound.director.jsonl"):
                candidate = path / filename
                if candidate.exists():
                    resolved_paths.append(candidate)
                    found_in_dir = True
            if not found_in_dir:
                raise FileNotFoundError(f"目录内未找到 inbound/outbound director 数据: {raw_part}")
        else:
            resolved_paths.append(path)

    unique_paths = []
    seen = set()
    for path in resolved_paths:
        path_str = str(path.resolve())
        if path_str not in seen:
            seen.add(path_str)
            unique_paths.append(path)
    return unique_paths


_URL_PATTERN = re.compile(
    r"(?:https?://|ftp://|www\.)\S+",
    re.IGNORECASE,
)


def _sample_has_url(raw_item) -> bool:
    """检查样本中所有对话 turn 的文本是否含有 URL，有则返回 True（整个样本应被丢弃）。"""
    dialogue = raw_item.get("conversations") or raw_item.get("messages") or raw_item.get("dialogue") or []
    for turn in dialogue:
        if isinstance(turn, dict):
            text = turn.get("content") or turn.get("text") or turn.get("value") or ""
        else:
            text = str(turn)
        if _URL_PATTERN.search(text):
            return True
    # 同时检查 prompt / system_prompt 等顶层字段
    for field in ("prompt", "system_prompt", "system"):
        text = raw_item.get(field) or ""
        if isinstance(text, str) and _URL_PATTERN.search(text):
            return True
    return False


def load_samples(input_path: str, config: RuntimeConfig):
    clone_profiles = scan_clone_profiles(config.user_clone_prompt_dir, config.user_clone_prompt_text)
    samples = []
    sample_index = 0
    skipped_url = 0
    seen_session_names = set()
    for path in resolve_input_paths(input_path):
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if _sample_has_url(item):
                        skipped_url += 1
                        logging.debug(f"样本含 URL，已跳过: {item.get('sample_id', sample_index)}")
                        continue
                    sample = normalize_sample(item, sample_index, config, clone_profiles)
                    session_name = sample["session_name"]
                    if session_name in seen_session_names:
                        logging.warning("检测到重复 session_name=%s (sample_id=%s)", session_name, sample.get("sample_id", ""))
                    seen_session_names.add(session_name)
                    samples.append(sample)
                    sample_index += 1
                    if config.max_samples > 0 and len(samples) >= config.max_samples:
                        if skipped_url:
                            logging.info(f"URL 过滤：共跳过 {skipped_url} 条含 URL 的样本。")
                        return samples
            continue

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("data") or [payload]
        for item in payload:
            if _sample_has_url(item):
                skipped_url += 1
                logging.debug(f"样本含 URL，已跳过: {item.get('sample_id', sample_index)}")
                continue
            sample = normalize_sample(item, sample_index, config, clone_profiles)
            session_name = sample["session_name"]
            if session_name in seen_session_names:
                logging.warning("检测到重复 session_name=%s (sample_id=%s)", session_name, sample.get("sample_id", ""))
            seen_session_names.add(session_name)
            samples.append(sample)
            sample_index += 1
            if config.max_samples > 0 and len(samples) >= config.max_samples:
                if skipped_url:
                    logging.info(f"URL 过滤：共跳过 {skipped_url} 条含 URL 的样本。")
                return samples
    if skipped_url:
        logging.info(f"URL 过滤：共跳过 {skipped_url} 条含 URL 的样本。")
    # ── 多机分片：按顺序下标取模，保证每个节点处理不重叠的 session ──
    if config.nnodes > 1:
        samples = [s for i, s in enumerate(samples) if i % config.nnodes == config.node_rank]
        logging.info(
            f"多机分片：RANK {config.node_rank}/{config.nnodes}，本节点处理 {len(samples)} 条样本。"
        )
    return samples


def build_parsed_dialogue(sample):
    parsed_dialogue = []
    voice_plan = sample["voice_plan"]
    for idx, item in enumerate(sample["dialogue"]):
        role = item["role"]
        raw_text = item.get("text", "")
        parsed = parse_special_tags(raw_text)
        utterance_voice_plan = voice_plan["agent"] if role == "Agent" else voice_plan["user"]
        parsed_dialogue.append(
            {
                "index": idx,
                "role": role,
                "raw_text": raw_text,
                "clean_text": parsed["clean_text"],
                "display_text": parsed["display_text"],
                "tts_text": parsed["clean_text"],
                "full_text": parsed["clean_text"],
                "pend_chunks": parsed["pend_chunks"],
                "triggers": parsed["triggers"],
                "cut_position": parsed["cut_position"],
                "s2_messages": parsed["s2_messages"],
                "s2_controls": parsed["s2_controls"],
                "ghost_text": parsed["ghost_text"],
                "voice_plan": utterance_voice_plan,
                "sample_id": sample["sample_id"],
                "session_name": sample["session_name"],
            }
        )
    return parsed_dialogue


def apply_alignment_result(item, aligned_char_timestamps, aligned_duration):
    """
    将 align 结果写入 item。
    音频在 TTS 完成后已经 pad 到 CHUNK_SAMPLES 整数倍，无需再裁剪或 pad。
    duration 直接使用 pad 后的值（已在 TTS 时写入 item["duration"]）。
    """
    item["aligned_char_timestamps"] = aligned_char_timestamps
    item["aligned_duration"] = aligned_duration


# ─────────────────────────────────────────────────────────────
#  时间戳辅助函数
# ─────────────────────────────────────────────────────────────

def _aligned_time_for_position(aligned_char_timestamps, position):
    if not aligned_char_timestamps:
        raise RuntimeError("缺少 aligned_char_timestamps")
    acc = 0
    for char_info in aligned_char_timestamps:
        acc += len(char_info["text"])
        if acc >= position:
            return char_info["end_sec"]
    return aligned_char_timestamps[-1]["end_sec"]


def _aligned_start_time_for_position(aligned_char_timestamps, position):
    if not aligned_char_timestamps:
        raise RuntimeError("缺少 aligned_char_timestamps")
    if position <= 0:
        return aligned_char_timestamps[0]["start_sec"]
    acc = 0
    for idx, char_info in enumerate(aligned_char_timestamps):
        next_acc = acc + len(char_info["text"])
        if position < next_acc:
            return char_info["start_sec"]
        if position == next_acc:
            if idx + 1 < len(aligned_char_timestamps):
                return aligned_char_timestamps[idx + 1]["start_sec"]
            return char_info["end_sec"]
        acc = next_acc
    return aligned_char_timestamps[-1]["end_sec"]


def _tts_visible_chunk_for_abs_time(abs_time, chunk_dur_sec):
    return max(0, int(abs_time / chunk_dur_sec) - 1)


def get_char_timestamps(utterance):
    """
    返回绝对时间戳列表。必须已有 aligned_char_timestamps，否则直接 raise。
    不做任何 fallback。
    """
    if utterance.get("aligned_char_timestamps"):
        base = utterance["start_time"]
        return [
            {
                **char_info,
                "start_sec": char_info["start_sec"] + base,
                "end_sec": char_info["end_sec"] + base,
                "start_ms": (char_info["start_sec"] + base) * 1000,
                "end_ms": (char_info["end_sec"] + base) * 1000,
            }
            for char_info in utterance["aligned_char_timestamps"]
        ]
    if utterance.get("tts_text"):
        raise RuntimeError(
            f"非空文本缺少对齐结果，无 fallback: text={utterance['tts_text'][:80]!r}"
        )
    return []


def build_aligned_tts_output_blocks(utterance, chunk_dur_sec):
    char_timestamps = get_char_timestamps(utterance)
    if not char_timestamps:
        return []
    block_map = {}
    for char_info in char_timestamps:
        start_time = char_info["start_sec"]
        if utterance["cut_time"] is not None and start_time >= utterance["cut_time"]:
            continue
        p_start = int(start_time / chunk_dur_sec)
        target_chunk = max(0, p_start - 1)
        block_map[target_chunk] = block_map.get(target_chunk, "") + char_info["text"]
    return [{"chunk_id": chunk_id, "text": text} for chunk_id, text in sorted(block_map.items()) if text]


# ─────────────────────────────────────────────────────────────
#  后处理 + 落盘
# ─────────────────────────────────────────────────────────────

def postprocess_and_save(valid_items, sample, session_name, session_dir, output_dir):
    if not valid_items:
        raise RuntimeError(f"{session_name} 没有任何有效 utterance")

    random.seed((hash(session_name) % (2**32)) + int(time.time() * 1_000_000))
    logging.info(f"🧮 [{session_name}] 正在计算时序...")
    role_last_end_time = {"User": 0.0, "Agent": 0.0}
    trigger_pool = []
    final_start_times = [0.0] * len(valid_items)
    final_end_times = [0.0] * len(valid_items)
    cut_times = [None] * len(valid_items)

    for idx, item in enumerate(valid_items):
        current_role = item["role"]
        my_earliest_start = role_last_end_time.get(current_role, 0.0)
        trigger_time = 0.0
        found_trigger = False
        if idx == 0:
            found_trigger = True
        else:
            for trigger in trigger_pool:
                if not trigger["used"] and trigger["source_role"] != current_role:
                    trigger_time = trigger["time"]
                    trigger["used"] = True
                    found_trigger = True
                    break
        if not found_trigger and idx > 0:
            trigger_time = max(final_end_times[:idx]) if idx > 0 else 0.0

        actual_start_time = max(trigger_time, my_earliest_start)
        actual_start_time += item.get("pend_chunks", 0) * (CHUNK_MS / 1000)

        if (
            current_role == "User"
            and idx > 0
            and valid_items[idx - 1]["role"] == "Agent"
            and not (valid_items[idx - 1].get("triggers") or [])
        ):
            rng = random.Random((hash(session_name) + idx) % (2**32))
            actual_start_time += rng.uniform(0.2, 3.0)

        if (
            idx > 0
            and actual_start_time < final_end_times[idx - 1]
            and valid_items[idx - 1].get("cut_position") is not None
        ):
            actual_start_time = max(0.0, actual_start_time - CHUNK_MS / 1000)

        # 计算 [CUT] 对应的绝对时间：由 force-aligner 对 cut_position 字符位置定位，而非音频总时长
        if item["cut_position"] is not None:
            aligned_ts = item.get("aligned_char_timestamps") or []
            if aligned_ts:
                cut_position = item["cut_position"]
                # 检查 ^ 和 [CUT] 是否紧邻：如果任意 trigger_pos >= cut_pos - 2（即 [CUT] 在 ^ 前 0~2 个字符内）
                # 紧邻时，cut_position 向后 pend 1~2 个字符，模拟打断延迟
                trigger_positions = item.get("triggers") or []
                is_adjacent = any(
                    (cut_position - tpos) <= 2 and (cut_position - tpos) >= 0
                    for tpos in trigger_positions
                )
                if is_adjacent:
                    # 向后 pend 2 个字符（即 cut_position + 2，不超过文本长度）
                    total_chars = sum(len(c["text"]) for c in aligned_ts)
                    cut_position = min(cut_position + 2, total_chars)
                rel_cut_time = _aligned_time_for_position(aligned_ts, cut_position)
            else:
                rel_cut_time = item.get("duration", 0.0)
            cut_times[idx] = actual_start_time + rel_cut_time
            final_end_times[idx] = cut_times[idx]
        else:
            final_end_times[idx] = actual_start_time + item.get("duration", 0.0)

        final_start_times[idx] = actual_start_time
        role_last_end_time[current_role] = final_end_times[idx]

        for trigger_pos in item.get("triggers") or []:
            if item.get("duration", 0.0) <= 0:
                continue
            rel_time = _aligned_time_for_position(item["aligned_char_timestamps"], trigger_pos)
            abs_time = actual_start_time + rel_time
            trigger_pool.append({"time": abs_time, "source_role": current_role, "used": False})

    logging.info(f"🔨 [{session_name}] 正在混音...")
    total_duration = max(final_end_times) if final_end_times else 0.0
    total_samples = max(1, int(total_duration * SAMPLE_RATE) + SAMPLE_RATE)
    mixed_audio = np.zeros(total_samples, dtype=np.int16)
    role1_audio = np.zeros(total_samples, dtype=np.int16)
    role2_audio = np.zeros(total_samples, dtype=np.int16)
    role1_utterances = []
    role2_utterances = []

    for idx, item in enumerate(valid_items):
        pcm_data = np.frombuffer(item.get("audio_data", b""), dtype=np.int16)
        start_sample = int(final_start_times[idx] * SAMPLE_RATE)

        # 对于有 [CUT] 的 Agent 语音，混音时只使用截断点之前的 PCM
        # cut_times[idx] 是绝对时间（由 aligner 精确确定），相对于帧起点的偏移量即实际有效采样数
        if cut_times[idx] is not None:
            # 截断到 cut_time 对应的相对采样数
            rel_cut_samples = int((cut_times[idx] - final_start_times[idx]) * SAMPLE_RATE)
            pcm_data = pcm_data[:rel_cut_samples]

        end_sample = start_sample + len(pcm_data)
        if end_sample > total_samples:
            end_sample = total_samples
            pcm_data = pcm_data[: end_sample - start_sample]
        if len(pcm_data) > 0:
            mixed_audio[start_sample:end_sample] += pcm_data
        if item["role"] == "User":
            if len(pcm_data) > 0:
                role1_audio[start_sample:end_sample] += pcm_data
            role1_utterances.append(
                {
                    "index": idx,
                    "text": item["clean_text"],
                    "display_text": item["display_text"],
                    "start_time": final_start_times[idx],
                    "end_time": final_end_times[idx],
                    "duration": item.get("duration", 0.0),
                    "audio_data": item.get("audio_data", b""),
                    "s2_controls": item["s2_controls"],
                    "s2_messages": item["s2_messages"],
                    "aligned_char_timestamps": item.get("aligned_char_timestamps", []),
                    "pend_chunks": item.get("pend_chunks", 0),
                }
            )
        else:
            if len(pcm_data) > 0:
                role2_audio[start_sample:end_sample] += pcm_data
            role2_utterances.append(
                {
                    "index": idx,
                    "text": item["clean_text"],
                    "display_text": item["display_text"],
                    "full_text": item["full_text"],
                    "tts_text": item["tts_text"],
                    "start_time": final_start_times[idx],
                    "logic_start_time": final_start_times[idx],
                    "end_time": final_end_times[idx],
                    "duration": item.get("duration", 0.0),
                    "audio_data": item.get("audio_data", b""),
                    "s2_messages": item["s2_messages"],
                    "cut_time": cut_times[idx],
                    "aligned_char_timestamps": item.get("aligned_char_timestamps", []),
                    "pend_chunks": item.get("pend_chunks", 0),
                }
            )

    mixed_audio = np.clip(mixed_audio, -32768, 32767).astype(np.int16)
    role1_audio = np.clip(role1_audio, -32768, 32767).astype(np.int16)
    role2_audio = np.clip(role2_audio, -32768, 32767).astype(np.int16)

    logging.info(f"🔪 [{session_name}] 开始生成时间片数据集...")
    chunk_dur_sec = CHUNK_MS / 1000

    for utterance in role2_utterances:
        utterance["aligned_tts_blocks"] = build_aligned_tts_output_blocks(utterance, chunk_dur_sec)

    max_chunk_index = max(1, int(np.ceil(total_duration * 1000 / CHUNK_MS)))
    dataset = []
    for chunk_idx in range(max_chunk_index):
        chunk_start_sec = chunk_idx * chunk_dur_sec
        chunk_end_sec = (chunk_idx + 1) * chunk_dur_sec
        chunk_data = {
            "chunk_id": chunk_idx,
            "r1_audio_bytes": _extract_chunk_bytes(role1_audio, chunk_idx, chunk_start_sec, chunk_end_sec),
            "r2_audio_bytes": _extract_chunk_bytes(role2_audio, chunk_idx, chunk_start_sec, chunk_end_sec),
            "from_s2": "",
            "asr_context": [],
            "tts_control": "", "system2_control": "", "asr": "", "tts": "",
        }

        for utterance in role2_utterances:
            for s2_pos, s2_msg in utterance["s2_messages"]:
                rel = _aligned_start_time_for_position(utterance["aligned_char_timestamps"], s2_pos)
                injection_time = utterance["start_time"] + rel
                target_input_chunk_id = _tts_visible_chunk_for_abs_time(injection_time, chunk_dur_sec)
                if chunk_idx == target_input_chunk_id:
                    chunk_data["from_s2"] += s2_msg + " "

        for utterance in role1_utterances:
            if chunk_end_sec <= utterance["start_time"] or chunk_start_sec >= utterance["end_time"]:
                continue
            char_timestamps = get_char_timestamps(utterance)
            for s2_pos, s2_msg in utterance["s2_messages"]:
                injection_time = utterance["start_time"]
                accumulated_len = 0
                for char_info in char_timestamps:
                    char_len = len(char_info["text"])
                    if accumulated_len + char_len > s2_pos:
                        injection_time = char_info["start_sec"]
                        break
                    accumulated_len += char_len
                    if accumulated_len == s2_pos:
                        injection_time = char_info["end_sec"]
                        break
                target_chunk = int(injection_time // chunk_dur_sec)
                if chunk_idx == target_chunk:
                    chunk_data["from_s2"] += s2_msg + " "

        if chunk_idx > 0:
            asr_history = []
            for past_idx in range(max(0, chunk_idx - 10), chunk_idx):
                past_asr = dataset[past_idx]["asr"]
                if past_asr:
                    asr_history.append(past_asr)
            recent_3_chunks_asr = [dataset[past_idx]["asr"] for past_idx in range(max(0, chunk_idx - 3), chunk_idx)]
            if len(recent_3_chunks_asr) >= 3 and all(asr == "" for asr in recent_3_chunks_asr):
                chunk_data["asr_context"] = []
            else:
                chunk_data["asr_context"] = asr_history[-3:]

        for utterance in role1_utterances:
            char_timestamps = get_char_timestamps(utterance)
            for char_info in char_timestamps:
                start_time = char_info["start_sec"]
                end_time = char_info["end_sec"]
                p_start = int(start_time / chunk_dur_sec)
                p_end = int(end_time / chunk_dur_sec)
                target_chunk = p_end if p_start != p_end else p_start
                if target_chunk == chunk_idx:
                    chunk_data["asr"] += char_info["text"]

        for utterance in role1_utterances:
            char_timestamps = get_char_timestamps(utterance)
            total_text_len = sum(len(it["text"]) for it in char_timestamps)
            for s2_pos, s2_control in utterance["s2_controls"]:
                control_time = None
                if s2_pos >= total_text_len:
                    control_time = utterance["end_time"]
                else:
                    accumulated_chars = 0
                    for char_info in char_timestamps:
                        char_len = len(char_info["text"])
                        if accumulated_chars <= s2_pos < accumulated_chars + char_len:
                            control_time = char_info["end_sec"]
                            break
                        accumulated_chars += char_len
                if control_time is not None and int(control_time / chunk_dur_sec) == chunk_idx:
                    chunk_data["system2_control"] = f"[{s2_control}]"

        for utterance in role2_utterances:
            for block in utterance["aligned_tts_blocks"]:
                if block["chunk_id"] == chunk_idx:
                    chunk_data["tts"] += block["text"]

        for utterance in role2_utterances:
            if utterance["cut_time"] is None:
                continue
            cut_chunk = int(utterance["cut_time"] // chunk_dur_sec)
            if cut_chunk > 0:
                cut_chunk -= 1
            if chunk_idx == cut_chunk:
                chunk_data["tts_control"] = "[STOP]"

        chunk_data["from_s2"] = chunk_data["from_s2"].strip()
        dataset.append(chunk_data)

    voice_plan     = sample.get("voice_plan") or {}
    agent_speaker  = voice_plan.get("agent", {}).get("speaker", DEFAULT_AGENT_SPEAKER)
    user_speaker   = voice_plan.get("user",  {}).get("speaker", "")
    voice_plan_str = json.dumps(voice_plan, ensure_ascii=False)
    sample_id      = sample.get("sample_id", "")
    split          = sample.get("split", "")

    rows = []
    for cd in dataset:
        rows.append({
            "session_id":       session_name,
            "sample_id":        sample_id,
            "split":            split,
            "chunk_id":         int(cd["chunk_id"]),
            "time_start":       float(cd["chunk_id"] * CHUNK_MS / 1000),
            "time_end":         float((cd["chunk_id"] + 1) * CHUNK_MS / 1000),
            "r1_audio_bytes":   cd["r1_audio_bytes"],
            "r2_audio_bytes":   cd["r2_audio_bytes"],
            "from_s2":          cd["from_s2"],
            "asr_context":      ",".join(cd["asr_context"]) if isinstance(cd["asr_context"], list) else (cd["asr_context"] or ""),
            "asr":              cd["asr"],
            "tts":              cd["tts"],
            "tts_control":      cd["tts_control"],
            "system2_control":  cd["system2_control"],
            "voice_plan":       voice_plan_str,
            "agent_speaker":    agent_speaker,
            "user_speaker":     user_speaker,
        })

    if _BATCH_WRITER is not None:
        _BATCH_WRITER.add_session(session_name, rows)
    logging.info(f"✅ [{session_name}] 后台处理完成！({len(rows)} chunks)")

    # 落盘完成后立即释放音频内存，防止大规模运行时内存爆炸
    for item in valid_items:
        item["audio_data"] = b""


# ─────────────────────────────────────────────────────────────
#  主流水线：TTS ‖ Align ‖ 后处理落盘  —— 三阶段完全并发（纯线程）
# ─────────────────────────────────────────────────────────────

def main_sync(args):
    """
    主流水线（纯线程，无 asyncio）：
    - TTS 完成后，回调直接在 align_executor 线程池中提交 align 任务并等待结果
    - align 完成后检查 session 是否全部完成，若是则在 finalize_executor 中落盘
    - 三个阶段（TTS / Align / 落盘）完全异步，互不阻塞
    - 实时进度条：session 完成速度、预估剩余时间、平均 RTF
    """
    global _BATCH_WRITER
    config = RuntimeConfig(args)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.cut_dir, exist_ok=True)
    _BATCH_WRITER = BatchParquetWriter(config.cut_dir, sessions_per_batch=100)
    _done_sessions = _BATCH_WRITER.load_done_sessions() if config.resume else set()

    # ── 读取 User clone profiles ─────────────────────────────
    clone_profiles = scan_clone_profiles(config.user_clone_prompt_dir, config.user_clone_prompt_text)
    if clone_profiles:
        logging.info(
            "✅ 读取到 %d 个 User clone profile: %s",
            len(clone_profiles),
            ", ".join(p["profile_id"] for p in clone_profiles),
        )
    else:
        logging.info("ℹ️  未找到 User clone profile，User 侧将使用 custom_voice。")

    samples = load_samples(args.input_path, config)
    total_sessions = len(samples)
    logging.info(f"共加载 {total_sessions} 条样本。")

    # ── 统计 User 声线实际分配情况（含 custom_voice / base_clone 配比）──────────
    if clone_profiles:
        custom_count = 0
        clone_count = 0
        profile_counter: dict = {}
        for sample in samples:
            sn = sample.get("sample_id") or sample.get("session_name") or "sample"
            rng_value = stable_int(sn)
            use_custom = (rng_value % 10000) / 10000.0 < config.user_custom_ratio
            if use_custom:
                custom_count += 1
            else:
                clone_count += 1
                pid = clone_profiles[rng_value % len(clone_profiles)]["profile_id"]
                profile_counter[pid] = profile_counter.get(pid, 0) + 1
        logging.info(
            "User 声线分配: custom_voice=%d (%.1f%%) | base_clone=%d (%.1f%%)",
            custom_count, custom_count / total_sessions * 100,
            clone_count, clone_count / total_sessions * 100,
        )
        if profile_counter:
            logging.info(
                "User clone profile 分布（仅走 base_clone 的样本）: %s",
                " | ".join(f"{pid}:{cnt}" for pid, cnt in sorted(profile_counter.items())),
            )

    # ── 启动 TTS synthesizer 和 Aligner pool（并行启动）─────
    synthesizer = MultiGPUPTSynthesizer(
        gpu_ids=config.tts_gpu_ids,
        custom_model_path=config.custom_model_path,
        base_model_path=config.base_model_path,
        agent_ref_audio=config.agent_ref_audio,
        agent_ref_text=config.agent_ref_text,
        user_clone_prompts_data=clone_profiles,
        batch_size=config.tts_batch_size,
        accumulate_timeout=config.tts_accumulate_timeout,
        max_retries=config.tts_max_retries,
    )
    aligner_pool = ThreadedAlignerPool(config.aligner_gpu_ids, config.aligner_model_path)

    tts_start_event = threading.Event()
    aligner_start_event = threading.Event()
    tts_start_error = [None]
    aligner_start_error = [None]

    def _start_tts():
        try:
            synthesizer.start()
        except Exception as exc:
            tts_start_error[0] = exc
        finally:
            tts_start_event.set()

    def _start_aligner():
        try:
            aligner_pool.start()
        except Exception as exc:
            aligner_start_error[0] = exc
        finally:
            aligner_start_event.set()

    threading.Thread(target=_start_tts, daemon=True).start()
    threading.Thread(target=_start_aligner, daemon=True).start()
    tts_start_event.wait()
    aligner_start_event.wait()

    if tts_start_error[0]:
        raise RuntimeError(f"TTS synthesizer 启动失败: {tts_start_error[0]}") from tts_start_error[0]
    if aligner_start_error[0]:
        raise RuntimeError(f"Aligner pool 启动失败: {aligner_start_error[0]}") from aligner_start_error[0]

    logging.info("✅ TTS synthesizer 和 Aligner pool 均已就绪，开始流水线处理。")

    # ── 线程池：align 和落盘各有独立线程池，互不阻塞 ──────
    align_workers = len(config.aligner_gpu_ids) * 4
    align_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=align_workers, thread_name_prefix="align"
    )
    finalize_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=config.finalize_workers, thread_name_prefix="finalize"
    )

    # ── session 状态追踪（线程安全）──────────────────────────
    session_state: dict = {}
    session_state_lock = threading.Lock()

    # 用于等待所有落盘任务完成
    finalize_futures: list = []
    finalize_futures_lock = threading.Lock()

    # ── RTF & 进度统计（线程安全）────────────────────────────
    _stats_lock = threading.Lock()
    _total_audio_sec = [0.0]   # 累计生成音频总时长（秒）
    _total_wall_sec  = [0.0]   # 累计 TTS wall time（秒，用完成时间减提交时间近似）
    _sessions_done   = [0]     # 已落盘 session 数（含 resume 跳过）
    _pipeline_t0 = time.monotonic()

    # tqdm 进度条（如果可用）
    if _TQDM_AVAILABLE:
        _pbar = _tqdm_cls(
            total=total_sessions,
            desc="Sessions",
            unit="sess",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
    else:
        _pbar = None

    def _update_progress(delta_audio_sec=0.0, delta_wall_sec=0.0, delta_done=0):
        with _stats_lock:
            _total_audio_sec[0] += delta_audio_sec
            _total_wall_sec[0]  += delta_wall_sec
            _sessions_done[0]   += delta_done
            done = _sessions_done[0]
            avg_rtf = _total_wall_sec[0] / _total_audio_sec[0] if _total_audio_sec[0] > 0 else 0.0
        if _pbar is not None:
            _pbar.n = done
            _pbar.set_postfix(RTF=f"{avg_rtf:.3f}", refresh=True)
        elif done % 10 == 0 or done == total_sessions:
            elapsed = time.monotonic() - _pipeline_t0
            spd = done / elapsed if elapsed > 0 else 0.0
            eta = (total_sessions - done) / spd if spd > 0 else float("inf")
            logging.info(
                "进度: %d/%d session | 速度 %.2f sess/s | 预估剩余 %.0fs | 平均 RTF=%.3f",
                done, total_sessions, spd, eta, avg_rtf,
            )

    def _on_align_done_check_session(session_name, session_dir, cfg,
                                     item_audio_sec=0.0, item_wall_sec=0.0):
        """align 完成后，检查当前 session 是否所有 item 都 align 完毕，若是则触发落盘。"""
        with session_state_lock:
            state = session_state.get(session_name)
        if state is None:
            return
        with state["lock"]:
            state["aligned_count"] += 1
            state["accum_audio_sec"] += item_audio_sec
            state["accum_wall_sec"]  += item_wall_sec
            aligned_count = state["aligned_count"]
            need_align = state["need_align"]
            all_aligned = aligned_count >= need_align

        if all_aligned:
            with session_state_lock:
                state = session_state.pop(session_name, None)
            if state is None:
                return
            logging.debug("[Pipeline] session=%s 全部 align 完成，触发落盘", session_name)
            audio_sec = state["accum_audio_sec"]
            wall_sec  = state["accum_wall_sec"]
            fut = finalize_executor.submit(
                _finalize_session_sync,
                state["sample"], state["items"], session_dir, cfg,
                audio_sec, wall_sec, _update_progress,
            )
            with finalize_futures_lock:
                finalize_futures.append(fut)

    def _do_align_and_check(tts_future, item, session_name, session_dir, cfg, item_t0):
        """在 align_executor 线程中执行：等待 TTS 结果 → align → 检查落盘。"""
        try:
            audio_bytes, duration_sec = tts_future.result()
        except Exception as exc:
            logging.error("[Pipeline] TTS 失败，session=%s item=%d: %s",
                          session_name, item["index"], exc)
            with session_state_lock:
                evicted = session_state.pop(session_name, None)
            if evicted is not None:
                _update_progress(delta_done=1)
            return

        item_wall = time.monotonic() - item_t0
        item["audio_data"] = audio_bytes
        item["duration"] = duration_sec

        try:
            ts, dur = aligner_pool.submit_sync(
                audio_bytes, item["tts_text"], session_name, item["index"]
            )
        except Exception as exc:
            logging.error("[Pipeline] Align 失败，session=%s item=%d: %s",
                          session_name, item["index"], exc)
            with session_state_lock:
                evicted = session_state.pop(session_name, None)
            if evicted is not None:
                _update_progress(delta_done=1)
            return

        apply_alignment_result(item, ts, dur)
        _on_align_done_check_session(
            session_name, session_dir, cfg,
            item_audio_sec=duration_sec,
            item_wall_sec=item_wall,
        )

    try:
        # ── 对每个 sample 提交 TTS job ────────────────────────
        for sample in samples:
            session_dir = os.path.join(config.cut_dir, sample["session_name"])
            if config.resume and sample["session_name"] in _done_sessions:
                logging.debug("⏭️  %s 已完成（manifest），跳过。", sample["session_name"])
                _update_progress(delta_done=1)
                continue

            items = build_parsed_dialogue(sample)
            session_name = sample["session_name"]

            tts_needed = sum(1 for it in items if it["tts_text"])
            need_align = tts_needed

            if need_align == 0:
                for it in items:
                    it["audio_data"] = b""
                    it["aligned_char_timestamps"] = []
                    it["aligned_duration"] = 0.0
                    it["duration"] = 0.0
                fut = finalize_executor.submit(
                    _finalize_session_sync, sample, items, session_dir, config,
                    0.0, 0.0, _update_progress,
                )
                with finalize_futures_lock:
                    finalize_futures.append(fut)
                continue

            # 为该 session 选取 User clone profile（按 session_name hash 平均轮转）
            user_clone_profile_id = None
            if clone_profiles:
                user_clone_profile_id = clone_profiles[
                    stable_int(session_name) % len(clone_profiles)
                ]["profile_id"]

            with session_state_lock:
                session_state[session_name] = {
                    "sample": sample,
                    "items": items,
                    "need_align": need_align,
                    "aligned_count": 0,
                    "lock": threading.Lock(),
                    "accum_audio_sec": 0.0,
                    "accum_wall_sec": 0.0,
                }

            for it in items:
                if not it["tts_text"]:
                    it["audio_data"] = b""
                    it["aligned_char_timestamps"] = []
                    it["aligned_duration"] = 0.0
                    it["duration"] = 0.0
                    continue

                request_id = f"{session_name}__{it['index']:04d}"
                it["request_id"] = request_id
                voice_plan = it["voice_plan"]

                is_agent = it["role"] == "Agent"

                if is_agent:
                    # Agent：固定 Voice Clone（保证跨句音色一致性）
                    job = {
                        "request_id": request_id,
                        "tts_type": "agent_clone" if bool(config.agent_ref_audio) else "custom",
                        "text": it["tts_text"],
                        "language": detect_language(it["tts_text"]),
                        "speaker": voice_plan.get("speaker", config.agent_speaker),
                        "instruct": voice_plan.get("instruct", config.agent_instruct),
                        "session_name": session_name,
                    }
                elif voice_plan.get("backend") == "base_clone" and user_clone_profile_id:
                    # User：Base clone（按 user_custom_ratio 概率选中）
                    job = {
                        "request_id": request_id,
                        "tts_type": "user_clone",
                        "text": it["tts_text"],
                        "language": detect_language(it["tts_text"]),
                        "speaker": "",
                        "instruct": "",
                        "clone_profile_id": user_clone_profile_id,
                        "session_name": session_name,
                    }
                else:
                    # User：custom_voice（按 user_custom_ratio 概率选中，或无 clone profile 时 fallback）
                    job = {
                        "request_id": request_id,
                        "tts_type": "custom",
                        "text": it["tts_text"],
                        "language": detect_language(it["tts_text"]),
                        "speaker": voice_plan.get("speaker", config.user_custom_speakers[0]),
                        "instruct": "",
                        "session_name": session_name,
                    }

                item_t0 = time.monotonic()
                tts_future = synthesizer.submit_job(job)

                # 注册回调：TTS 完成后立即提交到 align_executor（非阻塞）
                def _schedule_align(fut, _it=it, _sn=session_name, _sd=session_dir, _t0=item_t0):
                    align_executor.submit(_do_align_and_check, fut, _it, _sn, _sd, config, _t0)

                tts_future.add_done_callback(_schedule_align)

        # ── 等待所有 TTS job 完成 ──────────────────────────────
        synthesizer.wait_all()
        logging.info("所有 TTS job 已完成，等待 align 和落盘...")

        # ── 等待 session_state 清空（所有 session align 完成）──
        while True:
            with session_state_lock:
                remaining = len(session_state)
            if remaining == 0:
                break
            time.sleep(1.0)

        # ── 等待所有落盘任务完成 ───────────────────────────────
        with finalize_futures_lock:
            futs_copy = list(finalize_futures)
        for fut in concurrent.futures.as_completed(futs_copy):
            try:
                fut.result()
            except Exception as exc:
                logging.exception("[Finalize] 落盘任务失败: %s", exc)

        if _pbar is not None:
            _pbar.close()

        if _BATCH_WRITER is not None:
            _BATCH_WRITER.flush_all()

        elapsed_total = time.monotonic() - _pipeline_t0
        with _stats_lock:
            avg_rtf = _total_wall_sec[0] / _total_audio_sec[0] if _total_audio_sec[0] > 0 else 0.0
            done = _sessions_done[0]
        logging.info(
            "✅ 所有任务处理完成！%d/%d session | 总耗时 %.1fs | 平均 RTF=%.3f",
            done, total_sessions, elapsed_total, avg_rtf,
        )

    finally:
        if _pbar is not None:
            _pbar.close()
        synthesizer.close()
        aligner_pool.close()
        align_executor.shutdown(wait=False)
        finalize_executor.shutdown(wait=False)


def _finalize_session_sync(sample, items, session_dir, config: RuntimeConfig,
                            audio_sec: float = 0.0, wall_sec: float = 0.0,
                            update_progress=None):
    """同步执行落盘，在 finalize_executor 线程中调用。落盘完成后回调进度更新。"""
    session_name = sample["session_name"]
    try:
        postprocess_and_save(items, sample, session_name, session_dir, config.output_dir)
    except Exception as exc:
        logging.exception("[Finalize] session=%s 落盘失败: %s", session_name, exc)
        raise
    finally:
        # 无论成功失败都推进计数（失败也算做了）
        if update_progress is not None:
            update_progress(delta_audio_sec=audio_sec, delta_wall_sec=wall_sec, delta_done=1)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Qwen3-TTS offline finalcut pipeline (PT batch inference)")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cut-dir", default=DEFAULT_CUT_DIR)
    parser.add_argument("--custom-model-path", default=DEFAULT_CUSTOM_MODEL_PATH)
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--aligner-model-path", default=DEFAULT_ALIGNER_MODEL_PATH)
    parser.add_argument("--aligner-gpu-ids", default="0,1,2,3,4,5,6,7",
                        help="逗号分隔的 GPU id 列表，用于 forced aligner（每张卡一个 aligner worker）")
    parser.add_argument("--tts-gpu-ids", default=DEFAULT_TTS_GPU_IDS,
                        help="逗号分隔的 GPU id 列表，用于 PT TTS 推理（每张卡一个模型进程）")
    parser.add_argument("--tts-batch-size", type=int, default=DEFAULT_TTS_BATCH_SIZE,
                        help="每张 GPU 上的 batch size（总 batch = tts-batch-size × num_gpus）")
    parser.add_argument("--tts-accumulate-timeout", type=float, default=DEFAULT_TTS_ACCUMULATE_TIMEOUT,
                        help="攒批超时秒数：超过此时间没有新请求则触发最后一批推理")
    parser.add_argument("--tts-max-retries", type=int, default=DEFAULT_TTS_MAX_RETRIES,
                        help="TTS 推理失败时的最大重试次数")
    parser.add_argument("--session-concurrency", type=int, default=32)
    parser.add_argument("--finalize-workers", type=int, default=16)
    parser.add_argument("--user-custom-ratio", type=float, default=DEFAULT_USER_CUSTOM_RATIO)
    parser.add_argument("--agent-speaker", default=DEFAULT_AGENT_SPEAKER)
    parser.add_argument("--agent-instruct", default=DEFAULT_AGENT_INSTRUCT)
    parser.add_argument("--agent-ref-audio", default=DEFAULT_AGENT_REF_AUDIO,
                        help="Agent 声线的参考音频路径，非空时启用 Voice Clone（跨句一致性更好）；空字符串则回退到 custom_voice")
    parser.add_argument("--agent-ref-text", default=DEFAULT_AGENT_REF_TEXT,
                        help="Agent 参考音频对应的文本转写")
    parser.add_argument("--user-custom-speakers", default=",".join(DEFAULT_USER_CUSTOM_SPEAKERS))
    parser.add_argument("--user-clone-prompt-dir", default=DEFAULT_CLONE_PROMPT_DIR)
    parser.add_argument("--user-clone-prompt-text", default=DEFAULT_CLONE_PROMPT_TEXT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="最大处理样本数（0 = 全部）")
    parser.add_argument("--node-rank", type=int, default=0,
                        help="当前节点编号（0-based），多机时由启动脚本传入")
    parser.add_argument("--nnodes", type=int, default=1,
                        help="总节点数，多机时由启动脚本传入；1 表示单机模式")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    main_sync(args)


if __name__ == "__main__":
    main()
