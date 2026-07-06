#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激情在燃烧！批量加噪脚本。

- 读取所有 user 音频，仅用「有声音部分」（音量非 0）计算响度基准与时长。
- 噪声：不循环，全随机拿取 + 随机拼接；每条数据随机选 MUSAN 或 FSD50K，每条加噪不同。
- 每条：整条 user 拼成一段，一次效果+混音后按原边界切分写 wav；多进程并行，不设 seed。
- 所有随机范围在下方配置区写清，便于调参。
"""

import os
os.environ["PEDALBOARD_NO_AVX"] = "1"

import io
import json
import random
import re
import time
import wave
import struct
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, **kwargs):
        return iterable

try:
    from pedalboard import (
        Pedalboard, Reverb, Chorus, Distortion, Phaser, Compressor,
        LowpassFilter, HighpassFilter, PitchShift
    )
    HAS_PEDALBOARD = True
except Exception:
    HAS_PEDALBOARD = False
    Pedalboard = Reverb = Chorus = Distortion = Phaser = Compressor = None
    LowpassFilter = HighpassFilter = PitchShift = None

# 全量加噪时与 build_background_noise 核心逻辑一致：复用其效果预设与 apply_audio_effect
try:
    import build_background_noise as _bbn
except Exception:
    _bbn = None

# 多进程 worker 内全局：每个进程初始化时构建一次效果池，避免 pickle Pedalboard
_WORKER_PRESETS = None
_WORKER_USE_BBN = None
_WORKER_BASELINE_RMS = None
_WORKER_MUSAN_PATHS = None
_WORKER_FSD_PATHS = None

def _worker_init(use_bbn_effect, baseline_rms=None, musan_paths=None, fsd_paths=None):
    global _WORKER_PRESETS, _WORKER_USE_BBN
    global _WORKER_BASELINE_RMS, _WORKER_MUSAN_PATHS, _WORKER_FSD_PATHS
    _WORKER_USE_BBN = use_bbn_effect
    _WORKER_BASELINE_RMS = baseline_rms
    _WORKER_MUSAN_PATHS = musan_paths or []
    _WORKER_FSD_PATHS = fsd_paths or []
    rng = random.Random()
    if _bbn and _bbn.HAS_PEDALBOARD:
        _WORKER_PRESETS = _bbn.build_audio_effect_presets_pedalboard(
            _bbn.NUM_AUDIO_EFFECTS, _bbn.EFFECT_PRESET_SEED,
            wet_lo=_bbn.PEDALBOARD_EFFECT_WET_LO, wet_hi=_bbn.PEDALBOARD_EFFECT_WET_HI
        )
    elif _bbn:
        _WORKER_PRESETS = _bbn.build_audio_effect_presets_fallback(_bbn.NUM_AUDIO_EFFECTS, _bbn.EFFECT_PRESET_SEED)
    else:
        _WORKER_PRESETS = build_effect_presets(rng)

# ================= 路径 =================
BASE = os.environ.get("VOICEAGENT_ROOT", ".")
NOISEBASE = os.environ.get("NOISE_ROOT", BASE)
RUNS_ALL_DIR = os.path.join(BASE, "api_generate", "postprocess_qwen3tts_runs_notag", "runs_all")
MUSAN_NOISE_DIRS = [
    os.path.join(NOISEBASE, "MUSAN", "raw", "musan", "noise", "free-sound"),
    os.path.join(NOISEBASE, "MUSAN", "raw", "musan", "noise", "sound-bible"),
]
FSD50K_AUDIO = os.path.join(NOISEBASE, "FSD50K", "raw", "FSD50K", "FSD50K.dev_audio")

# parquet 默认路径（strip 之后、codec 之后均可）
INPUT_DIR  = os.path.join(BASE, "api_generate", "pipeline_outputs_parquet", "training_v7_codec_noself")
OUTPUT_DIR = os.path.join(BASE, "api_generate", "pipeline_outputs_parquet", "training_v7_codec_noself_noised")

SAMPLE_RATE = 24000   # 全程 24kHz，noise 文件重采样到 24k，不改时长音调
NUM_WORKERS = 256

# ================= 随机范围配置（全部可调，无 seed，每条都不同） =================
# 响度基准：只统计 |sample| > 该阈值的部分（有声音部分），避免静音拉低均值
NON_SILENT_THRESHOLD = 1e-5
# 噪声相对「user 有声音部分」RMS 的比例，每条在 [LO, HI] 内随机
NOISE_MIX_RATIO_LO, NOISE_MIX_RATIO_HI = 0.05, 0.15
# user 轨响度随时间平滑波动比例 [LO, HI]，例如 0.7~1.1 表示 70%~110%
USER_GAIN_LO, USER_GAIN_HI = 0.5, 1.2
# 噪声轨响度随时间平滑波动比例 [LO, HI]
NOISE_GAIN_LO, NOISE_GAIN_HI = 0.5, 1.2
# 平滑增益关键帧间隔（秒）[min, max]，关键帧间线性插值
GAIN_KEYFRAME_INTERVAL = (1.2, 4.0)
# 噪声片段拼接时的交叉淡入淡出时长（秒）[min, max]，每条随机
CROSSFADE_SEC = (0.05, 5)
# 音频效果：pedalboard 时效果量 wet/mix 在 [LO, HI] 随机；效果池大小
PEDALBOARD_EFFECT_WET_LO, PEDALBOARD_EFFECT_WET_HI = 0.0, 0.9
NUM_AUDIO_EFFECTS = 520
# user 轨随机升降调半音 [LO, HI]，在效果前做，正=更尖锐负=更低沉，不改变语速
USER_PITCH_SEMITONES_LO, USER_PITCH_SEMITONES_HI = -5.2, 5.2


def load_wav(path, max_frames=None, start_frame=0):
    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            nch = w.getnchannels()
            total = w.getnframes()
            if start_frame > 0:
                start_frame = min(start_frame, total)
                w.setpos(start_frame)
                nframes = total - start_frame
            else:
                nframes = total
            if max_frames is not None and nframes > max_frames:
                nframes = max_frames
            raw = w.readframes(nframes)
    except Exception:
        return None, None
    if not raw:
        return None, None
    n = len(raw) // (2 * nch)
    fmt = "<" + "h" * (n * nch)
    try:
        y = np.array(struct.unpack(fmt, raw[: n * nch * 2]), dtype=np.float32)
    except Exception:
        return None, None
    if nch == 2:
        y = (y[::2] + y[1::2]) / 2.0
    return y / 32768.0, sr


def resample_to_16k(y, orig_sr):
    """重采样到目标采样率（SAMPLE_RATE），时长/音调不变。"""
    if orig_sr == SAMPLE_RATE:
        return y
    n_old = len(y)
    n_new = int(round(n_old * SAMPLE_RATE / orig_sr))
    return np.interp(np.linspace(0, n_old - 1, n_new), np.arange(n_old), y).astype(np.float32)


def load_audio_bytes(audio_bytes) -> tuple:
    """从 parquet large_binary 加载音频为 float32 array。
    - RIFF header → WAV 格式（silence），返回 (y, sr)
    - 无 RIFF header → 裸 PCM int16 @ SAMPLE_RATE（chunk audio），返回 (y, SAMPLE_RATE)
    """
    raw = bytes(audio_bytes)
    if raw[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(raw), "rb") as w:
                sr  = w.getframerate()
                nch = w.getnchannels()
                data = w.readframes(w.getnframes())
            y = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            if nch == 2:
                y = (y[::2] + y[1::2]) / 2.0
            return y / 32768.0, sr
        except Exception:
            return None, None
    # 裸 PCM int16 @ SAMPLE_RATE
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return y, SAMPLE_RATE


def audio_to_pcm_bytes(y: np.ndarray) -> bytes:
    """float32 array → raw PCM int16 bytes（与输入 chunk 格式一致，无 WAV header）。"""
    return (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def rms(y):
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(y ** 2)))


def rms_non_silent(y, threshold=1e-5):
    """只计算 |y| > threshold 的样本的 RMS，作为有声音部分的响度。"""
    mask = np.abs(y) > threshold
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean(y[mask] ** 2)))


def collect_wav_paths(dirs_or_one):
    if isinstance(dirs_or_one, (list, tuple)):
        paths = []
        for d in dirs_or_one:
            if os.path.isdir(d):
                for root, _, names in os.walk(d):
                    for name in names:
                        if name.lower().endswith(".wav"):
                            paths.append(os.path.join(root, name))
        return paths
    if os.path.isdir(dirs_or_one):
        return [os.path.join(r, n) for r, _, names in os.walk(dirs_or_one) for n in names if n.lower().endswith(".wav")]
    return []


def compute_baseline_rms_and_durations(train_entries, sample_rate, threshold, max_segments=2000):
    """
    从所有条目的 user 轨（偶数下标）里取路径，只对有声音部分算 RMS，求平均作为基准；
    同时可返回每条 user 总时长（本脚本主要用每条自己的时长建噪声）。
    """
    all_user_segment_rms = []
    for ent in train_entries:
        audios = ent.get("audios") or []
        for i in range(0, len(audios), 2):
            p = audios[i]
            if "silence" in os.path.normpath(p).lower():
                continue
            y, sr = load_wav(p)
            if y is None or len(y) < 50:
                continue
            if sr != sample_rate:
                y = resample_to_16k(y, sr)
            r = rms_non_silent(y, threshold)
            if r > 1e-6:
                all_user_segment_rms.append(r)
            if len(all_user_segment_rms) >= max_segments:
                break
        if len(all_user_segment_rms) >= max_segments:
            break
    if not all_user_segment_rms:
        return 0.02
    return float(np.mean(all_user_segment_rms))


def validate_entry_alignment(entry, line_index):
    messages = entry.get("messages") or []
    audios = entry.get("audios") or []
    if not messages:
        raise ValueError("第 %d 条缺少 messages，无法对齐音频与对话轮次" % line_index)
    if messages[0].get("role") != "system":
        raise ValueError("第 %d 条 messages[0] 不是 system，当前脚本假设首条是 system" % line_index)
    if not audios:
        if len(messages) != 1:
            raise ValueError("第 %d 条无 audios，但 messages 数量不是 1" % line_index)
        return
    if len(audios) % 2 != 0:
        raise ValueError("第 %d 条 audios 数量 %d 不是偶数，无法按 user/self 成对处理" % (line_index, len(audios)))
    if len(messages) != len(audios) + 1:
        raise ValueError("第 %d 条 messages/audios 数量不匹配: %d vs %d" % (line_index, len(messages), len(audios)))
    for msg_idx in range(1, len(messages)):
        expected_role = "user" if msg_idx % 2 == 1 else "assistant"
        role = messages[msg_idx].get("role")
        if role != expected_role:
            raise ValueError("第 %d 条 messages[%d].role=%r，期望 %r" % (line_index, msg_idx, role, expected_role))


def build_output_entry(entry, new_audios):
    new_entry = dict(entry)
    new_entry["audios"] = new_audios
    return new_entry


def append_entry_to_jsonl(output_jsonl_path, jsonl_lock, entry):
    with jsonl_lock:
        with open(output_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def crossfade_append(out, new_chunk, crossfade_n):
    if crossfade_n <= 0 or len(out) < crossfade_n or len(new_chunk) < crossfade_n:
        return np.concatenate([out, new_chunk])
    n = crossfade_n
    out[-n:] = out[-n:] * np.linspace(1, 0, n, dtype=np.float32) + new_chunk[:n] * np.linspace(0, 1, n, dtype=np.float32)
    return np.concatenate([out, new_chunk[n:]])


def build_noise_track_no_loop(musan_paths, fsd_paths, target_samples, sample_rate, rng):
    """
    不循环：随机选 MUSAN 或 FSD50K，全随机拿取片段并随机顺序拼接，直到长度 >= target_samples。
    每条调用都重新随机，保证每条数据加的噪声不同。
    """
    use_musan = (len(musan_paths) > 0 and (len(fsd_paths) == 0 or rng.random() < 0.5))
    source_paths = musan_paths if use_musan else fsd_paths
    if not source_paths:
        return np.zeros(target_samples, dtype=np.float32)
    crossfade_sec = rng.uniform(CROSSFADE_SEC[0], CROSSFADE_SEC[1])
    crossfade_n = int(round(crossfade_sec * sample_rate))
    indices = list(range(len(source_paths)))
    rng.shuffle(indices)
    out = np.zeros(0, dtype=np.float32)
    idx = 0
    while len(out) < target_samples:
        path = source_paths[indices[idx % len(indices)]]
        idx += 1
        y, sr = load_wav(path)
        if y is None or len(y) < 100:
            continue
        if sr != sample_rate:
            y = resample_to_16k(y, sr)
        need = target_samples - len(out)
        if len(y) >= need + crossfade_n:
            start = rng.randint(0, max(0, len(y) - need - crossfade_n))
            chunk = y[start : start + need + crossfade_n]
        else:
            chunk = y
        if len(out) == 0:
            out = chunk.copy()
        else:
            out = crossfade_append(out, chunk, crossfade_n)
    return out[:target_samples].astype(np.float32)


def normalize_to_target_rms(y, target_rms):
    cur = rms(y)
    if cur < 1e-9:
        return y
    return np.clip(y * (target_rms / cur), -1.0, 1.0).astype(np.float32)


def pitch_shift_no_speed_change(y, semitones):
    if abs(semitones) < 0.01:
        return y
    n = len(y)
    ratio = 2.0 ** (semitones / 12.0)
    n_mid = max(1, int(round(n / ratio)))
    y_mid = np.interp(np.linspace(0, n - 1, n_mid), np.arange(n), y).astype(np.float32)
    return np.interp(np.linspace(0, n_mid - 1, n), np.arange(n_mid), y_mid).astype(np.float32)


def smooth_random_gain_curve(n_samples, sample_rate, gain_lo, gain_hi, rng):
    t_end = n_samples / float(sample_rate)
    times = [0.0]
    gains = [rng.uniform(gain_lo, gain_hi)]
    while times[-1] < t_end:
        step = rng.uniform(GAIN_KEYFRAME_INTERVAL[0], GAIN_KEYFRAME_INTERVAL[1])
        times.append(times[-1] + step)
        gains.append(rng.uniform(gain_lo, gain_hi))
    t_all = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
    return np.interp(t_all, np.array(times), np.array(gains)).astype(np.float32)


def build_effect_presets(rng):
    if HAS_PEDALBOARD:
        presets = []
        types = ["reverb", "chorus", "distortion", "phaser", "compressor", "lowpass", "highpass",
                 "reverb+chorus", "distortion+lowpass", "phaser+reverb", "chorus+highpass"]
        for i in range(NUM_AUDIO_EFFECTS):
            etype = types[i % len(types)]
            wet = rng.uniform(PEDALBOARD_EFFECT_WET_LO, PEDALBOARD_EFFECT_WET_HI)
            board = Pedalboard()
            if "reverb" in etype:
                board.append(Reverb(room_size=rng.uniform(0.3, 0.95), damping=rng.uniform(0.3, 0.9), wet_level=wet, dry_level=rng.uniform(0.4, 0.9)))
            if "chorus" in etype:
                board.append(Chorus(rate_hz=rng.uniform(0.5, 3.0), depth=rng.uniform(0.2, 0.7), centre_delay_ms=rng.uniform(5, 15), feedback=rng.uniform(0.1, 0.4), mix=wet))
            if "distortion" in etype:
                board.append(Distortion(drive_db=rng.uniform(5, 25)))
            if "phaser" in etype:
                board.append(Phaser(rate_hz=rng.uniform(0.3, 2.0), depth=rng.uniform(0.3, 0.8), centre_frequency_hz=rng.uniform(500, 2000), feedback=rng.uniform(0.2, 0.6), mix=wet))
            if "compressor" in etype:
                board.append(Compressor(threshold_db=rng.uniform(-30, -10), ratio=rng.uniform(2, 8)))
            if "lowpass" in etype:
                board.append(LowpassFilter(cutoff_frequency_hz=rng.uniform(1500, 5000)))
            if "highpass" in etype:
                board.append(HighpassFilter(cutoff_frequency_hz=rng.uniform(100, 400)))
            presets.append((etype, board))
        return presets
    # numpy fallback: 简单回声预设
    presets = []
    r = random.Random(rng.random())
    for _ in range(NUM_AUDIO_EFFECTS):
        n_taps = r.randint(4, 9)
        delays_ms = sorted([r.uniform(25, 500) for _ in range(n_taps)])
        gains = [r.uniform(0.08, 0.55) for _ in range(n_taps)]
        presets.append(("echo", {"reverb": (delays_ms, gains)}))
    return presets


def apply_effect(y, sample_rate, preset, y_orig):
    etype, data = preset
    if isinstance(data, dict):
        return apply_numpy_effect(y, sample_rate, data, y_orig)
    try:
        return data(y, sample_rate).astype(np.float32)
    except Exception:
        return y


def apply_numpy_effect(y, sample_rate, params, y_orig):
    out = y.copy()
    if "reverb" in params:
        delays_ms, gains = params["reverb"]
        n = len(out)
        for d_ms, g in zip(delays_ms, gains):
            d_samp = min(int(sample_rate * d_ms / 1000.0), n - 1)
            if d_samp > 0:
                out[d_samp:] += y_orig[: n - d_samp] * g
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def write_wav(path, y, sample_rate=16000):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((np.clip(y, -1.0, 1.0) * 32767).astype(np.int16).tobytes())


def run_debug_one():
    """
    DEBUG 模式：随机取一条训练数据，用与 build_background_noise.py 完全一致的核心逻辑处理，
    仅写出 noise_preview/demo_user_reverb_noise.wav，便于对比试听加噪效果。
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build_background_noise as bbn

    # 让 build_background_noise 使用本脚本的 TRAIN_JSONL 与输出目录
    bbn.TRAIN_JSONL = TRAIN_JSONL
    bbn.OUTPUT_DIR = NOISE_PREVIEW_DIR

    rng = random.Random()
    entries = []
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    if not entries:
        print("DEBUG: 无训练数据，退出")
        return
    entry = rng.choice(entries)
    print("DEBUG: 随机选中 1 条，使用 build_background_noise 相同逻辑，仅输出 demo_user_reverb_noise.wav")

    target_rms = bbn.LOUDNESS_RATIO * bbn.compute_train_avg_rms(
        num_segments=bbn.TRAIN_SEGMENTS_FOR_LOUDNESS,
        segment_duration=bbn.SEGMENT_DURATION_FOR_LOUDNESS,
    )
    musan_paths = collect_wav_paths(MUSAN_NOISE_DIRS)
    fsd_paths = collect_wav_paths(FSD50K_AUDIO)
    y_musan = None
    y_fsd = None
    if musan_paths:
        y_musan = bbn.build_noise_track(
            musan_paths, bbn.TARGET_DURATION_SEC, SAMPLE_RATE, bbn.CROSSFADE_SEC, rng
        )
        if y_musan is not None:
            y_musan = bbn.normalize_to_target_rms(y_musan, target_rms)
    if fsd_paths:
        y_fsd = bbn.build_noise_track(
            fsd_paths, bbn.TARGET_DURATION_SEC, SAMPLE_RATE, bbn.CROSSFADE_SEC, rng
        )
        if y_fsd is not None:
            y_fsd = bbn.normalize_to_target_rms(y_fsd, target_rms)

    if bbn.HAS_PEDALBOARD:
        effect_presets = bbn.build_audio_effect_presets_pedalboard(
            bbn.NUM_AUDIO_EFFECTS, bbn.EFFECT_PRESET_SEED,
            wet_lo=bbn.PEDALBOARD_EFFECT_WET_LO, wet_hi=bbn.PEDALBOARD_EFFECT_WET_HI
        )
    else:
        effect_presets = bbn.build_audio_effect_presets_fallback(bbn.NUM_AUDIO_EFFECTS, bbn.EFFECT_PRESET_SEED)

    user_raw = bbn.build_user_track(entry, SAMPLE_RATE)
    if user_raw is None or len(user_raw) < SAMPLE_RATE:
        print("DEBUG: 该条 user 轨过短或为空，退出")
        return
    n_user = len(user_raw)
    preset = rng.choice(effect_presets)
    user_reverb = bbn.apply_audio_effect(user_raw, SAMPLE_RATE, preset)
    user_gain = bbn.smooth_random_gain_curve(
        n_user, SAMPLE_RATE, bbn.USER_GAIN_LO, bbn.USER_GAIN_HI,
        bbn.GAIN_KEYFRAME_INTERVAL, rng
    )
    user_reverb = (user_reverb * user_gain).astype(np.float32)

    noise_base = y_musan if (y_musan is not None) else y_fsd
    if noise_base is not None:
        noise_same_len = bbn.mix_noise_to_length(noise_base, n_user, rng)
        user_rms_val = rms(user_reverb)
        if user_rms_val > 1e-9:
            noise_rms_target = user_rms_val * bbn.NOISE_MIX_RATIO
            noise_same_len = bbn.normalize_to_target_rms(noise_same_len, noise_rms_target)
        noise_gain = bbn.smooth_random_gain_curve(
            n_user, SAMPLE_RATE, bbn.NOISE_GAIN_LO, bbn.NOISE_GAIN_HI,
            bbn.GAIN_KEYFRAME_INTERVAL, rng
        )
        noise_same_len = (noise_same_len * noise_gain).astype(np.float32)
        mixed = np.clip(user_reverb + noise_same_len, -1.0, 1.0).astype(np.float32)
    else:
        mixed = user_reverb

    out_path = os.path.join(NOISE_PREVIEW_DIR, "demo_user_reverb_noise.wav")
    write_wav(out_path, mixed, SAMPLE_RATE)
    print("DEBUG: 已写出 %s" % out_path)


def process_one_entry(line_index, entry, baseline_rms, musan_paths, fsd_paths,
                      output_voice_dir, output_jsonl_path, jsonl_lock):
    """
    处理一条：整条 user 拼成一段 → 一次升降调/效果/增益 → 与等长噪声混合 → 按原边界切分写 wav；
    self 轨（奇数下标）不加噪、不复制，JSONL 中保留原始路径。写完后立即追加 JSONL。
    保证 len(new_audios)==len(audios) 且顺序一致，不丢任何数据。
    """
    audios = entry.get("audios") or []
    if not audios:
        append_entry_to_jsonl(output_jsonl_path, jsonl_lock, dict(entry))
        return
    out_dir = os.path.join(output_voice_dir, "id_%07d" % line_index)
    rng = random.Random()
    n_audios = len(audios)
    # 整条 user 拼成一段，记录每段长度（与 audios 偶数下标一一对应）
    user_chunks = []
    segment_lengths = []
    for i in range(0, n_audios, 2):
        y, sr = load_wav(audios[i])
        if y is None or len(y) < 10:
            segment_lengths.append(0)
            continue
        if sr != SAMPLE_RATE:
            y = resample_to_16k(y, sr)
        user_chunks.append(y)
        segment_lengths.append(len(y))
    if not user_chunks:
        # 本条无有效 user 段：不写任何文件，全部保留原路径，不丢数据
        new_audios = list(audios)
        append_entry_to_jsonl(output_jsonl_path, jsonl_lock, build_output_entry(entry, new_audios))
        return
    os.makedirs(out_dir, exist_ok=True)
    user_track = np.concatenate(user_chunks).astype(np.float32)
    user_total_samples = len(user_track)
    # 整条：一次升降调
    pitch_sem = rng.uniform(USER_PITCH_SEMITONES_LO, USER_PITCH_SEMITONES_HI)
    user_track = pitch_shift_no_speed_change(user_track, pitch_sem)
    # 整条：一次效果（用 worker 内效果池）
    presets = _WORKER_PRESETS if _WORKER_PRESETS is not None else build_effect_presets(rng)
    preset = rng.choice(presets)
    if _WORKER_USE_BBN and _bbn is not None:
        user_eff = _bbn.apply_audio_effect(user_track, SAMPLE_RATE, preset)
    else:
        user_eff = apply_effect(user_track, SAMPLE_RATE, preset, user_track)
    # 整条：一次 user 增益曲线
    user_gain = smooth_random_gain_curve(len(user_eff), SAMPLE_RATE, USER_GAIN_LO, USER_GAIN_HI, rng)
    user_eff = (user_eff * user_gain).astype(np.float32)
    # 等长噪声：一次生成、归一化、增益
    noise_track = build_noise_track_no_loop(musan_paths, fsd_paths, user_total_samples, SAMPLE_RATE, rng)
    noise_mix_ratio = rng.uniform(NOISE_MIX_RATIO_LO, NOISE_MIX_RATIO_HI)
    noise_track = normalize_to_target_rms(noise_track, baseline_rms * noise_mix_ratio)
    noise_gain = smooth_random_gain_curve(len(noise_track), SAMPLE_RATE, NOISE_GAIN_LO, NOISE_GAIN_HI, rng)
    noise_track = (noise_track * noise_gain).astype(np.float32)
    # 整条混合后按原边界切分
    mixed = np.clip(user_eff + noise_track[: len(user_eff)], -1.0, 1.0).astype(np.float32)
    boundaries = np.cumsum([0] + segment_lengths)
    new_audios = [None] * n_audios  # 保证长度与顺序一致，不丢数据
    user_seg_idx = 0
    for i in range(n_audios):
        if i % 2 == 1:
            # self 轨：不加噪、不复制，直接保留原路径
            new_audios[i] = audios[i]
            continue
        out_path = os.path.join(out_dir, "audio_%d.wav" % i)
        start, end = int(boundaries[user_seg_idx]), int(boundaries[user_seg_idx + 1])
        user_seg_idx += 1
        if start < end and end <= len(mixed):
            write_wav(out_path, mixed[start:end], SAMPLE_RATE)
        else:
            # 该 user 段被跳过或越界：写原段到 out_path，保证有一条可用的 user 轨
            y, sr = load_wav(audios[i])
            if y is not None and len(y) > 0:
                if sr != SAMPLE_RATE:
                    y = resample_to_16k(y, sr)
                write_wav(out_path, y, SAMPLE_RATE)
            else:
                if os.path.isfile(audios[i]):
                    shutil.copy2(audios[i], out_path)
        new_audios[i] = out_path
    append_entry_to_jsonl(output_jsonl_path, jsonl_lock, build_output_entry(entry, new_audios))


def compute_baseline_rms_parquet(parquet_files, threshold=1e-5, max_segments=2000):
    """从 parquet 的 audios bytes 列计算 user 有声音部分 RMS 基准。
    跳过 RIFF 格式（silence），只统计裸 PCM chunk audio。"""
    total_rms = cnt = 0.0
    for pf in parquet_files:
        if cnt >= max_segments:
            break
        pfile = pq.ParquetFile(str(pf))
        for batch in pfile.iter_batches(batch_size=128, columns=["audios"]):
            for row_audios in batch.column("audios").to_pylist():
                if cnt >= max_segments:
                    break
                for ab in (row_audios or []):
                    raw = bytes(ab)
                    if raw[:4] == b"RIFF":   # silence, skip
                        continue
                    y, _ = load_audio_bytes(raw)
                    if y is None or len(y) < 10:
                        continue
                    r = rms_non_silent(y, threshold)
                    if r < threshold:
                        continue
                    total_rms += r
                    cnt += 1
                    if cnt >= max_segments:
                        break
            if cnt >= max_segments:
                break
    return total_rms / cnt if cnt > 0 else 1e-3


def process_one_entry_parquet(row_dict, baseline_rms, musan_paths, fsd_paths):
    """parquet 版加噪：audios 全部为 user audio bytes（strip 之后）。
    - RIFF header → WAV/silence chunk，也加入完整时间轴加噪
    - 无 RIFF header → 裸 PCM chunk audio，加噪
    """
    row = dict(row_dict)
    audios = list(row["audios"]) if row.get("audios") is not None else []
    if not audios:
        return row

    rng = random.Random()

    # 加载所有 chunk audio；RIFF silence 也要铺背景噪。
    user_chunks, segment_lengths = [], []
    user_indices = []
    for i in range(len(audios)):
        y, sr = load_audio_bytes(bytes(audios[i]))
        if y is None or len(y) < 10:
            continue
        if sr != SAMPLE_RATE:
            y = resample_to_16k(y, sr)   # noise 路径也用这个，名字不改
        user_indices.append(i)
        user_chunks.append(y)
        segment_lengths.append(len(y))

    if not user_chunks:
        return row

    # 拼成一段 → 一次 pitch / effect / gain
    user_track = np.concatenate(user_chunks).astype(np.float32)
    user_total  = len(user_track)

    pitch_sem  = rng.uniform(USER_PITCH_SEMITONES_LO, USER_PITCH_SEMITONES_HI)
    user_track = pitch_shift_no_speed_change(user_track, pitch_sem)

    presets = _WORKER_PRESETS if _WORKER_PRESETS is not None else build_effect_presets(rng)
    preset  = rng.choice(presets)
    if _WORKER_USE_BBN and _bbn is not None:
        user_eff = _bbn.apply_audio_effect(user_track, SAMPLE_RATE, preset)
    else:
        user_eff = apply_effect(user_track, SAMPLE_RATE, preset, user_track)

    user_gain = smooth_random_gain_curve(len(user_eff), SAMPLE_RATE, USER_GAIN_LO, USER_GAIN_HI, rng)
    user_eff  = (user_eff * user_gain).astype(np.float32)

    # 噪声轨
    noise_track     = build_noise_track_no_loop(musan_paths, fsd_paths, user_total, SAMPLE_RATE, rng)
    noise_mix_ratio = rng.uniform(NOISE_MIX_RATIO_LO, NOISE_MIX_RATIO_HI)
    noise_track     = normalize_to_target_rms(noise_track, baseline_rms * noise_mix_ratio)
    noise_gain      = smooth_random_gain_curve(len(noise_track), SAMPLE_RATE, NOISE_GAIN_LO, NOISE_GAIN_HI, rng)
    noise_track     = (noise_track * noise_gain).astype(np.float32)

    mixed = np.clip(user_eff + noise_track[:len(user_eff)], -1.0, 1.0).astype(np.float32)

    # 按原边界切回，写回 bytes
    boundaries  = np.cumsum([0] + segment_lengths)
    new_audios  = list(audios)   # 无法解码的 chunk 原样保留
    valid_chunk = 0
    for j, idx in enumerate(user_indices):
        if segment_lengths[j] == 0:
            continue
        start = int(boundaries[valid_chunk])
        end   = int(boundaries[valid_chunk + 1])
        valid_chunk += 1
        if start < end <= len(mixed):
            new_audios[idx] = audio_to_pcm_bytes(mixed[start:end])
        # 失败则保留原 bytes 不变

    row["audios"] = new_audios
    return row


def _process_row_worker(args):
    row_dict, baseline_rms, musan_paths, fsd_paths = args
    return process_one_entry_parquet(row_dict, baseline_rms, musan_paths, fsd_paths)


def _process_row_worker_global(row_dict):
    if _WORKER_BASELINE_RMS is None:
        raise RuntimeError("row worker was not initialized with baseline RMS")
    return process_one_entry_parquet(
        row_dict,
        _WORKER_BASELINE_RMS,
        _WORKER_MUSAN_PATHS,
        _WORKER_FSD_PATHS,
    )


def _worker_ready():
    return True


def _schema_with_images(schema):
    if "images" in schema.names:
        return schema, False
    empty_images_type = pa.list_(pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path",  pa.string()),
    ]))
    return schema.append(pa.field("images", empty_images_type)), True


def process_parquet_file_streaming(
    input_path,
    output_path,
    row_executor,
    batch_rows,
    compression,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    tmp_path = Path(str(output_path) + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    start = time.time()
    pfile = pq.ParquetFile(str(input_path))
    schema, add_empty_images = _schema_with_images(pfile.schema_arrow)
    writer = None
    rows_read = 0
    rows_written = 0
    batches_written = 0
    print(
        "[start] %s rows=%d row_groups=%d" % (
            input_path.name,
            pfile.metadata.num_rows,
            pfile.metadata.num_row_groups,
        ),
        flush=True,
    )
    try:
        writer = pq.ParquetWriter(str(tmp_path), schema=schema, compression=compression)
        for batch in pfile.iter_batches(batch_size=max(1, int(batch_rows))):
            rows = batch.to_pylist()
            if add_empty_images:
                for row in rows:
                    row["images"] = []
            rows_read += len(rows)
            futures = [row_executor.submit(_process_row_worker_global, row) for row in rows]
            new_rows = [future.result() for future in futures]
            table = pa.Table.from_pylist(new_rows, schema=schema)
            writer.write_table(table, row_group_size=max(1, table.num_rows))
            rows_written += table.num_rows
            batches_written += 1
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    else:
        if writer is not None:
            writer.close()
        os.replace(str(tmp_path), str(output_path))

    return {
        "input": input_path.name,
        "output": output_path.name,
        "rows_read": rows_read,
        "rows_written": rows_written,
        "batches_written": batches_written,
        "seconds": time.time() - start,
    }


def parse_path_list(value):
    if not value:
        return []
    return [item for item in re.split(r"[,;]", value) if item]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="parquet 版加噪：读 input parquet，对 user audio bytes 加噪，原地写回 output parquet")
    ap.add_argument("--input-dir",  default=INPUT_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--input-glob", default="*.parquet")
    ap.add_argument("--musan-noise-dirs", default=",".join(MUSAN_NOISE_DIRS), help="逗号分隔的 MUSAN noise wav 目录；可为空")
    ap.add_argument("--fsd50k-audio-dir", default=FSD50K_AUDIO, help="FSD50K.dev_audio wav 目录；可为空")
    ap.add_argument("--workers",    type=int, default=NUM_WORKERS, help="全局 row 加噪 worker 总数")
    ap.add_argument("--file-workers", type=int, default=8, help="并发处理的 parquet 文件数")
    ap.add_argument("--batch-rows", type=int, default=128, help="每个 row group/batch 的 session 数")
    ap.add_argument("--compression", default="snappy")
    ap.add_argument("--max-files", type=int, default=0, help="仅处理前 N 个输入 parquet，0 表示全量")
    ap.add_argument("--resume",     action="store_true", help="跳过已有输出文件")
    ap.add_argument("--max-baseline-segments", type=int, default=2000)
    args = ap.parse_args()

    print("=" * 60)
    print(
        "激情在燃烧！parquet 批量加噪 @%dHz，file_workers=%d row_workers=%d batch_rows=%d"
        % (SAMPLE_RATE, args.file_workers, args.workers, args.batch_rows)
    )
    print("=" * 60)

    parquets = sorted(Path(args.input_dir).glob(args.input_glob))
    if not parquets:
        raise FileNotFoundError("找不到 %s: %s" % (args.input_glob, args.input_dir))
    if args.max_files > 0:
        parquets = parquets[:args.max_files]
    os.makedirs(args.output_dir, exist_ok=True)

    musan_dirs = parse_path_list(args.musan_noise_dirs)
    fsd_dir = args.fsd50k_audio_dir.strip()
    musan_paths = collect_wav_paths(musan_dirs)
    fsd_paths   = collect_wav_paths(fsd_dir) if fsd_dir else []
    print("MUSAN %d 个，FSD50K %d 个" % (len(musan_paths), len(fsd_paths)))
    if not musan_paths and not fsd_paths:
        raise FileNotFoundError(
            "找不到 MUSAN/FSD50K wav。请设置 --musan-noise-dirs、--fsd50k-audio-dir 或 NOISE_ROOT。"
        )

    print("计算 user 有声音部分 RMS 基准（最多 %d 段）..." % args.max_baseline_segments)
    baseline_rms = compute_baseline_rms_parquet(parquets, NON_SILENT_THRESHOLD, args.max_baseline_segments)
    print("baseline RMS: %.4f" % baseline_rms)

    use_bbn = (_bbn is not None)
    jobs = []
    skipped = 0
    for pfile in parquets:
        out = Path(args.output_dir) / pfile.name
        if args.resume and out.exists():
            skipped += 1
            continue
        jobs.append((pfile, out))

    print(
        "input=%s output=%s files=%d jobs=%d skipped=%d file_workers=%d row_workers=%d "
        "batch_rows=%d compression=%s"
        % (
            args.input_dir,
            args.output_dir,
            len(parquets),
            len(jobs),
            skipped,
            args.file_workers,
            args.workers,
            args.batch_rows,
            args.compression,
        ),
        flush=True,
    )
    if not jobs:
        print("No pending files.")
        return

    total_done = 0
    completed = 0
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_worker_init,
        initargs=(use_bbn, baseline_rms, musan_paths, fsd_paths),
    ) as row_executor:
        # 先在主线程启动 row workers，避免文件线程首次 submit 时再 fork。
        warmups = [
            row_executor.submit(_worker_ready)
            for _ in range(max(1, int(args.workers)))
        ]
        for future in warmups:
            future.result()

        if args.file_workers <= 1:
            iterator = tqdm(jobs, total=len(jobs), desc="parquet files")
            for pfile, out in iterator:
                result = process_parquet_file_streaming(
                    pfile,
                    out,
                    row_executor,
                    args.batch_rows,
                    args.compression,
                )
                completed += 1
                total_done += int(result["rows_written"])
                print(
                    "[done] %s rows=%d batches=%d seconds=%.1f"
                    % (
                        result["output"],
                        result["rows_written"],
                        result["batches_written"],
                        result["seconds"],
                    ),
                    flush=True,
                )
                if int(result["rows_read"]) != int(result["rows_written"]):
                    raise RuntimeError(
                        "row count mismatch for %s: %d != %d"
                        % (pfile.name, result["rows_read"], result["rows_written"])
                    )
        else:
            with ThreadPoolExecutor(max_workers=max(1, int(args.file_workers))) as file_executor:
                future_map = {
                    file_executor.submit(
                        process_parquet_file_streaming,
                        pfile,
                        out,
                        row_executor,
                        args.batch_rows,
                        args.compression,
                    ): (pfile, out)
                    for pfile, out in jobs
                }
                for future in tqdm(as_completed(future_map), total=len(future_map), desc="parquet files"):
                    pfile, _ = future_map[future]
                    result = future.result()
                    completed += 1
                    total_done += int(result["rows_written"])
                    print(
                        "[done] %s rows=%d batches=%d seconds=%.1f"
                        % (
                            result["output"],
                            result["rows_written"],
                            result["batches_written"],
                            result["seconds"],
                        ),
                        flush=True,
                    )
                    if int(result["rows_read"]) != int(result["rows_written"]):
                        raise RuntimeError(
                            "row count mismatch for %s: %d != %d"
                            % (pfile.name, result["rows_read"], result["rows_written"])
                        )

    print("完成！done=%d skipped=%d rows=%d 输出: %s" % (completed, skipped, total_done, args.output_dir))
    print("=" * 60)


if __name__ == "__main__":
    main()
