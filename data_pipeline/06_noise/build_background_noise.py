#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激情在燃烧！
1) 随机合成 1 分钟背景噪声：FSD50K、MUSAN 各一条。
2) 随机一条训练数据：user 轨拼接 + 随机升降调（模拟不同声调）+ 100+ 种音频效果随机选一种
   + user 响度 70%-110% 随时间平滑波动；背景噪声 50%-120% 随时间平滑波动（噪声不加效果）。

依赖: numpy, pedalboard (Spotify 音频效果库)
安装: pip install pedalboard
运行: conda activate megaswift && python build_background_noise.py
"""

import os
# 必须在 import pedalboard 之前设置，避免部分 CPU 上 AVX 指令集导致的 Illegal instruction
os.environ["PEDALBOARD_NO_AVX"] = "1"

import json
import random
import argparse
import re
import wave
import struct
import numpy as np

try:
    from pedalboard import (
        Pedalboard, Reverb, Chorus, Distortion, Phaser, Compressor,
        LowpassFilter, HighpassFilter, Gain, Limiter, PitchShift
    )
    HAS_PEDALBOARD = True
except Exception as e:
    HAS_PEDALBOARD = False
    Pedalboard = Reverb = Chorus = Distortion = Phaser = Compressor = None
    LowpassFilter = HighpassFilter = Gain = Limiter = PitchShift = None
    print("提示：pedalboard 不可用（%s），将使用纯 numpy 实现的 100+ 音频效果" % str(e))

# ================= 路径配置 =================
BASE = os.environ.get("VOICEAGENT_ROOT", ".")
NOISEBASE = os.environ.get("NOISE_ROOT", BASE)
# 训练集 JSONL（每条一行，含 messages / audios）
TRAIN_JSONL = os.path.join(BASE, "voiceagent", "gsm8k_train", "train_data_v4.jsonl")
# MUSAN 噪声 wav 所在目录（free-sound + sound-bible）
MUSAN_NOISE_DIRS = [
    os.path.join(NOISEBASE, "MUSAN", "raw", "musan", "noise", "free-sound"),
    os.path.join(NOISEBASE, "MUSAN", "raw", "musan", "noise", "sound-bible"),
]
# FSD50K 音频目录
FSD50K_AUDIO = os.path.join(NOISEBASE, "FSD50K", "raw", "FSD50K", "FSD50K.dev_audio")
# 输出 wav 与试听文件保存目录
OUTPUT_DIR = os.path.join(BASE, "voiceagent", "gsm8k_train", "noise_preview")

# ================= 噪声合成配置 =================
# 合成背景噪声的目标时长（秒）
TARGET_DURATION_SEC = 60.0
# 多段噪声拼接时的交叉淡入淡出时长（秒），越大衔接越平滑
CROSSFADE_SEC = 1
# 从训练集中取多少段非静音 0.5s 来统计平均响度，用于确定噪声目标 RMS
TRAIN_SEGMENTS_FOR_LOUDNESS = 800
# 上述每段的时长（秒）
SEGMENT_DURATION_FOR_LOUDNESS = 0.5
# 目标噪声 RMS = 训练集平均 RMS × 该比例（例如 0.3 表示 30%）
LOUDNESS_RATIO = 0.3
# 全局采样率（Hz），与训练数据一致
SAMPLE_RATE = 16000

# ================= 随机种子 =================
# 若为 None 则每次运行随机；设为整数可复现
RANDOM_SEED = None

# ================= user + 噪声试听配置 =================
# 叠加时背景噪声相对 user 轨 RMS 的比例（例如 0.2 表示噪声约为 user 的 20%）
NOISE_MIX_RATIO = 0.4
# user 轨响度随时间平滑波动的范围 [下限, 上限]，例如 0.7~1.1 表示 70%~110%
USER_GAIN_LO, USER_GAIN_HI = 0.7, 1.1
# 背景噪声响度随时间平滑波动的范围（同上）
NOISE_GAIN_LO, NOISE_GAIN_HI = 0.5, 1.2
# 平滑增益曲线的关键帧间隔（秒），(min, max) 随机，越小变化越密
GAIN_KEYFRAME_INTERVAL = (1.2, 4.0)

# ================= 音频效果配置 =================
# 预设效果数量（混响、失真、滤波、合唱、相位等至少 100+）
NUM_AUDIO_EFFECTS = 520
# 生成效果预设时的随机种子，保证预设池可复现
EFFECT_PRESET_SEED = None
# Spotify pedalboard 效果强度比例范围：每种效果的 wet/mix 在此范围内随机，例如 (0.3, 0.8) 表示 30%~80% 效果量
PEDALBOARD_EFFECT_WET_LO, PEDALBOARD_EFFECT_WET_HI = 0.0, 0.5

# ================= user 轨随机升降调（混响前） =================
# 模拟不同声调的人说话：随机半音范围，正为升调（更尖锐），负为降调（更低沉），不改变语速
USER_PITCH_SEMITONES_LO, USER_PITCH_SEMITONES_HI = -60.2, 60.2


def load_wav(path, max_frames=None, start_frame=0):
    """Load wav to float32 mono [-1,1]. Returns (samples, sr) or (None, None)."""
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
    width = 2
    n = len(raw) // (width * nch)
    fmt = "<" + "h" * (n * nch)
    try:
        y = np.array(struct.unpack(fmt, raw[: n * nch * width]), dtype=np.float32)
    except Exception:
        return None, None
    if nch == 2:
        y = (y[::2] + y[1::2]) / 2.0
    y = y / 32768.0
    return y, sr


def resample_to_16k(y, orig_sr):
    if orig_sr == SAMPLE_RATE:
        return y
    n_old = len(y)
    n_new = int(round(n_old * SAMPLE_RATE / orig_sr))
    x_old = np.linspace(0, n_old - 1, n_old)
    x_new = np.linspace(0, n_old - 1, n_new)
    return np.interp(x_new, x_old, y).astype(np.float32)


def rms(y):
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(y ** 2)))


def get_train_non_silence_paths(limit=50000):
    paths = []
    if not os.path.isfile(TRAIN_JSONL):
        return paths
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ent = json.loads(line)
            except Exception:
                continue
            for p in ent.get("audios") or []:
                if "silence" not in os.path.normpath(p).lower():
                    paths.append(p)
            if len(paths) >= limit:
                break
    return paths


def compute_train_avg_rms(num_segments=800, segment_duration=0.5):
    paths = get_train_non_silence_paths()
    if not paths:
        print("  未找到训练集非静音路径，使用默认目标 RMS=0.02")
        return 0.02
    random.shuffle(paths)
    segment_frames = int(segment_duration * SAMPLE_RATE)
    rms_list = []
    for p in paths:
        if len(rms_list) >= num_segments:
            break
        y, sr = load_wav(p, max_frames=segment_frames)
        if y is None or len(y) < 100:
            continue
        if sr != SAMPLE_RATE:
            y = resample_to_16k(y, sr)
        y = y[:segment_frames]
        if len(y) < segment_frames // 2:
            continue
        r = rms(y)
        if r > 1e-6:
            rms_list.append(r)
    if not rms_list:
        print("  未能从训练集算出 RMS，使用默认 0.02")
        return 0.02
    avg_rms = float(np.mean(rms_list))
    print("  训练集非静音 %d 段 0.5s 平均 RMS: %.4f" % (len(rms_list), avg_rms))
    return avg_rms


def collect_wav_paths(dirs_or_file_dir):
    if isinstance(dirs_or_file_dir, (list, tuple)):
        paths = []
        for d in dirs_or_file_dir:
            if os.path.isdir(d):
                for root, _, names in os.walk(d):
                    for name in names:
                        if name.lower().endswith(".wav"):
                            paths.append(os.path.join(root, name))
        return paths
    d = dirs_or_file_dir
    if not os.path.isdir(d):
        return []
    paths = []
    for root, _, names in os.walk(d):
        for name in names:
            if name.lower().endswith(".wav"):
                paths.append(os.path.join(root, name))
    return paths


def crossfade_append(out, new_chunk, crossfade_samples):
    if crossfade_samples <= 0 or len(out) < crossfade_samples or len(new_chunk) < crossfade_samples:
        return np.concatenate([out, new_chunk])
    n = crossfade_samples
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[-n:] = out[-n:] * fade_out + new_chunk[:n] * fade_in
    return np.concatenate([out, new_chunk[n:]])


def build_noise_track(source_paths, target_duration_sec, sample_rate, crossfade_sec, rng):
    target_samples = int(round(target_duration_sec * sample_rate))
    crossfade_n = int(round(crossfade_sec * sample_rate))
    out = np.zeros(0, dtype=np.float32)
    indices = list(range(len(source_paths)))
    rng.shuffle(indices)
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
    out = out[:target_samples]
    return out


def normalize_to_target_rms(y, target_rms):
    current = rms(y)
    if current < 1e-9:
        return y
    scale = target_rms / current
    y = y * scale
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def load_one_random_train_entry(rng):
    """随机读一条训练数据，返回 dict 含 messages, audios，无则 None。"""
    if not os.path.isfile(TRAIN_JSONL):
        return None
    lines = []
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except Exception:
                continue
    if not lines:
        return None
    return rng.choice(lines)


def build_user_track(entry, sample_rate):
    """把一条训练数据的 user 轨（audios 偶数下标 0,2,4,...）拼成一条连续音频。"""
    audios = entry.get("audios") or []
    user_paths = [audios[i] for i in range(0, len(audios), 2)]
    chunks = []
    for p in user_paths:
        y, sr = load_wav(p)
        if y is None or len(y) < 10:
            continue
        if sr != sample_rate:
            y = resample_to_16k(y, sr)
        chunks.append(y)
    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32)


def pitch_shift_no_speed_change(y, semitones):
    """
    只改音高、不改语速：升调（semitones>0）更尖锐，降调（semitones<0）更低沉。
    通过先重采样压缩/拉伸再拉回原长度实现，保持样本数不变。
    """
    if abs(semitones) < 0.01:
        return y
    n = len(y)
    ratio = 2.0 ** (semitones / 12.0)
    n_mid = max(1, int(round(n / ratio)))
    x_old = np.linspace(0, n - 1, n)
    x_mid = np.linspace(0, n - 1, n_mid)
    y_mid = np.interp(x_mid, x_old, y).astype(np.float32)
    x_out = np.linspace(0, n_mid - 1, n)
    y_out = np.interp(x_out, np.arange(n_mid), y_mid).astype(np.float32)
    return y_out


def build_audio_effect_presets_pedalboard(num_presets=120, seed=12345, wet_lo=0.3, wet_hi=0.8):
    """
    用 pedalboard 生成 100+ 种音频效果；每种效果的效果量（wet/mix）在 [wet_lo, wet_hi] 内随机。
    """
    if not HAS_PEDALBOARD:
        return None
    r = random.Random(seed)
    presets = []
    effect_types = [
        "reverb", "chorus", "distortion", "phaser", "compressor",
        "lowpass", "highpass", "pitch_shift", "reverb+chorus",
        "distortion+lowpass", "phaser+reverb", "chorus+highpass"
    ]
    for i in range(num_presets):
        etype = effect_types[i % len(effect_types)]
        wet = r.uniform(wet_lo, wet_hi)
        board = Pedalboard()

        if "reverb" in etype:
            board.append(Reverb(
                room_size=r.uniform(0.3, 0.95),
                damping=r.uniform(0.3, 0.9),
                wet_level=wet,
                dry_level=r.uniform(0.4, 0.9)
            ))
        if "chorus" in etype:
            board.append(Chorus(
                rate_hz=r.uniform(0.5, 3.0),
                depth=r.uniform(0.2, 0.7),
                centre_delay_ms=r.uniform(5, 15),
                feedback=r.uniform(0.1, 0.4),
                mix=wet
            ))
        if "distortion" in etype:
            board.append(Distortion(drive_db=r.uniform(5, 25)))
        if "phaser" in etype:
            board.append(Phaser(
                rate_hz=r.uniform(0.3, 2.0),
                depth=r.uniform(0.3, 0.8),
                centre_frequency_hz=r.uniform(500, 2000),
                feedback=r.uniform(0.2, 0.6),
                mix=wet
            ))
        if "compressor" in etype:
            board.append(Compressor(
                threshold_db=r.uniform(-30, -10),
                ratio=r.uniform(2, 8)
            ))
        if "lowpass" in etype:
            board.append(LowpassFilter(cutoff_frequency_hz=r.uniform(1500, 5000)))
        if "highpass" in etype:
            board.append(HighpassFilter(cutoff_frequency_hz=r.uniform(100, 400)))
        if "pitch_shift" in etype and i % 20 == 0:
            board.append(PitchShift(semitones=r.uniform(-1.5, 1.5)))

        presets.append((etype, board))
    return presets


def build_audio_effect_presets_fallback(num_presets=120, seed=12345):
    """纯 numpy 实现 100+ 种音频效果（混响、滤波、失真、合唱、相位等）。"""
    r = random.Random(seed)
    presets = []
    effect_types = [
        "reverb_short", "reverb_long", "lowpass", "highpass", "bandpass",
        "distortion_soft", "distortion_hard", "chorus", "phaser", 
        "reverb+lowpass", "reverb+highpass", "distortion+bandpass"
    ]
    for i in range(num_presets):
        etype = effect_types[i % len(effect_types)]
        params = {}
        
        if "reverb" in etype:
            n_taps = r.randint(4, 9)
            if "short" in etype:
                delays_ms = sorted([r.uniform(25, 150) for _ in range(n_taps)])
            else:
                delays_ms = sorted([r.uniform(50, 500) for _ in range(n_taps)])
            gains = [r.uniform(0.08, 0.55) for _ in range(n_taps)]
            params["reverb"] = (delays_ms, gains)
        
        if "lowpass" in etype:
            params["lowpass_cutoff"] = r.uniform(1500, 5000)
        
        if "highpass" in etype:
            params["highpass_cutoff"] = r.uniform(150, 600)
        
        if "bandpass" in etype:
            center = r.uniform(800, 3000)
            width = r.uniform(400, 1500)
            params["bandpass"] = (center, width)
        
        if "distortion" in etype:
            if "soft" in etype:
                params["distortion_gain"] = r.uniform(2, 6)
            else:
                params["distortion_gain"] = r.uniform(8, 20)
        
        if "chorus" in etype:
            params["chorus_delay"] = r.uniform(15, 35)
            params["chorus_depth"] = r.uniform(0.3, 0.7)
            params["chorus_rate"] = r.uniform(0.5, 2.5)
        
        if "phaser" in etype:
            params["phaser_rate"] = r.uniform(0.3, 2.0)
            params["phaser_depth"] = r.uniform(0.4, 0.8)
        
        presets.append((etype, params))
    return presets


def apply_numpy_effect(y, sample_rate, params):
    """纯 numpy 实现音频效果。"""
    out = y.copy()
    
    # 混响（延迟叠加）
    if "reverb" in params:
        delays_ms, gains = params["reverb"]
        n = len(out)
        for d_ms, g in zip(delays_ms, gains):
            d_samp = int(sample_rate * d_ms / 1000.0)
            if d_samp < n:
                out[d_samp:] += y[: n - d_samp] * g
    
    # 低通滤波（简单平均）
    if "lowpass_cutoff" in params:
        fc = params["lowpass_cutoff"]
        kernel_len = max(3, int(sample_rate / fc * 2))
        if kernel_len % 2 == 0:
            kernel_len += 1
        kernel = np.ones(kernel_len) / kernel_len
        out = np.convolve(out, kernel, mode="same").astype(np.float32)
    
    # 高通滤波（原信号减低通）
    if "highpass_cutoff" in params:
        fc = params["highpass_cutoff"]
        kernel_len = max(3, int(sample_rate / fc * 4))
        if kernel_len % 2 == 0:
            kernel_len += 1
        kernel = np.ones(kernel_len) / kernel_len
        lowpassed = np.convolve(out, kernel, mode="same")
        out = (out - lowpassed * 0.9).astype(np.float32)
    
    # 带通滤波（高通+低通）
    if "bandpass" in params:
        center, width = params["bandpass"]
        low_fc = max(100, center - width / 2)
        high_fc = min(sample_rate / 2 - 100, center + width / 2)
        k1 = max(3, int(sample_rate / high_fc * 4))
        if k1 % 2 == 0:
            k1 += 1
        lowpassed = np.convolve(out, np.ones(k1) / k1, mode="same")
        highpassed = out - lowpassed * 0.9
        k2 = max(3, int(sample_rate / low_fc * 2))
        if k2 % 2 == 0:
            k2 += 1
        out = np.convolve(highpassed, np.ones(k2) / k2, mode="same").astype(np.float32)
    
    # 失真（软/硬削波）
    if "distortion_gain" in params:
        gain = params["distortion_gain"]
        out = out * gain
        out = np.tanh(out).astype(np.float32)
    
    # 合唱（原信号+延迟+轻微调制）
    if "chorus_delay" in params:
        delay_ms = params["chorus_delay"]
        depth = params["chorus_depth"]
        rate = params["chorus_rate"]
        d_samp = int(sample_rate * delay_ms / 1000.0)
        n = len(out)
        if d_samp < n:
            t = np.arange(n) / float(sample_rate)
            mod = (1.0 + depth * np.sin(2 * np.pi * rate * t)).astype(np.float32)
            delayed = np.zeros(n, dtype=np.float32)
            delayed[d_samp:] = y[: n - d_samp]
            out = (out + delayed * mod * 0.6).astype(np.float32)
    
    # 相位器（正弦调制延迟）
    if "phaser_rate" in params:
        rate = params["phaser_rate"]
        depth = params["phaser_depth"]
        n = len(out)
        t = np.arange(n) / float(sample_rate)
        mod_delay_ms = 1 + depth * (np.sin(2 * np.pi * rate * t) + 1) / 2 * 3
        mod_samp = (mod_delay_ms * sample_rate / 1000.0).astype(int)
        phased = np.zeros(n, dtype=np.float32)
        for i in range(n):
            src = i - mod_samp[i]
            if 0 <= src < n:
                phased[i] = y[src]
        out = (out * 0.7 + phased * 0.5).astype(np.float32)
    
    # 防溢出
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def apply_audio_effect(y, sample_rate, preset):
    """应用预设效果：pedalboard 或纯 numpy。"""
    effect_type, effect_data = preset
    
    if HAS_PEDALBOARD and isinstance(effect_data, Pedalboard):
        try:
            out = effect_data(y, sample_rate)
            return out.astype(np.float32)
        except Exception as e:
            print("  pedalboard 应用失败（%s），使用 numpy 备用" % str(e))
            return y.astype(np.float32)
    else:
        return apply_numpy_effect(y, sample_rate, effect_data)


def smooth_random_gain_curve(n_samples, sample_rate, gain_lo, gain_hi, interval_range, rng):
    """
    随时间随机平滑波动的增益曲线，范围 [gain_lo, gain_hi]。
    interval_range: (min_sec, max_sec) 关键帧间隔，线性插值。
    """
    t_end = n_samples / float(sample_rate)
    times = [0.0]
    gains = [rng.uniform(gain_lo, gain_hi)]
    while times[-1] < t_end:
        step = rng.uniform(interval_range[0], interval_range[1])
        times.append(times[-1] + step)
        gains.append(rng.uniform(gain_lo, gain_hi))
    times = np.array(times, dtype=np.float32)
    gains = np.array(gains, dtype=np.float32)
    t_all = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
    curve = np.interp(t_all, times, gains)
    return curve


def mix_noise_to_length(noise_1min, target_samples, rng):
    """把 1 分钟噪声裁成或循环成 target_samples 长，随机起点。"""
    n = len(noise_1min)
    if n >= target_samples:
        start = rng.randint(0, n - target_samples) if n > target_samples else 0
        return noise_1min[start : start + target_samples].copy()
    # 循环拼接
    out = np.zeros(target_samples, dtype=np.float32)
    pos = rng.randint(0, n - 1) if n > 1 else 0
    written = 0
    while written < target_samples:
        take = min(n - pos, target_samples - written)
        out[written : written + take] = noise_1min[pos : pos + take]
        written += take
        pos = (pos + take) % n
    return out


def write_wav(path, y, sample_rate=16000):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    y_int = (y * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(y_int.tobytes())


def parse_path_list(value):
    if not value:
        return []
    return [item for item in re.split(r"[,;]", value) if item]


def main():
    global TRAIN_JSONL, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Build MUSAN/FSD50K background-noise preview wav files.")
    parser.add_argument("--train-jsonl", default=TRAIN_JSONL, help="Optional JSONL used to estimate user RMS.")
    parser.add_argument("--musan-noise-dirs", default=",".join(MUSAN_NOISE_DIRS), help="Comma-separated MUSAN noise wav directories; may be empty.")
    parser.add_argument("--fsd50k-audio-dir", default=FSD50K_AUDIO, help="FSD50K.dev_audio wav directory; may be empty.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for preview wav outputs.")
    args = parser.parse_args()
    TRAIN_JSONL = args.train_jsonl
    OUTPUT_DIR = args.output_dir
    musan_noise_dirs = parse_path_list(args.musan_noise_dirs)
    fsd50k_audio_dir = args.fsd50k_audio_dir.strip()

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)

    print("=" * 60)
    print("激情在燃烧！合成 1 分钟背景噪声试听")
    print("=" * 60)

    target_rms = LOUDNESS_RATIO * compute_train_avg_rms(
        num_segments=TRAIN_SEGMENTS_FOR_LOUDNESS,
        segment_duration=SEGMENT_DURATION_FOR_LOUDNESS,
    )
    print("  目标噪声响度(30%% 训练平均): RMS = %.4f" % target_rms)

    musan_paths = collect_wav_paths(musan_noise_dirs)
    fsd_paths = collect_wav_paths(fsd50k_audio_dir) if fsd50k_audio_dir else []
    print("  MUSAN 噪声文件数: %d" % len(musan_paths))
    print("  FSD50K 文件数: %d" % len(fsd_paths))
    if not musan_paths and not fsd_paths:
        raise FileNotFoundError(
            "找不到 MUSAN/FSD50K wav。请设置 --musan-noise-dirs、--fsd50k-audio-dir 或 NOISE_ROOT。"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    y_musan = None
    y_fsd = None
    if musan_paths:
        print("\n  正在合成 MUSAN 1 分钟...")
        y_musan = build_noise_track(
            musan_paths, TARGET_DURATION_SEC, SAMPLE_RATE, CROSSFADE_SEC, rng
        )
        if y_musan is not None:
            y_musan = normalize_to_target_rms(y_musan, target_rms)
            out_musan = os.path.join(OUTPUT_DIR, "noise_musan_1min.wav")
            write_wav(out_musan, y_musan, SAMPLE_RATE)
            print("  已写出: %s" % out_musan)
    else:
        print("  MUSAN 无 wav，跳过")

    if fsd_paths:
        print("\n  正在合成 FSD50K 1 分钟...")
        y_fsd = build_noise_track(
            fsd_paths, TARGET_DURATION_SEC, SAMPLE_RATE, CROSSFADE_SEC, rng
        )
        if y_fsd is not None:
            y_fsd = normalize_to_target_rms(y_fsd, target_rms)
            out_fsd = os.path.join(OUTPUT_DIR, "noise_fsd50k_1min.wav")
            write_wav(out_fsd, y_fsd, SAMPLE_RATE)
            print("  已写出: %s" % out_fsd)
    else:
        print("  FSD50K 无 wav，跳过")

    # ---------- 随机一条训练数据：user 轨 + 随机升降调 + 100+ 种音频效果随机一种 + 平滑响度波动 ----------
    if HAS_PEDALBOARD:
        effect_presets = build_audio_effect_presets_pedalboard(
            NUM_AUDIO_EFFECTS, EFFECT_PRESET_SEED,
            wet_lo=PEDALBOARD_EFFECT_WET_LO, wet_hi=PEDALBOARD_EFFECT_WET_HI
        )
        print("\n  音频效果池（pedalboard，效果量 %.0f%%-%.0f%% 随机）: %d 种" % (
            PEDALBOARD_EFFECT_WET_LO * 100, PEDALBOARD_EFFECT_WET_HI * 100, len(effect_presets)))
    else:
        effect_presets = build_audio_effect_presets_fallback(NUM_AUDIO_EFFECTS, EFFECT_PRESET_SEED)
        print("\n  音频效果池（纯 numpy 混响/滤波/失真/合唱/相位等）: %d 种" % len(effect_presets))

    print("\n" + "=" * 60)
    print("激情在燃烧！user 70%%-110%% 平滑波动 + 100+ 音频效果随机一种 + 噪声 50%%-120%% 平滑波动")
    print("=" * 60)
    entry = load_one_random_train_entry(rng)
    if entry is None:
        print("  无训练数据，跳过 user+噪声试听")
    else:
        user_raw = build_user_track(entry, SAMPLE_RATE)
        if user_raw is None or len(user_raw) < SAMPLE_RATE:
            print("  该条 user 轨过短或为空，跳过")
        else:
            n_user = len(user_raw)
            dur = n_user / float(SAMPLE_RATE)
            print("  随机选中一条，user 轨拼接时长: %.1f s" % dur)
            preset = rng.choice(effect_presets)
            effect_type = preset[0]
            print("  随机音频效果类型: %s" % effect_type)
            user_reverb = apply_audio_effect(user_raw, SAMPLE_RATE, preset)
            user_gain = smooth_random_gain_curve(
                n_user, SAMPLE_RATE, USER_GAIN_LO, USER_GAIN_HI,
                GAIN_KEYFRAME_INTERVAL, rng
            )
            user_reverb = (user_reverb * user_gain).astype(np.float32)

            noise_base = y_musan if musan_paths and y_musan is not None else (y_fsd if fsd_paths and y_fsd is not None else None)
            if noise_base is not None:
                noise_same_len = mix_noise_to_length(noise_base, n_user, rng)
                user_rms = rms(user_reverb)
                if user_rms > 1e-9:
                    noise_rms_target = user_rms * NOISE_MIX_RATIO
                    noise_same_len = normalize_to_target_rms(noise_same_len, noise_rms_target)
                noise_gain = smooth_random_gain_curve(
                    n_user, SAMPLE_RATE, NOISE_GAIN_LO, NOISE_GAIN_HI,
                    GAIN_KEYFRAME_INTERVAL, rng
                )
                noise_same_len = (noise_same_len * noise_gain).astype(np.float32)
                mixed = user_reverb + noise_same_len
                mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
            else:
                mixed = user_reverb

            prefix = os.path.join(OUTPUT_DIR, "demo_user")
            write_wav(prefix + "_raw.wav", user_raw, SAMPLE_RATE)
            write_wav(prefix + "_reverb.wav", user_reverb, SAMPLE_RATE)
            if noise_base is not None:
                write_wav(prefix + "_reverb_noise.wav", mixed, SAMPLE_RATE)
            print("  已写出: %s_raw.wav（原始 user 轨）" % prefix)
            print("  已写出: %s_reverb.wav（100+ 音频效果随机一种 + user 70%%-110%% 平滑波动）" % prefix)
            print("  已写出: %s_reverb_noise.wav（+ 噪声 50%%-120%% 平滑波动，噪声不加效果）" % prefix)

    print("\n" + "=" * 60)
    print("冲！试听完毕再调 CROSSFADE / LOUDNESS_RATIO / 混响参数")
    print("=" * 60)


if __name__ == "__main__":
    main()
