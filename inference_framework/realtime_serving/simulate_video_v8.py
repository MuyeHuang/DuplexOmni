import asyncio
import os
import wave
import numpy as np
import json
import base64
import logging
import re
import time
import random
import io
import ast
from collections import deque
from openai import AsyncOpenAI
import requests
import websockets
import av
from PIL import Image

# ================= 配置区域 =================
S1_OMNI_BASE_URL = "http://127.0.0.1:21000/v1"
S1_AUDIO_WS_BASE = "ws://127.0.0.1:21000/v1/audio/stream"
S1_CLEAR_SESSION_BASE = "http://127.0.0.1:21000/v1/session"
S1_HEALTH_URL = "http://127.0.0.1:21000/health"
S1_MODEL_NAME = os.environ.get("S1_MODEL_NAME", "models/qwen3-omni-checkpoint")

S2_THINK_BASE_URL = os.environ.get("S2_THINK_BASE_URL", "http://localhost:8000/v1")
S2_MODEL_NAME = os.environ.get("S2_MODEL_NAME", "gpt-4.1")
S2_API_KEY = os.environ.get("S2_API_KEY", "EMPTY")
S1_API_KEY = os.environ.get("S1_API_KEY", "EMPTY")

# TTS 已移除：S1 使用完整 Qwen3-Omni，直接返回 text + audio

# 视频输入路径（支持任意格式）
TEST_INPUT_VIDEO = os.environ.get("TEST_INPUT_VIDEO", "assets/input.mp4")
OUTPUT_FILENAME = os.environ.get("OUTPUT_FILENAME", "outputs/simulate_video_output.wav")

# 每隔多少个 chunk 插入一帧（和训练数据保持一致）
VIDEO_FRAME_INTERVAL = 4

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPWIDTH = 2
CHUNK_MS = 480
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
BYTES_PER_SAMPLE = 2
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE
AUDIO_STITCH_JITTER_MS = 100
AUDIO_STITCH_JITTER_SAMPLES = int(SAMPLE_RATE * AUDIO_STITCH_JITTER_MS / 1000)

# 缓冲策略参数
FIRST_PACKET_SIZE = 2           # 首包最少 token 数
FIRST_PACKET_MAX_SIZE = 99      # 首包最大 token 数（超过则强制发送，即使没有标点）
SUBSEQUENT_PACKET_MIN_SIZE = 5  # 后续包最少 token 数（有标点时的最小值）
SUBSEQUENT_PACKET_SIZE = 15     # 后续包基础最大 token 数
SUBSEQUENT_PACKET_MAX_SIZE = 25 # 后续包动态缓冲的上限
BUFFER_MULTIPLIER = 3           # 发送 N 个字可以缓冲 N*3 个字
MIN_PUNCT_RATIO = 0.67          # 大缓冲时，标点必须在后 1/3 才截断（位置 > 总量的 2/3）
FLUSH_TIMEOUT = 1.51

# S2 指令放行间隔：按仿真时间轴（每拍 CHUNK_MS），与 HTTP 墙钟耗时无关
S2_CMD_COOLDOWN_MIN_S = 0.5  # 仿真秒内随机间隔下限
S2_CMD_COOLDOWN_MAX_S = 1.0  # 仿真秒内随机间隔上限

# 标点符号集合
PUNCTUATION = set('，。！？；：、,!?.;:')

S2_SYSTEM_PROMPT = """你是指导 System 1 说话的 System 2，当前场景是视频通话。用户和 System 1 正在共同观看同一段视频。

你的任务是根据当前视频画面内容，指导 System 1 自然地回应用户的观察和问题。给出简短、口语化、可直接念出的话术。每条用【】包起来。

核心风格：
- 回应要锚定画面里可见的具体细节，不要泛泛而谈。
- 不要提及画面中还没出现的内容。
必须说的多一些，丰富一些。多用【】包裹写东西。发送 2 个【】，描述过的东西就不要再重复说了
仔细看用户在问什么，最后给出答案。问的是英文就说英文。
"""


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ================= 辅助工具函数 =================

def tokenize_text(text):
    """将文本分割为字符单元，英文单词作为一个整体，标点符号不计入token"""
    tokens = []
    i = 0
    while i < len(text):
        char = text[i]
        
        if char in PUNCTUATION:
            if tokens:
                tokens[-1] = (tokens[-1][0] + char, True)  # 标点符号，不计入token
            else:
                tokens.append((char, True))
            i += 1
        elif '\u4e00' <= char <= '\u9fff':
            tokens.append((char, False))  # 中文字符，计入token
            i += 1
        elif char.isalpha():
            word = ''
            while i < len(text) and (text[i].isalpha() or text[i] == "'") and not ('\u4e00' <= text[i] <= '\u9fff'):
                word += text[i]
                i += 1
            tokens.append((word, False))  # 英文单词，计入token
        elif char.isdigit():
            num = ''
            while i < len(text) and text[i].isdigit():
                num += text[i]
                i += 1
            tokens.append((num, False))  # 数字，计入token
        else:
            if tokens:
                tokens[-1] = (tokens[-1][0] + char, True)  # 其他字符，不计入token
            else:
                tokens.append((char, True))
            i += 1
    return tokens

def count_tokens(text):
    """统计文本中的有效token数量（不包括标点符号）"""
    tokens = tokenize_text(text)
    return sum(1 for _, is_punctuation in tokens if not is_punctuation)

def get_text_by_token_count(text, target_token_count):
    """从文本中提取指定token数量的内容"""
    tokens = tokenize_text(text)
    result = ""
    token_count = 0
    
    for token_text, is_punctuation in tokens:
        result += token_text
        if not is_punctuation:
            token_count += 1
            if token_count >= target_token_count:
                break
    
    return result

def find_punctuation_split_point(text, min_tokens, max_tokens, min_punct_ratio=0.0):
    """
    在文本中找到合适的标点分割点
    
    策略：
    1. 在 [min_tokens, max_tokens] 范围内寻找最后一个标点符号位置
    2. 如果 min_punct_ratio > 0，只有标点位置 > max_tokens * min_punct_ratio 才算有效
       （用于大缓冲时避免在前 2/3 位置截断）
    3. 如果找到有效标点，返回标点符号之后的位置（包含标点）
    4. 如果没找到标点，但 token 数 >= max_tokens，返回 max_tokens 对应的位置
    5. 如果 token 数 < min_tokens，返回 None（不够发送）
    
    返回: (切分文本, 是否在标点处切分) 或 (None, False)
    """
    tokens = tokenize_text(text)
    
    if not tokens:
        return None, False
    
    # 构建位置映射：每个 token 结束后的累积文本和 token 计数
    positions = []  # [(累积文本, 累积token数, 是否以标点结尾)]
    cumulative_text = ""
    token_count = 0
    
    for token_text, is_punctuation in tokens:
        cumulative_text += token_text
        if not is_punctuation:
            token_count += 1
        
        # 检查这个 token 是否以标点结尾（标点符号会附加到前一个 token）
        ends_with_punct = any(cumulative_text.endswith(p) for p in PUNCTUATION)
        positions.append((cumulative_text, token_count, ends_with_punct))
    
    total_tokens = token_count
    
    # 如果总 token 数不够最小值，返回 None
    if total_tokens < min_tokens:
        return None, False
    
    # 计算标点的最小有效位置（用于大缓冲时限制只在后部截断）
    min_punct_position = int(max_tokens * min_punct_ratio) if min_punct_ratio > 0 else min_tokens
    
    # 在 [min_tokens, max_tokens] 范围内寻找最后一个标点位置
    # 但标点位置必须 >= min_punct_position
    last_punct_text = None
    last_punct_found = False
    
    for cumulative_text, cum_token_count, ends_with_punct in positions:
        if cum_token_count < min_tokens:
            continue
        if cum_token_count > max_tokens:
            break
        # 只有当标点位置超过最小有效位置时才算有效
        if ends_with_punct and cum_token_count >= min_punct_position:
            last_punct_text = cumulative_text
            last_punct_found = True
    
    # 如果在范围内找到了有效标点，在标点处切分
    if last_punct_found and last_punct_text:
        return last_punct_text, True
    
    # 没找到有效标点，检查是否达到 max_tokens
    if total_tokens >= max_tokens:
        return get_text_by_token_count(text, max_tokens), False
    
    # token 数在 [min_tokens, max_tokens) 之间但没有有效标点，不发送（等更多内容或超时）
    return None, False

# ================= 辅助工具类 =================

class AudioBuffer:
    def __init__(self):
        self.buffer = bytearray()
        self.lock = asyncio.Lock()
    
    async def push(self, data: bytes):
        async with self.lock:
            if data:
                if len(data) % 2 != 0:
                    data = data[:-1]
                self.buffer.extend(data)
    
    async def pop(self, num_bytes: int) -> bytes:
        async with self.lock:
            if len(self.buffer) < num_bytes:
                result = self.buffer[:]
                padding = b'\x00' * (num_bytes - len(self.buffer))
                self.buffer.clear()
                return bytes(result) + padding
            else:
                result = self.buffer[:num_bytes]
                del self.buffer[:num_bytes]
                return bytes(result)

    async def pop_existing(self, num_bytes: int) -> bytes:
        async with self.lock:
            count = min(len(self.buffer), num_bytes)
            result = self.buffer[:count]
            del self.buffer[:count]
            return bytes(result)

    async def clear(self):
        async with self.lock:
            self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)
        
def trim_pcm_silence(pcm_data, threshold=1000):
    if not pcm_data:
        return b""
    audio_array = np.frombuffer(pcm_data, dtype=np.int16)
    non_silent_indices = np.where(np.abs(audio_array) > threshold)[0]
    if non_silent_indices.size == 0:
        return b""
    start_index = non_silent_indices[0]
    end_index = non_silent_indices[-1] + 1
    return audio_array[start_index:end_index].tobytes()

def trim_pcm_silence_left(pcm_data, threshold=1000):
    """只 trim 左边（开头）的静音"""
    if not pcm_data:
        return b""
    audio_array = np.frombuffer(pcm_data, dtype=np.int16)
    non_silent_indices = np.where(np.abs(audio_array) > threshold)[0]
    if non_silent_indices.size == 0:
        return b""
    start_index = non_silent_indices[0]
    return audio_array[start_index:].tobytes()

def trim_pcm_silence_right(pcm_data, threshold=1000):
    """只 trim 右边（结尾）的静音"""
    if not pcm_data:
        return b""
    audio_array = np.frombuffer(pcm_data, dtype=np.int16)
    non_silent_indices = np.where(np.abs(audio_array) > threshold)[0]
    if non_silent_indices.size == 0:
        return b""
    end_index = non_silent_indices[-1] + 1
    return audio_array[:end_index].tobytes()

def preprocess_audio_noise_gate(file_path, threshold=0):
    if not os.path.exists(file_path): return
    with wave.open(file_path, 'rb') as wf:
        params = wf.getparams()
        if params.sampwidth != 2: return
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).copy()
    mask = np.abs(audio) < threshold
    audio[mask] = 0
    with wave.open(file_path, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(audio.tobytes())

def pcm_to_b64_wav(pcm_bytes):
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPWIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        return base64.b64encode(wav_io.getvalue()).decode('utf-8')


def wav_bytes_to_pcm(wav_bytes: bytes) -> bytes:
    if not wav_bytes:
        return b""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes())


def is_pcm_silent(pcm_bytes: bytes, rms_threshold: float = 50.0) -> bool:
    if not pcm_bytes or len(pcm_bytes) < 2:
        return True
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64)
    if arr.size == 0:
        return True
    rms = np.sqrt(np.mean(arr ** 2))
    return rms < rms_threshold


def stretch_pcm_to_chunk(pcm_bytes: bytes, target_ms: int = CHUNK_MS) -> bytes:
    """
    将 PCM 拉伸或截断到恰好 target_ms（默认 480ms），避免每段偏短时产生细碎断裂。
    - 若不足 target_ms：线性插值拉伸到 target_ms。
    - 若超过 target_ms：只取前 target_ms，多出的部分由调用方决定是否入 buffer。
    """
    if not pcm_bytes:
        return b"\x00" * (int(SAMPLE_RATE * target_ms / 1000) * BYTES_PER_SAMPLE)
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    current_samples = len(arr)
    target_samples = int(SAMPLE_RATE * target_ms / 1000)
    if current_samples >= target_samples:
        return arr[:target_samples].tobytes()
    # 拉伸：线性插值 current_samples -> target_samples
    x_old = np.linspace(0, 1, current_samples, dtype=np.float64)
    x_new = np.linspace(0, 1, target_samples, dtype=np.float64)
    interpolated = np.interp(x_new, x_old, arr.astype(np.float64))
    return np.clip(interpolated, -32768, 32767).astype(np.int16).tobytes()


# ================= Qwen3-Omni 音频解码 =================

def decode_omni_audio(completion) -> bytes:
    """
    从 Qwen3-Omni /v1/chat/completions 的 response 中解析出 PCM 音频字节。
    支持两种格式：
    1) vllm-omni：同一 choice 内 message.audio 为带 .data 的对象（base64）；
    2) 部分实现：choices[0]=text, choices[1]=audio。
    """
    pcm = b""
    try:
        if not getattr(completion, "choices", None) or len(completion.choices) < 1:
            return pcm
        # 先看 choices[0].message.audio（vllm-omni 官方格式）
        msg = completion.choices[0].message
        audio_obj = getattr(msg, "audio", None)
        if audio_obj is None and len(completion.choices) >= 2:
            msg = completion.choices[1].message
            audio_obj = getattr(msg, "audio", None)
        if audio_obj is None:
            return pcm
        if hasattr(audio_obj, "data"):
            raw = base64.b64decode(audio_obj.data)
        elif isinstance(audio_obj, str):
            raw = base64.b64decode(audio_obj)
        elif isinstance(audio_obj, dict):
            raw = base64.b64decode(audio_obj.get("data", audio_obj.get("bytes", "")))
        else:
            return pcm
        if not raw:
            return pcm
        if raw[:4] == b"RIFF":
            with wave.open(io.BytesIO(raw), "rb") as wf:
                pcm = wf.readframes(wf.getnframes())
        else:
            pcm = raw
    except Exception as e:
        logging.warning(f"decode_omni_audio failed: {e}")
    return pcm

# ================= S2 智能体模块 (完全重构为持续运行模式) =================

class System2Agent:
    def __init__(self):
        self.client = AsyncOpenAI(base_url=S2_THINK_BASE_URL, api_key=S2_API_KEY)
        self.history = [{"role": "system", "content": S2_SYSTEM_PROMPT}]
        self.context_buffer = []
        self.output_commands = deque()
        
        self.is_thinking = False
        self.should_stop = False
        self.thinking_task = None
        self.pending_context = None
        self._pending_frames_b64: list = []  # 当次 THINK 的帧，不存入 history
        # 下一条 S2 指令最早可在第几个仿真 chunk（浮点，与 stimulate 时间线一致）
        self._next_cmd_allowed_chunk: float = 0.0

        self.lock = asyncio.Lock()

    def add_context(self, source, text):
        """添加上下文片段到缓冲区"""
        if not text: return
        self.context_buffer.append(f"[{source}]: {text}")
        print(f"[S2 收到上下文] [{source}]: {text}")
        logging.info(f"📝 S2 Context Added: [{source}] {text[:50]}...")

    async def trigger_think(self, frames_b64: list = None):
        """S1发送THINK信号：启动或重启S2思考。frames_b64 为触发时刻前的视频帧列表（base64 JPEG）。"""
        async with self.lock:
            logging.info(f"🧠 S2 Received [THINK] Signal. Current state: thinking={self.is_thinking}")

            new_context = self._consolidate_context()

            if self.is_thinking:
                logging.info("⚠️ S2 Interrupting current thinking...")
                self.should_stop = True
                if self.thinking_task and not self.thinking_task.done():
                    try:
                        await asyncio.wait_for(self.thinking_task, timeout=1.0)
                    except asyncio.TimeoutError:
                        self.thinking_task.cancel()
                        try:
                            await self.thinking_task
                        except asyncio.CancelledError:
                            pass
            
            # 文字上下文存入 history，帧单独缓存不存 history（避免帧在 history 中累积超限）
            if new_context:
                self.history.append({"role": "user", "content": new_context})
            self._pending_frames_b64 = list(frames_b64) if frames_b64 else []
            logging.info(f"🧠 S2 Context Consolidated (frames={len(self._pending_frames_b64)}):\n{new_context}")

            self.should_stop = False
            self.is_thinking = True
            self.thinking_task = asyncio.create_task(self._continuous_thinking())

    async def trigger_wait(self):
        """S1发送WAIT信号：立即停止S2思考"""
        async with self.lock:
            logging.info(f"🛑 S2 Received [WAIT] Signal. Current state: thinking={self.is_thinking}")
            if self.is_thinking:
                self.should_stop = True
                if self.thinking_task and not self.thinking_task.done():
                    try:
                        await asyncio.wait_for(self.thinking_task, timeout=1.0)
                    except asyncio.TimeoutError:
                        self.thinking_task.cancel()
                        try:
                            await self.thinking_task
                        except asyncio.CancelledError:
                            pass
                self.is_thinking = False
                logging.info("✅ S2 Stopped thinking")

    def _consolidate_context(self):
        """合并上下文缓冲区，相同说话人的连续消息合并"""
        if not self.context_buffer:
            return ""
        
        parsed = []
        for item in self.context_buffer:
            match = re.match(r"\[(User|Agent)\]: (.*)", item, re.DOTALL)
            if match:
                parsed.append({"source": match.group(1), "text": match.group(2)})
        
        if not parsed:
            result = "\n".join(self.context_buffer)
            self.context_buffer = []
            return result
        
        consolidated = []
        current_source = parsed[0]["source"]
        current_text = ""
        
        for item in parsed:
            if item["source"] == current_source:
                current_text += item["text"]
            else:
                consolidated.append(f"[{current_source}]: {current_text}")
                current_source = item["source"]
                current_text = item["text"]
        
        consolidated.append(f"[{current_source}]: {current_text}")
        self.context_buffer = []
        return "\n".join(consolidated)

    async def _continuous_thinking(self):
        """持续的思考任务：流式输出直到被外部信号中断或自然完成"""
        full_response = ""
        last_match_end = 0
        pattern = re.compile(r"【(.*?)】")
        
        try:
            logging.info("🧠 S2 Started Continuous Thinking...")
            # 帧作为临时消息拼入请求，不存入 history，避免帧累积超限
            messages_for_request = list(self.history)
            if self._pending_frames_b64:
                frame_content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    for b64 in self._pending_frames_b64
                ]
                frame_content.append({"type": "text", "text": "以上是当前视频画面帧序列，请根据画面内容指导 System 1 回应。"})
                messages_for_request.append({"role": "user", "content": frame_content})
                self._pending_frames_b64 = []

            stream = await self.client.chat.completions.create(
                model=S2_MODEL_NAME,
                messages=messages_for_request,
                stream=True,
                temperature=1.0,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
                max_tokens=2048
            )
            
            async for chunk in stream:
                if self.should_stop:
                    logging.info("⏸️ S2 Thinking interrupted by external signal")
                    break
                
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_response += content
                    print(f"[S2 流式收到] {content}", end="", flush=True)
                    for match in pattern.finditer(full_response, last_match_end):
                        command_content = match.group(1)
                        full_command = f"【{command_content}】"
                        logging.info(f"🧠 S2 Command Extracted: {full_command}")
                        self.output_commands.append(full_command)
                        last_match_end = match.end()
            
            if full_response:
                self.history.append({"role": "assistant", "content": full_response})
                print()  # S2 流式换行
                print(f"[S2 完整收到文本] {full_response}")
                logging.info(f"✅ S2 Thinking Complete. Total output: {len(full_response)} chars")
            
        except asyncio.CancelledError:
            logging.info("❌ S2 Thinking task cancelled")
            raise
        except Exception as e:
            logging.error(f"❌ S2 Thinking Error: {e}", exc_info=True)
        finally:
            async with self.lock:
                self.is_thinking = False
                self.should_stop = False

    def get_new_commands(self, sim_chunk_idx: int):
        """S1 获取 S2 命令。间隔按仿真 chunk：每拍长度 CHUNK_MS，与请求耗时的墙钟时间无关。"""
        if not self.output_commands:
            return ""
        if float(sim_chunk_idx) + 1e-9 < self._next_cmd_allowed_chunk:
            return ""
        cmd = self.output_commands.popleft()
        gap_s = random.uniform(S2_CMD_COOLDOWN_MIN_S, S2_CMD_COOLDOWN_MAX_S)
        sec_per_chunk = CHUNK_MS / 1000.0
        gap_chunks = gap_s / sec_per_chunk
        self._next_cmd_allowed_chunk = float(sim_chunk_idx) + gap_chunks
        logging.info(
            "🚦 S2 指令放行 sim_chunk=%d 队列剩余=%d 仿真间隔=%.2fs(≈%.2f拍) 下次≥chunk=%.2f: %s",
            sim_chunk_idx,
            len(self.output_commands),
            gap_s,
            gap_chunks,
            self._next_cmd_allowed_chunk,
            cmd,
        )
        return cmd

# ================= 仿真主循环 =================

def extract_video_audio_pcm(video_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """从视频中提取音频轨，重采样到 target_sr，返回 int16 PCM numpy 数组。"""
    container = av.open(video_path)
    audio_stream = container.streams.audio[0]
    orig_sr = audio_stream.rate
    samples = []
    for pkt in container.demux(audio_stream):
        for frame in pkt.decode():
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            samples.append(arr.astype(np.float32))
    container.close()
    audio_f32 = np.concatenate(samples)
    # int16 原始值需先归一化到 [-1, 1]，否则乘以 32767 会溢出
    if audio_stream.format.name in ('s16', 's16p'):
        audio_f32 = audio_f32 / 32768.0
    elif audio_stream.format.name in ('s32', 's32p'):
        audio_f32 = audio_f32 / 2147483648.0
    # 此时 audio_f32 在 [-1, 1]，resample 后乘回 int16 范围
    if orig_sr != target_sr:
        import librosa
        audio_f32 = librosa.resample(audio_f32, orig_sr=orig_sr, target_sr=target_sr)
    return np.clip(audio_f32 * 32767, -32768, 32767).astype(np.int16)


def _resize_frame(img: Image.Image, short_side: int, quality: int = 85) -> bytes:
    w, h = img.size
    if h > w:
        new_h, new_w = short_side, int(short_side * w / h)
    else:
        new_w, new_h = short_side, int(short_side * h / w)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def extract_frame_at_chunk(video_path: str, chunk_idx: int, chunk_ms: int = CHUNK_MS):
    """提取 chunk_idx * chunk_ms 时间点的视频帧。
    返回 (s1_jpg_bytes, s2_jpg_bytes)，s1 为 784p/q85，s2 为 720p/q95。
    失败时返回 (b'', b'')。
    """
    time_sec = chunk_idx * chunk_ms / 1000.0
    container = av.open(video_path)
    vs = container.streams.video[0]
    duration = float(vs.duration * vs.time_base) if vs.duration else 0.0
    target_sec = min(max(time_sec, 0.0), max(duration - 0.01, 0.0))
    target_pts = int(target_sec / float(vs.time_base))
    container.seek(target_pts, stream=vs)
    frame = None
    for f in container.decode(vs):
        frame = f
        break
    container.close()
    if frame is None:
        return b"", b""
    img = frame.to_image()
    return _resize_frame(img, 784), _resize_frame(img, 720, quality=95)


class OfflineSimulator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.s1_client = AsyncOpenAI(base_url=S1_OMNI_BASE_URL, api_key=S1_API_KEY)

        logging.info(f"📹 从视频提取音频: {video_path}")
        self.user_audio_pcm = extract_video_audio_pcm(video_path)

        rem = len(self.user_audio_pcm) % CHUNK_SAMPLES
        if rem > 0:
            self.user_audio_pcm = np.concatenate([self.user_audio_pcm, np.zeros(CHUNK_SAMPLES - rem, dtype=np.int16)])

        self.total_chunks = len(self.user_audio_pcm) // CHUNK_SAMPLES
        logging.info(f"📹 总 chunk 数: {self.total_chunks} ({self.total_chunks * CHUNK_MS / 1000:.1f}s)")
        self.final_mix_audio = bytearray()
        self.s2_agent = System2Agent()
        self.agent_audio_events = []
        self.sim_start_mono = None
        self.ws_ready = asyncio.Event()
        self.ws_error = None
        self.drop_agent_audio = False
        # 已见帧缓存：最多保留 50 帧（OpenAI API 单次请求上限），THINK 时全部传给 S2
        self._recent_frames_b64: list = []
        self._max_frames_for_s2 = 50

    async def _get_remote_active_session(self):
        try:
            resp = await asyncio.to_thread(requests.get, S1_HEALTH_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            active = data.get("active_session_id")
            if isinstance(active, str) and active.strip():
                return active.strip()
        except Exception as e:
            logging.warning(f"query orchestrator health failed: {e}")
        return None

    async def _prepare_remote_session(self, session_id: str):
        active_session_id = await self._get_remote_active_session()
        if active_session_id and active_session_id != session_id:
            logging.warning(
                "发现 orchestrator 残留旧 session，先清理：old=%s new=%s",
                active_session_id,
                session_id,
            )
            await self._clear_remote_session(active_session_id)
            await asyncio.sleep(0.2)

    async def _audio_ws_listener(self, session_id: str):
        ws_url = f"{S1_AUDIO_WS_BASE}/{session_id}"
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                async with websockets.connect(ws_url, max_size=None) as ws:
                    self.ws_error = None
                    self.ws_ready.set()
                    while True:
                        msg = await ws.recv()
                        if not isinstance(msg, (bytes, bytearray)):
                            continue
                        if self.drop_agent_audio:
                            logging.info("🔇 丢弃一段到达的 talker 音频（STOP 后抑制）")
                            continue
                        pcm_audio = wav_bytes_to_pcm(bytes(msg))
                        if not pcm_audio:
                            continue
                        pcm_audio = stretch_pcm_to_chunk(pcm_audio, target_ms=CHUNK_MS)
                        # 时间冻结：按顺序累加，不依赖墙钟时间
                        # 视频帧提取会拖慢循环，墙钟时间不可靠，改为顺序拼接
                        arrival_s = len(self.agent_audio_events) * (CHUNK_MS / 1000.0)
                        self.agent_audio_events.append((arrival_s, pcm_audio))
                        logging.info(
                            "🔊 收到 talker 音频片段 arrival_s=%.3f len_bytes=%d",
                            arrival_s,
                            len(pcm_audio),
                        )
            except websockets.exceptions.InvalidStatus as e:
                if getattr(e.response, "status_code", None) == 409 and attempt < max_attempts:
                    active_session_id = await self._get_remote_active_session()
                    logging.warning(
                        "WS 建连 409，第 %d/%d 次重试，active_session_id=%s target_session_id=%s",
                        attempt,
                        max_attempts,
                        active_session_id,
                        session_id,
                    )
                    if active_session_id and active_session_id != session_id:
                        await self._clear_remote_session(active_session_id)
                    await asyncio.sleep(0.5)
                    continue
                self.ws_error = e
                self.ws_ready.set()
                raise
            except Exception as e:
                self.ws_error = e
                self.ws_ready.set()
                raise
            return

    async def _clear_remote_session(self, session_id: str):
        try:
            await asyncio.to_thread(
                requests.delete,
                f"{S1_CLEAR_SESSION_BASE}/{session_id}",
                timeout=30,
            )
        except Exception as e:
            logging.warning(f"clear session failed: {e}")

    def _render_final_mix(self):
        user_arr = self.user_audio_pcm.astype(np.int32)
        max_len = len(user_arr)
        rendered_events = []
        prev_end_idx = None
        corrected_events = 0
        corrected_gap_samples = 0
        corrected_overlap_samples = 0
        for arrival_s, pcm_bytes in self.agent_audio_events:
            agent_arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.int32)
            start_idx = int(round(arrival_s * SAMPLE_RATE))
            if prev_end_idx is not None:
                drift_samples = start_idx - prev_end_idx
                # Keep the first talker packet anchored at its real arrival time,
                # then absorb small websocket scheduling jitter so chunks stay seamless.
                if abs(drift_samples) <= AUDIO_STITCH_JITTER_SAMPLES:
                    corrected_events += 1
                    if drift_samples > 0:
                        corrected_gap_samples += drift_samples
                    elif drift_samples < 0:
                        corrected_overlap_samples += -drift_samples
                    start_idx = prev_end_idx
            end_idx = start_idx + len(agent_arr)
            max_len = max(max_len, end_idx)
            rendered_events.append((start_idx, agent_arr))
            prev_end_idx = end_idx

        user_track = np.zeros(max_len, dtype=np.int32)
        user_track[:len(user_arr)] += user_arr
        agent_track = np.zeros(max_len, dtype=np.int32)
        for start_idx, agent_arr in rendered_events:
            agent_track[start_idx : start_idx + len(agent_arr)] += agent_arr
        if corrected_events:
            logging.info(
                "🔧 audio stitch corrected events=%d gap_ms=%.2f overlap_ms=%.2f",
                corrected_events,
                corrected_gap_samples * 1000.0 / SAMPLE_RATE,
                corrected_overlap_samples * 1000.0 / SAMPLE_RATE,
            )
        mixed = np.clip(user_track + agent_track, -32768, 32767).astype(np.int16)
        self.final_mix_audio = bytearray(mixed.tobytes())

    async def run(self):
        logging.info("🚀 Simulation Started (S1-S2, Qwen3-Omni 直接输出 text+audio，无 TTS)...")
        
        conversation_id = f"stimulate-talker-{int(time.time() * 1000)}"
        self.sim_start_mono = time.monotonic()
        self.ws_ready.clear()
        self.ws_error = None
        await self._prepare_remote_session(conversation_id)
        ws_task = asyncio.create_task(self._audio_ws_listener(conversation_id))
        await self.ws_ready.wait()
        if self.ws_error is not None:
            raise RuntimeError(f"audio websocket setup failed: {self.ws_error}") from self.ws_error
        try:
            # 从训练数据分布中随机采样风格和开场，保持与训练一致
            _styles = ["冷静型", "简洁", "耐心", "控场型", "可信赖型"]
            _openings = ["先接题后简短描述", "先一句判断再往下讲", "先轻回应再补充细节", "先短答不展开"]
            _style = random.choice(_styles)
            _opening = random.choice(_openings)
            _system = f"你是视频通话助手\n你的助手风格是：控场型。\n你的开场方式要求：先短答不展开，耐心听用户说完所有选项再回答。"
            s1_messages = [{"role": "system", "content": _system}]
            logging.info(f"📋 System: {_system}")
            sim_time = 0.0  # 仿真时钟（秒），每 chunk 推进 CHUNK_MS
            for i in range(self.total_chunks):
                chunk_start_wall = time.monotonic()

                # 帧提取等耗时操作在下方执行，执行完后补足到 CHUNK_MS
                start_idx = i * CHUNK_SAMPLES
                end_idx = (i + 1) * CHUNK_SAMPLES
                chunk_user_bytes = self.user_audio_pcm[start_idx:end_idx].tobytes()
                s2_cmd = self.s2_agent.get_new_commands(i)
                b64_user = pcm_to_b64_wav(chunk_user_bytes)

                has_frame = (i % VIDEO_FRAME_INTERVAL == 0)
                if has_frame:
                    frame_s1_jpg, frame_s2_jpg = extract_frame_at_chunk(self.video_path, i)
                    b64_frame = base64.b64encode(frame_s1_jpg).decode("utf-8") if frame_s1_jpg else ""
                    has_frame = bool(b64_frame)
                    if has_frame:
                        b64_frame_s2 = base64.b64encode(frame_s2_jpg).decode("utf-8") if frame_s2_jpg else b64_frame
                        self._recent_frames_b64.append(b64_frame_s2)

                if has_frame:
                    s1_content = [
                        {"type": "text", "text": "{'audio_input': '"},
                        {"type": "input_audio", "input_audio": {"data": b64_user, "format": "wav"}},
                        {"type": "text", "text": f"', 'from_s2': '{s2_cmd}', 'input_video': '"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}"}},
                        {"type": "text", "text": "'}"},
                    ]
                else:
                    s1_content = [
                        {"type": "text", "text": "{'audio_input': '"},
                        {"type": "input_audio", "input_audio": {"data": b64_user, "format": "wav"}},
                        {"type": "text", "text": f"', 'from_s2': '{s2_cmd}'}}"},
                    ]
                s1_messages.append({"role": "user", "content": s1_content})
                
                omni_start = time.time()
                s1_response_text = "{}"
                try:
                    completion = await self.s1_client.chat.completions.create(
                        model=S1_MODEL_NAME,
                        messages=s1_messages,
                        max_tokens=999,
                        temperature=0.01,
                        stream=False,
                        extra_body={"vllm_xargs": {"conversation_id": conversation_id}},
                        # modalities=["text", "audio"],  # 必须同时请求 text+audio，否则服务端会过滤掉 text（见 vLLM-Omni serving_chat 的 requested_modalities 逻辑）
                    )
                    # 服务端可能返回 choices[0]=text/choices[1]=audio，从有 content 的 choice 取文本
                    s1_response_text = "{}"
                    for ch in getattr(completion, "choices", []) or []:
                        cnt = getattr(ch.message, "content", None)
                        if cnt and isinstance(cnt, str) and cnt.strip():
                            s1_response_text = cnt.strip()
                            break
                    # 打印所有收到的 S1 返回文本
                    print(f"[Chunk {i}] S1 收到文本: {s1_response_text}")
                except Exception as e:
                    logging.error(f"Chunk {i} S1 Error: {e}")
                s1_messages.append({"role": "assistant", "content": s1_response_text})
                
                omni_duration = time.time() - omni_start
                await self._parse_and_execute_s1(s1_response_text, i)
                logging.info(f"[{i}] HTTP text 返回耗时 {omni_duration:.2f}s")

                # 时间冻结：确保每个 chunk 至少占用 CHUNK_MS 的仿真时间
                sim_time += CHUNK_MS / 1000.0
                elapsed = time.monotonic() - chunk_start_wall
                wait_s = max(0.0, (CHUNK_MS / 1000.0) - elapsed)
                if wait_s > 0:
                    await asyncio.sleep(wait_s)

            logging.info("⏳ Waiting trailing talker audio...")
            wait_deadline = time.monotonic() + 15.0
            expected_events = self.total_chunks
            while time.monotonic() < wait_deadline and len(self.agent_audio_events) < expected_events:
                await asyncio.sleep(0.05)

            self._render_final_mix()

            logging.info("💾 Saving Final Audio...")
            with wave.open(OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPWIDTH)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(self.final_mix_audio)
            logging.info(f"✅ Saved to {OUTPUT_FILENAME}")
        finally:
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.warning(f"audio websocket task ended with error: {e}")
            await self._clear_remote_session(conversation_id)

    async def _parse_and_execute_s1(self, text, chunk_idx):
        try:
            clean = text.replace("```json", "").replace("```", "").strip()
            if "}" in clean:
                clean = clean[:clean.rindex("}")+1]
            if "{" in clean:
                clean = clean[clean.index("{"):]
            data = {}
            try: data = json.loads(clean)
            except:
                try: data = ast.literal_eval(clean)
                except: pass
            
            asr = data.get("asr", "")
            tts = data.get("tts", "")
            tts_ctrl = data.get("tts_control", "")
            s2_ctrl = data.get("system2_control", "")
            
            if asr:
                print(f"[{chunk_idx}] 收到 ASR(用户): {asr}")
                logging.info(f"[{chunk_idx}] ASR: {asr}")
                self.s2_agent.add_context("User", asr)
            
            if "[STOP]" in tts_ctrl:
                logging.warning(f"[{chunk_idx}] Agent STOP: 后续到达的 talker 音频先抑制")
                self.drop_agent_audio = True
            if tts:
                print(f"[{chunk_idx}] 收到 Agent(回复): {tts}")
                logging.info(f"[{chunk_idx}] Agent 文本: {tts}")
                self.s2_agent.add_context("Agent", tts)
                self.drop_agent_audio = False
                
            if "[THINK]" in s2_ctrl:
                logging.info(f"[{chunk_idx}] 🧠 Triggering S2 THINK (frames={len(self._recent_frames_b64)})")
                await self.s2_agent.trigger_think(frames_b64=list(self._recent_frames_b64[-self._max_frames_for_s2:]))
            elif "[WAIT]" in s2_ctrl:
                logging.info(f"[{chunk_idx}] 🛑 Triggering S2 WAIT")
                await self.s2_agent.trigger_wait()
                
        except Exception as e:
            logging.error(f"Parse error at chunk {chunk_idx}: {e}")

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else TEST_INPUT_VIDEO
    if not os.path.exists(video_path):
        logging.error(f"视频文件不存在: {video_path}")
        sys.exit(1)
    sim = OfflineSimulator(video_path)
    asyncio.run(sim.run())
