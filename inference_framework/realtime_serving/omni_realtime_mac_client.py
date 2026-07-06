#!/usr/bin/env python3
"""
Omni 实时对话 —— Mac 本地客户端
与同一目录下的 omni_realtime_server.py 配套；协议、块长与 simulate_v8 一致（480ms / 24kHz / 单声道 PCM16）。

================================================================================
一、在 Mac 上准备环境
================================================================================

1) 建议 Python 3.10+，在终端进入本文件所在目录（或把路径写全）。

2) 安装 PortAudio 与 PyAudio（麦克风必用）::

    brew install portaudio
    pip install pyaudio websockets numpy

3) 可选：安装 pynput 后支持「按住空格开麦、松手闭麦」；不装则始终向服务器送音（闭麦时发静音）::

    pip install pynput

4) 首次运行若系统弹窗请求「麦克风」权限，请点「允许」。

================================================================================
二、先起后端（在模型/编排机或你放 omni_realtime_server.py 的机器上）
================================================================================

示例::

    python omni_realtime_server.py --host 0.0.0.0 --port 8765

确保本机或网络能访问该机的 8765 端口；S1/Omni 相关地址需与服务器上环境变量或脚本内默认一致。

================================================================================
三、在 Mac 上跑客户端：--server 到底写谁（最易错）
================================================================================

**omni_realtime_server 跑在哪台机器上，--server 就要能指到那台机器的 IP:端口。**

- **服务端和 Mac 是同一台电脑**：才用 ``ws://127.0.0.1:8765``（默认即如此）。

- **服务端在远端（例如 codelab、别的云主机）**：在 Mac 上 **禁止** 用 ``127.0.0.1`` 当
  服务端地址。``127.0.0.1`` 永远只表示 **你这台 Mac 自己**；远端监听的 8765 不会出现在
  Mac 的回环口上。这时请用 **从 Mac 能 ping/能访问到的** 那台服务器的地址，例如::

    python3 omni_realtime_mac_client.py --server ws://your-server-host:8765

  把 ``your-server-host`` 换成你实际要连的主机 IP 或域名；端口与远端
  ``omni_realtime_server.py --port`` 一致。

- **若报** ``did not receive a valid HTTP response``：常见是连到了「不是 WebSocket 的端口」
  或「本机无服务」。先确认上面这一点：远端起在 8765，Mac 就必须写 ``ws://远端IP:8765``，
  而不是 ``ws://127.0.0.1:8765``（除非你真做了端口映射到本机）。

- Shell 里不要写尖括号：用真实 IP，勿写 ``ws://<IP>:8765``（会触发重定向报错）。

（可选）若你 **自己** 用 SSH -L 把远端 8765 映到 Mac 的 8765，那时才在 Mac 上用
``ws://127.0.0.1:8765``；转发规则里远端那一端端口必须与 omni 一致。

================================================================================
四、调试与操作
================================================================================

1) 需要看 WebSocket 日志（``--server`` 与第三节一致：远端就写 ``ws://远端IP:端口``）::

    python3 omni_realtime_mac_client.py --server ws://your-server-host:8765 --debug

2) 操作说明

- 终端里会显示 ASR / 助手 TTS 文本 / S2 状态；耳机或扬声器播放服务器推回的音频。
- 若已安装 pynput：按住空格 = 开麦；松开 = 闭麦（减少音箱回灌）。未安装 pynput：一直开麦。
- 结束对话：Ctrl+C。

3) 常见问题

- 连不上 / ``did not receive a valid HTTP response``：多数是把服务端开在远端却用
  ``ws://127.0.0.1``；或端口/防火墙/安全组未放行，Mac 到不了对端 8765。
- 无声音：查 Mac 输出、远端推流；``ws://`` 与 ``wss://`` 不要混用。
- pyaudio：brew install portaudio；pip 与执行 ``python3`` 为同一环境。
"""

import asyncio
import argparse
import numpy as np
import threading
import queue
import time
import json
import sys
import logging
import shutil
import os
from collections import deque

# 配置 logging（禁用，改用终端 UI）
logging.basicConfig(
    level=logging.WARNING,  # 只显示警告以上级别
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

try:
    import pyaudio
except ImportError:
    print("错误: 请安装 pyaudio")
    print("Mac: brew install portaudio && pip install pyaudio")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("错误: 请安装 websockets")
    print("pip install websockets")
    sys.exit(1)

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# ================= 超参数配置区域 =================
# 输入音量微调 (1.0 = 不变, 1.2 = 略增, 0.8 = 略减)；主增益由 AGC 自动平衡
INPUT_VOLUME_GAIN = 1.0

# 响度自动平衡 (AGC)
AGC_TARGET_LEVEL = 13000    # 目标峰值 (int16)，正常说话约 8000~15000
AGC_MIN_LEVEL = 150         # 低于此视为静音，不参与 AGC 计算
AGC_MIN_GAIN = 0.5          # 自动增益下限
AGC_MAX_GAIN = 18.0          # 自动增益上限
AGC_SMOOTHING = 0.92        # 增益平滑系数，越大越平滑

# 是否启用实时耳返 (True = 开启, False = 关闭)
ENABLE_SIDETONE = False

# 耳返音量 (相对于输入音量的倍数, 1.0 = 和输入一样, 0.5 = 一半)
SIDETONE_VOLUME = 1.0

# ================= 音频参数（与 omni_realtime_server 一致，480ms）=================
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_MS = 480
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
BYTES_PER_SAMPLE = 2
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE

# 录音参数 (更小的块以减少延迟)
RECORD_CHUNK = 16  # 录音块大小
FORMAT = pyaudio.paInt16

# 静音检测阈值（与服务器一致）
SILENCE_THRESHOLD = 500

# 闭麦时发送的静音块（避免助手声音回传）
SILENCE_CHUNK = b"\x00" * CHUNK_BYTES

# ================= 音频处理函数 =================
def apply_volume_gain(pcm_bytes: bytes, gain: float) -> bytes:
    """应用音量增益（用于微调）"""
    if not pcm_bytes or len(pcm_bytes) < 2 or gain == 1.0:
        return pcm_bytes
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    audio = audio * gain
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return audio.tobytes()


class AutoGainController:
    """
    响度自动平衡：将输入提升/衰减到目标水平
    输入太小时自动放大，INPUT_VOLUME_GAIN 仅做微调
    """
    def __init__(self):
        self._level_ema = float(AGC_TARGET_LEVEL)  # 当前响度指数移动平均
        self._current_gain = 1.0
        self._lock = threading.Lock()
    
    def process(self, pcm_bytes: bytes) -> bytes:
        if not pcm_bytes or len(pcm_bytes) < 2:
            return pcm_bytes
        level = get_audio_level(pcm_bytes)
        
        with self._lock:
            # 非静音时更新响度估计
            if level > AGC_MIN_LEVEL:
                self._level_ema = AGC_SMOOTHING * self._level_ema + (1 - AGC_SMOOTHING) * level
                eff_level = max(self._level_ema, AGC_MIN_LEVEL)
                target_gain = AGC_TARGET_LEVEL / eff_level
                target_gain = max(AGC_MIN_GAIN, min(AGC_MAX_GAIN, target_gain))
                self._current_gain = 0.85 * self._current_gain + 0.15 * target_gain
            gain = self._current_gain * INPUT_VOLUME_GAIN
        
        return apply_volume_gain(pcm_bytes, gain)

def apply_noise_gate(pcm_bytes: bytes, threshold: int = SILENCE_THRESHOLD) -> bytes:
    """对音频应用噪声门限：响度 ≤ threshold 的样本归零"""
    if not pcm_bytes or len(pcm_bytes) < 2:
        return pcm_bytes
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).copy()
    mask = np.abs(audio) <= threshold
    audio[mask] = 0
    return audio.tobytes()

def get_audio_level(pcm_bytes: bytes) -> int:
    """获取音频最大响度"""
    if not pcm_bytes or len(pcm_bytes) < 2:
        return 0
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    return int(np.max(np.abs(audio)))

def is_silent(pcm_bytes: bytes, threshold: int = SILENCE_THRESHOLD) -> bool:
    """判断音频是否为静音"""
    return get_audio_level(pcm_bytes) <= threshold


# ================= 终端 UI 管理器 =================
class TerminalUI:
    """
    固定位置刷新的终端 UI
    三条信息流：ASR、TTS、S2
    支持逐字更新，不滚动
    """
    
    # ANSI 转义序列
    CLEAR_SCREEN = "\033[2J"
    CURSOR_HOME = "\033[H"
    CURSOR_HIDE = "\033[?25l"
    CURSOR_SHOW = "\033[?25h"
    CLEAR_LINE = "\033[2K"
    
    # 颜色
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # 前景色
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # 背景色
    BG_BLACK = "\033[40m"
    BG_BLUE = "\033[44m"
    
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        
        # 获取终端尺寸
        try:
            size = shutil.get_terminal_size()
            self.width = size.columns
            self.height = size.lines
        except:
            self.width = 80
            self.height = 24
        
        # 三条信息流的内容
        self.asr_text = ""      # 用户 ASR
        self.tts_text = ""      # 助手 TTS
        self.s2_text = ""       # S2 思考
        
        # 状态信息
        self.audio_level = 0
        self.is_speaking = False
        self.elapsed_time = 0.0
        self.send_count = 0
        self.recv_count = 0
        self.connected = False
        
        # 内容区域宽度（留边框和图标的空间）
        self.content_width = min(self.width - 4, 76)
        
    def start(self):
        """启动 UI"""
        self.running = True
        # 隐藏光标，清屏
        sys.stdout.write(self.CURSOR_HIDE)
        sys.stdout.write(self.CLEAR_SCREEN)
        sys.stdout.write(self.CURSOR_HOME)
        sys.stdout.flush()
        self._draw_frame()
        
    def stop(self):
        """停止 UI"""
        self.running = False
        # 显示光标，移动到底部
        sys.stdout.write(f"\033[{self.height};1H")
        sys.stdout.write(self.CURSOR_SHOW)
        sys.stdout.write("\n")
        sys.stdout.flush()
    
    def _move_cursor(self, row: int, col: int = 1):
        """移动光标到指定位置（1-based）"""
        return f"\033[{row};{col}H"
    
    def _draw_box_line(self, char_left: str, char_mid: str, char_right: str):
        """绘制边框行"""
        return char_left + char_mid * (self.content_width + 2) + char_right
    
    def _pad_text(self, text: str, width: int) -> str:
        """填充文本到指定宽度（考虑中文字符）"""
        # 计算实际显示宽度（中文字符占2个宽度）
        display_len = 0
        for ch in text:
            if ord(ch) > 127:
                display_len += 2
            else:
                display_len += 1
        
        if display_len >= width:
            # 截断
            result = ""
            curr_len = 0
            for ch in text:
                ch_width = 2 if ord(ch) > 127 else 1
                if curr_len + ch_width > width - 3:
                    result += "..."
                    break
                result += ch
                curr_len += ch_width
            return result
        else:
            # 填充空格
            padding = width - display_len
            return text + " " * padding
    
    def _wrap_text(self, text: str, width: int) -> list:
        """将文本换行成多行"""
        if not text:
            return [""]
        
        lines = []
        current_line = ""
        current_width = 0
        
        for ch in text:
            ch_width = 2 if ord(ch) > 127 else 1
            if current_width + ch_width > width:
                lines.append(current_line)
                current_line = ch
                current_width = ch_width
            else:
                current_line += ch
                current_width += ch_width
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [""]
    
    def _draw_frame(self):
        """绘制初始边框"""
        w = self.content_width
        
        output = []
        output.append(self._move_cursor(1))
        
        # 标题
        output.append(f"{self.BOLD}{self.CYAN}╔{'═' * (w + 2)}╗{self.RESET}")
        output.append(self._move_cursor(2))
        title = "🎙️  Omni 实时 (480ms) — 语音对话"
        output.append(f"{self.BOLD}{self.CYAN}║{self.RESET} {self._pad_text(title, w)} {self.CYAN}║{self.RESET}")
        
        # 状态栏
        output.append(self._move_cursor(3))
        output.append(f"{self.CYAN}╠{'═' * (w + 2)}╣{self.RESET}")
        output.append(self._move_cursor(4))
        output.append(f"{self.CYAN}║{self.RESET} {self._pad_text('📊 状态: 连接中...', w)} {self.CYAN}║{self.RESET}")
        
        # ASR 区域 (标题行6, 内容行7-10 共4行)
        output.append(self._move_cursor(5))
        output.append(f"{self.CYAN}╠{'═' * (w + 2)}╣{self.RESET}")
        output.append(self._move_cursor(6))
        output.append(f"{self.CYAN}║{self.RESET} {self.BOLD}{self.GREEN}👤 ASR (用户):{self.RESET}{' ' * (w - 14)} {self.CYAN}║{self.RESET}")
        for r in range(7, 11):
            output.append(self._move_cursor(r))
            output.append(f"{self.CYAN}║{self.RESET} {self._pad_text('', w)} {self.CYAN}║{self.RESET}")
        
        # TTS 区域 (标题行11, 内容行12-16 共5行)
        output.append(self._move_cursor(11))
        output.append(f"{self.CYAN}╠{'═' * (w + 2)}╣{self.RESET}")
        output.append(self._move_cursor(12))
        output.append(f"{self.CYAN}║{self.RESET} {self.BOLD}{self.YELLOW}🤖 TTS (助手):{self.RESET}{' ' * (w - 14)} {self.CYAN}║{self.RESET}")
        for r in range(13, 18):
            output.append(self._move_cursor(r))
            output.append(f"{self.CYAN}║{self.RESET} {self._pad_text('', w)} {self.CYAN}║{self.RESET}")
        
        # S2 区域 (标题行18, 内容行19-22 共4行)
        output.append(self._move_cursor(18))
        output.append(f"{self.CYAN}╠{'═' * (w + 2)}╣{self.RESET}")
        output.append(self._move_cursor(19))
        output.append(f"{self.CYAN}║{self.RESET} {self.BOLD}{self.MAGENTA}🧠 S2 (思考):{self.RESET}{' ' * (w - 13)} {self.CYAN}║{self.RESET}")
        for r in range(20, 24):
            output.append(self._move_cursor(r))
            output.append(f"{self.CYAN}║{self.RESET} {self._pad_text('', w)} {self.CYAN}║{self.RESET}")
        
        # 底部
        output.append(self._move_cursor(24))
        output.append(f"{self.CYAN}╚{'═' * (w + 2)}╝{self.RESET}")
        
        # 提示
        output.append(self._move_cursor(25))
        output.append(f"{self.DIM}按 Ctrl+C 停止{self.RESET}")
        
        sys.stdout.write("".join(output))
        sys.stdout.flush()
    
    def update_status(self, audio_level: int, is_speaking: bool, elapsed: float, 
                      send_count: int, recv_count: int, connected: bool,
                      ptt_active: bool = True):
        """更新状态栏。ptt_active: 是否正在按住空格开麦。"""
        with self.lock:
            self.audio_level = audio_level
            self.is_speaking = is_speaking
            self.elapsed_time = elapsed
            self.send_count = send_count
            self.recv_count = recv_count
            self.connected = connected
            
            if not self.running:
                return
            
            w = self.content_width
            
            # 生成音量条
            bar_len = min(audio_level // 500, 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            
            # 开麦/闭麦 与 状态文字
            if not connected:
                status = f"❌ 断开 | [{bar}]"
            elif not ptt_active:
                status = f"🔇 闭麦 (按住空格开麦) | [{bar}] | ⬆{send_count} ⬇{recv_count} | {elapsed:.1f}s"
            elif is_speaking:
                status = f"🎤 开麦·说话 | [{bar}] | ⬆{send_count} ⬇{recv_count} | {elapsed:.1f}s"
            else:
                status = f"🎤 开麦 | [{bar}] | ⬆{send_count} ⬇{recv_count} | {elapsed:.1f}s"
            
            output = self._move_cursor(4)
            output += f"{self.CYAN}║{self.RESET} {self._pad_text(status, w)} {self.CYAN}║{self.RESET}"
            
            sys.stdout.write(output)
            sys.stdout.flush()
    
    def update_asr(self, text: str):
        """更新 ASR 文本（逐字追加或替换）"""
        with self.lock:
            self.asr_text = text
            if not self.running:
                return
            self._redraw_section("asr")
    
    def update_tts(self, text: str):
        """更新 TTS 文本"""
        with self.lock:
            self.tts_text = text
            if not self.running:
                return
            self._redraw_section("tts")
    
    def update_s2(self, text: str):
        """更新 S2 文本"""
        with self.lock:
            self.s2_text = text
            if not self.running:
                return
            self._redraw_section("s2")
    
    def append_asr(self, char: str):
        """追加 ASR 字符（逐字蹦出）"""
        with self.lock:
            self.asr_text += char
            if not self.running:
                return
            self._redraw_section("asr")
    
    def append_tts(self, char: str):
        """追加 TTS 字符（逐字蹦出）"""
        with self.lock:
            self.tts_text += char
            if not self.running:
                return
            self._redraw_section("tts")
    
    def append_s2(self, char: str):
        """追加 S2 字符"""
        with self.lock:
            self.s2_text += char
            if not self.running:
                return
            self._redraw_section("s2")
    
    def clear_asr(self):
        """清空 ASR"""
        self.update_asr("")
    
    def clear_tts(self):
        """清空 TTS"""
        self.update_tts("")
    
    def clear_s2(self):
        """清空 S2"""
        self.update_s2("")
    
    def _redraw_section(self, section: str):
        """重绘指定区域"""
        w = self.content_width
        
        if section == "asr":
            lines = self._wrap_text(self.asr_text, w)
            start_row = 7
            max_lines = 4
            color = self.GREEN
        elif section == "tts":
            lines = self._wrap_text(self.tts_text, w)
            start_row = 13
            max_lines = 5
            color = self.YELLOW
        else:  # s2
            lines = self._wrap_text(self.s2_text, w)
            start_row = 20
            max_lines = 4
            color = self.MAGENTA
        
        # 累积模式：只显示最后几行（最新内容在下方）
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        
        # 填充到固定行数
        while len(lines) < max_lines:
            lines.append("")
        
        output = []
        for i, line in enumerate(lines):
            output.append(self._move_cursor(start_row + i))
            padded = self._pad_text(line, w)
            output.append(f"{self.CYAN}║{self.RESET} {color}{padded}{self.RESET} {self.CYAN}║{self.RESET}")
        
        sys.stdout.write("".join(output))
        sys.stdout.flush()

# ================= 实时耳返播放器 =================
class SidetonePlayer:
    """
    独立于主循环的实时耳返播放器
    直接在录音回调中播放，延迟极低
    """
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.enabled = ENABLE_SIDETONE
        self.volume = SIDETONE_VOLUME
        
    def start(self):
        """启动耳返输出流"""
        if not self.enabled:
            return
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=RECORD_CHUNK  # 小块以减少延迟
        )
        logging.info(f"👂 耳返已启动 (音量: {self.volume:.1f}x)")
        
    def play(self, pcm_bytes: bytes):
        """实时播放音频（在录音回调中调用）"""
        if not self.enabled or not self.stream:
            return
        try:
            # 应用耳返音量
            if self.volume != 1.0:
                audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                audio = audio * self.volume
                audio = np.clip(audio, -32768, 32767).astype(np.int16)
                pcm_bytes = audio.tobytes()
            self.stream.write(pcm_bytes)
        except Exception:
            pass  # 忽略播放错误，不影响录音
            
    def stop(self):
        """停止耳返"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.enabled:
            logging.info("👂 耳返已停止")
            
    def close(self):
        self.p.terminate()

# ================= 音频录制器 =================
class AudioRecorder:
    def __init__(self, sidetone: SidetonePlayer = None):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.chunk_queue = queue.Queue()
        self.sidetone = sidetone
        self.agc = AutoGainController()  # 响度自动平衡
        
    def start(self):
        """开始录音"""
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=RECORD_CHUNK,
            stream_callback=self._audio_callback
        )
        self.recording = True
        self.stream.start_stream()
        logging.info(f"🎤 麦克风已启动 (AGC目标: {AGC_TARGET_LEVEL}, 微调: {INPUT_VOLUME_GAIN:.1f}x)")
        
    def stop(self):
        """停止录音"""
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        logging.info("🎤 麦克风已停止")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """录音回调 - 响度自动平衡 + 微调"""
        if self.recording:
            processed_data = self.agc.process(in_data)
            
            # 实时耳返（独立于主循环，极低延迟）
            if self.sidetone:
                self.sidetone.play(processed_data)
            
            with self.lock:
                self.buffer.extend(processed_data)
                # 当积累够一个chunk时，放入队列
                while len(self.buffer) >= CHUNK_BYTES:
                    chunk = bytes(self.buffer[:CHUNK_BYTES])
                    del self.buffer[:CHUNK_BYTES]
                    # 应用噪声门限
                    chunk = apply_noise_gate(chunk)
                    self.chunk_queue.put(chunk)
        return (None, pyaudio.paContinue)
    
    def get_chunk(self, timeout=0.6) -> bytes:
        """获取一个音频块"""
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def close(self):
        self.p.terminate()

# ================= 音频播放器 =================
class AudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.playing = False
        self.buffer = deque()
        self.lock = threading.Lock()
        self.play_thread = None
        
        # 统计信息
        self.total_played_bytes = 0
        self.total_played_chunks = 0
        self.last_non_silent_time = 0
        
    def start(self):
        """开始播放"""
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=RECORD_CHUNK
        )
        self.playing = True
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()
        logging.info("🔊 扬声器已启动")
        
    def stop(self):
        """停止播放"""
        self.playing = False
        if self.play_thread:
            self.play_thread.join(timeout=1)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        logging.info("🔊 扬声器已停止")
    
    def _play_loop(self):
        """播放循环"""
        while self.playing:
            with self.lock:
                if self.buffer:
                    data = self.buffer.popleft()
                else:
                    data = None
            
            if data:
                try:
                    self.stream.write(data)
                    self.total_played_bytes += len(data)
                except Exception as e:
                    logging.error(f"播放错误: {e}")
            else:
                time.sleep(0.01)
    
    def add_audio(self, audio_bytes: bytes):
        """添加音频数据到播放队列"""
        with self.lock:
            self.buffer.append(audio_bytes)
            self.total_played_chunks += 1
            
            level = get_audio_level(audio_bytes)
            if level > SILENCE_THRESHOLD:
                self.last_non_silent_time = time.time()
    
    def clear(self):
        """清空播放队列"""
        with self.lock:
            self.buffer.clear()
    
    def get_queue_size(self) -> int:
        """获取队列中的数据量"""
        with self.lock:
            return sum(len(chunk) for chunk in self.buffer)
    
    def close(self):
        self.p.terminate()

# ================= WebSocket 客户端 =================
class VoiceClient:
    def __init__(self, server_url: str, debug: bool = False):
        self.server_url = server_url
        
        # 耳返播放器（独立于主循环）
        self.sidetone = SidetonePlayer()
        
        # 录音器（传入耳返播放器）
        self.recorder = AudioRecorder(sidetone=self.sidetone)
        
        # 主播放器（播放服务器返回的音频）
        self.player = AudioPlayer()
        
        # 终端 UI
        self.ui = TerminalUI()
        
        self.websocket = None
        self.running = False
        self.connected = False
        self.debug = debug
        
        # 统计信息
        self.send_count = 0
        self.recv_count = 0
        self.recv_audio_count = 0
        self.recv_non_silent_count = 0
        self.start_time = None
        
        # 累积文本状态（ASR/TTS 为整段累积，S2 分已提交 + 当前流）
        self.current_asr = ""
        self.current_tts = ""
        self.committed_s2 = ""      # 已结束的 S2 内容
        self.current_s2_stream = "" # 当前 S2 流式内容
        
        # 按住空格开麦（Push-to-Talk）
        self._push_to_talk_active = False
        self._ptt_lock = threading.Lock()
        self._keyboard_listener = None
        
    def _get_ptt_active(self) -> bool:
        with self._ptt_lock:
            return self._push_to_talk_active
    
    def _set_ptt_active(self, active: bool):
        with self._ptt_lock:
            self._push_to_talk_active = active
    
    def _start_keyboard_listener(self):
        """启动空格键监听：按住开麦，松开闭麦"""
        if not PYNPUT_AVAILABLE:
            return
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self._set_ptt_active(True)
            except Exception:
                pass
        def on_release(key):
            try:
                if key == keyboard.Key.space:
                    self._set_ptt_active(False)
            except Exception:
                pass
        self._keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._keyboard_listener.start()
    
    def _stop_keyboard_listener(self):
        if self._keyboard_listener:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            self._keyboard_listener = None
        
    async def connect(self):
        """连接服务器"""
        logging.info(f"🔗 正在连接服务器: {self.server_url}")
        try:
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=60,
                max_size=10 * 1024 * 1024  # 10MB
            )
            self.connected = True
            logging.info("✅ 连接成功!")
            return True
        except Exception as e:
            logging.error(f"❌ 连接失败: {e}")
            return False
    
    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
    
    async def send_audio_loop(self):
        """发送音频数据的循环。按住空格开麦，松开发送静音。"""
        loop = asyncio.get_event_loop()
        while self.running and self.connected:
            try:
                # 在线程池中运行阻塞的 get_chunk，避免阻塞事件循环
                chunk = await loop.run_in_executor(None, self.recorder.get_chunk, 0.1)
                if chunk:
                    ptt = self._get_ptt_active()
                    if ptt:
                        to_send = chunk
                        level = get_audio_level(chunk)
                    else:
                        to_send = SILENCE_CHUNK
                        level = 0
                    
                    await self.websocket.send(to_send)
                    self.send_count += 1
                    
                    # 更新 UI 状态栏（开麦/闭麦、响度）
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    is_speaking = ptt and level > SILENCE_THRESHOLD
                    self.ui.update_status(
                        audio_level=level,
                        is_speaking=is_speaking,
                        elapsed=elapsed,
                        send_count=self.send_count,
                        recv_count=self.recv_audio_count,
                        connected=self.connected,
                        ptt_active=ptt,
                    )
                    
            except websockets.exceptions.ConnectionClosed:
                self.connected = False
                break
            except Exception as e:
                if self.running:
                    pass  # 静默处理
                await asyncio.sleep(0.1)
    
    async def receive_loop(self):
        """接收服务器数据的循环 - 使用和 test_client 一样的异步迭代器模式"""
        try:
            async for message in self.websocket:
                if not self.running:
                    break
                    
                self.recv_count += 1
                
                if isinstance(message, bytes):
                    # 音频数据 - 播放
                    self.recv_audio_count += 1
                    self.player.add_audio(message)
                    
                else:
                    # JSON消息
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        
                        if msg_type == "asr":
                            # ASR 识别结果 - 累积并逐字显示
                            text = data.get('text', '')
                            await self._append_and_animate("asr", text)
                            
                        elif msg_type == "tts":
                            # TTS 文本 - 累积并逐字显示
                            text = data.get('text', '')
                            await self._append_and_animate("tts", text)
                            
                        elif msg_type == "s2":
                            # S2 思考内容（流式，更新当前条）
                            text = data.get('text', '')
                            self._update_s2_stream(text)
                            
                        elif msg_type == "s2_status":
                            # S2 状态 - 累积到 S2 区域
                            status = data.get('status', '')
                            self._append_s2_status(status)
                            
                        elif msg_type == "pong":
                            pass  # 心跳响应
                            
                    except json.JSONDecodeError:
                        pass
                            
        except websockets.exceptions.ConnectionClosed:
            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            pass
        finally:
            self.connected = False
    
    async def _append_and_animate(self, section: str, new_utterance: str):
        """累积：将新一句追加到对应区域，并逐字动画显示新内容"""
        if not new_utterance.strip():
            return
        sep = "\n"
        if section == "asr":
            prev = self.current_asr
            self.current_asr = (self.current_asr + sep + new_utterance) if prev else new_utterance
            full = self.current_asr
            # 只对新增部分逐字蹦
            new_part = new_utterance
            for char in new_part:
                self.ui.append_asr(char)
                await asyncio.sleep(0.02)
        else:  # tts
            prev = self.current_tts
            self.current_tts = (self.current_tts + sep + new_utterance) if prev else new_utterance
            new_part = new_utterance
            for char in new_part:
                self.ui.append_tts(char)
                await asyncio.sleep(0.02)
    
    def _update_s2_stream(self, text: str):
        """更新当前 S2 流式内容（不累积，只更新当前这条）"""
        self.current_s2_stream = text
        full = (self.committed_s2 + "\n" + text) if self.committed_s2 else text
        self.ui.update_s2(full)
    
    def _append_s2_status(self, status: str):
        """S2 状态更新：把当前流提交到累积，再追加状态行"""
        if self.current_s2_stream:
            self.committed_s2 = (self.committed_s2 + "\n" + self.current_s2_stream) if self.committed_s2 else self.current_s2_stream
            self.current_s2_stream = ""
        self.committed_s2 = (self.committed_s2 + "\n[" + status + "]") if self.committed_s2 else ("[" + status + "]")
        self.ui.update_s2(self.committed_s2)
    
    async def heartbeat_loop(self):
        """心跳保活"""
        while self.running and self.connected:
            try:
                await self.websocket.send(json.dumps({"type": "ping"}))
                if self.debug:
                    logging.debug("💓 发送心跳")
                await asyncio.sleep(10)
            except:
                break
    
    async def run(self):
        """运行客户端"""
        # 启动 UI
        self.ui.start()
        
        if not await self.connect():
            self.ui.stop()
            print("❌ 连接失败")
            return
        
        self.running = True
        self.start_time = time.time()
        
        # 按住空格开麦：启动键盘监听（未安装 pynput 则默认一直开麦）
        if PYNPUT_AVAILABLE:
            self._start_keyboard_listener()
        else:
            self._set_ptt_active(True)  # 无 pynput 时保持原行为：一直开麦
        
        # 启动音频设备（耳返必须在录音器之前启动）
        self.sidetone.start()
        self.recorder.start()
        self.player.start()
        
        try:
            # 并发运行发送、接收、心跳
            tasks = [
                asyncio.create_task(self.send_audio_loop()),
                asyncio.create_task(self.receive_loop()),
                asyncio.create_task(self.heartbeat_loop()),
            ]
            
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            self._stop_keyboard_listener()
            self.ui.stop()
            self.recorder.stop()
            self.sidetone.stop()
            self.player.stop()
            await self.disconnect()
            self.recorder.close()
            self.sidetone.close()
            self.player.close()
            
            # 打印最终统计
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"\n📊 最终统计:")
            print(f"   运行时间: {elapsed:.1f}s")
            print(f"   发送: {self.send_count} 块")
            print(f"   接收: {self.recv_audio_count} 块")
            print(f"   播放: {self.player.total_played_bytes / 1024:.1f} KB")
            print("\n👋 再见!")

# ================= 主函数 =================
async def main():
    parser = argparse.ArgumentParser(description="Omni 实时 Mac Client (480ms 全双工)")
    parser.add_argument(
        "--server", 
        type=str, 
        default="ws://127.0.0.1:8765",
        help="omni_realtime_server WebSocket 地址，如 ws://<ip>:8765"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用详细 debug 日志"
    )
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    client = VoiceClient(args.server, debug=args.debug)
    
    try:
        await client.run()
    except KeyboardInterrupt:
        client.running = False

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已退出")
