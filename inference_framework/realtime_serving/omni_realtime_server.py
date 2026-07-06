#!/usr/bin/env python3
"""
Qwen3-Omni 实时对话后端（仅新增文件，不修改 simulate_v8 等现有脚本）

- 与 simulate_v8 一致：CHUNK_MS=480、HTTP 调 S1 + extra_body conversation_id、
  talker 音频经 WebSocket 拉取、stretch 后推给客户端播放。
- 与 online_server 风格一致：全双工 WebSocket，接收麦克风 PCM、下发播放 PCM、
  JSON 推送 asr/tts/s2 供 Mac 端展示。

启动（路径按你机子上实际路径）::

    python3 omni_realtime_server.py --host 0.0.0.0 --port 8765

若报 ``address already in use``，说明 8765 已被占用，换端口即可::

    ... omni_realtime_server.py --host 0.0.0.0 --port 9876

或查占用: ``fuser 8765/tcp`` / ``lsof -i:8765`` 后结束旧进程再启。
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import base64
import io
import json
import logging
import os
import random
import re
import time
import wave
from collections import deque
from typing import Any, Optional

import numpy as np
import requests
import websockets
from openai import AsyncOpenAI

# ================= 默认配置（可用环境变量覆盖）=================
S1_OMNI_BASE_URL = os.environ.get(
    "S1_OMNI_BASE_URL", "http://127.0.0.1:21000/v1"
)
S1_AUDIO_WS_BASE = os.environ.get(
    "S1_AUDIO_WS_BASE", "ws://127.0.0.1:21000/v1/audio/stream"
)
S1_CLEAR_SESSION_BASE = os.environ.get(
    "S1_CLEAR_SESSION_BASE", "http://127.0.0.1:21000/v1/session"
)
S1_HEALTH_URL = os.environ.get("S1_HEALTH_URL", "http://127.0.0.1:21000/health")
S1_MODEL_NAME = os.environ.get(
    "S1_MODEL_NAME",
    "models/qwen3-omni-checkpoint",
)
S1_SYSTEM_PROMPT = os.environ.get(
    "S1_SYSTEM_PROMPT", "你是有用的助手\n你的助手风格是：耐心。\n你的开场方式要求：注重同理。"
)

S2_THINK_BASE_URL = os.environ.get(
    "S2_THINK_BASE_URL", "http://localhost:8000/v1"
)
S2_MODEL_NAME = os.environ.get("S2_MODEL_NAME", "gpt-4.1")
S2_API_KEY = os.environ.get("S2_API_KEY", "EMPTY")

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPWIDTH = 2
CHUNK_MS = 480
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
BYTES_PER_SAMPLE = 2
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE

S2_CMD_COOLDOWN_MIN_S = 0.5
S2_CMD_COOLDOWN_MAX_S = 1.0

# 与前端 Mac 端噪声门限一致时可微调
SILENCE_THRESHOLD = 500
MAX_S1_TURNS = int(os.environ.get("MAX_S1_TURNS", "150"))
MAX_PLAYBACK_BUFFER_BYTES = int(os.environ.get("MAX_PLAYBACK_BUFFER_BYTES", "4800000"))

S2_SYSTEM_PROMPT = os.environ.get(
    "S2_SYSTEM_PROMPT",
    """你是指导 System 1 说话的 System 2。

你的任务是直接给 System 1 可外说的话术，帮助它自然地回应用户的问题和需求。

输出必须全部口语化，短句、顺口、好读。
你每次输出都用【】包起来，且【】里的内容必须是非空、有实际信息、能直接念给用户听的话。
每条【】只说一件事，控制在15字以内，宁可多输出几条短的，不要一条里塞很多内容。
禁止输出空括号，如【】、【   】。
禁止同义重复：同一个意思只说一次。
不要使用【答案是：】这类总结前缀，直接给最终可说话术即可。""",
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)


# ================= 工具函数（与 simulate_v8 行为对齐）=================


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
                padding = b"\x00" * (num_bytes - len(self.buffer))
                self.buffer.clear()
                return bytes(result) + padding
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

    async def bytes_len(self) -> int:
        async with self.lock:
            return len(self.buffer)

    async def drop_oldest_if_over(self, max_bytes: int, target_after: int):
        """若缓冲超过 max_bytes，丢弃最旧部分直至不超过 target_after。"""
        async with self.lock:
            n = len(self.buffer)
            if n <= max_bytes:
                return
            drop = n - target_after
            if drop > 0:
                del self.buffer[:drop]
                logging.warning(
                    "playback 缓冲过大，丢弃最旧 %d bytes（余 %d）", drop, len(self.buffer)
                )


def pcm_to_b64_wav(pcm_bytes: bytes) -> str:
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPWIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        return base64.b64encode(wav_io.getvalue()).decode("utf-8")


def wav_bytes_to_pcm(wav_bytes: bytes) -> bytes:
    if not wav_bytes:
        return b""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes())


def stretch_pcm_to_chunk(pcm_bytes: bytes, target_ms: int = CHUNK_MS) -> bytes:
    if not pcm_bytes:
        return b"\x00" * (int(SAMPLE_RATE * target_ms / 1000) * BYTES_PER_SAMPLE)
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    current_samples = len(arr)
    target_samples = int(SAMPLE_RATE * target_ms / 1000)
    if current_samples >= target_samples:
        return arr[:target_samples].tobytes()
    x_old = np.linspace(0, 1, current_samples, dtype=np.float64)
    x_new = np.linspace(0, 1, target_samples, dtype=np.float64)
    interpolated = np.interp(x_new, x_old, arr.astype(np.float64))
    return np.clip(interpolated, -32768, 32767).astype(np.int16).tobytes()


def apply_noise_gate(pcm_bytes: bytes, threshold: int = SILENCE_THRESHOLD) -> bytes:
    if not pcm_bytes or len(pcm_bytes) < 2:
        return pcm_bytes
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).copy()
    mask = np.abs(audio) <= threshold
    audio[mask] = 0
    return audio.tobytes()


# ================= S2（与 simulate_v8 同构）=================


class System2Agent:
    def __init__(self, on_s2_text: Optional[Any] = None):
        self.client = AsyncOpenAI(base_url=S2_THINK_BASE_URL, api_key=S2_API_KEY)
        self.history = [{"role": "system", "content": S2_SYSTEM_PROMPT}]
        self.context_buffer = []
        self.output_commands = deque()
        self.is_thinking = False
        self.should_stop = False
        self.thinking_task: Optional[asyncio.Task] = None
        self._next_cmd_allowed_chunk: float = 0.0
        self.lock = asyncio.Lock()
        self.on_s2_text = on_s2_text

    def add_context(self, source, text):
        if not text:
            return
        self.context_buffer.append(f"[{source}]: {text}")
        logging.info(f"S2 上下文: [{source}] {text[:80]}...")

    def _consolidate_context(self):
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
        full_response = ""
        last_match_end = 0
        pattern = re.compile(r"【(.*?)】")
        try:
            stream = await self.client.chat.completions.create(
                model=S2_MODEL_NAME,
                messages=self.history,
                stream=True,
                temperature=1.0,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
                max_tokens=2048,
            )
            async for chunk in stream:
                if self.should_stop:
                    break
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_response += content
                    if self.on_s2_text:
                        try:
                            await self.on_s2_text({"type": "s2", "text": full_response})
                        except Exception:
                            pass
                    for match in pattern.finditer(full_response, last_match_end):
                        command_content = match.group(1)
                        self.output_commands.append(f"【{command_content}】")
                        last_match_end = match.end()
            if full_response:
                self.history.append({"role": "assistant", "content": full_response})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.error(f"S2 思考错误: {e}", exc_info=True)
        finally:
            # 不在此持锁，避免与 trigger_think 中 await 旧任务时死锁
            self.is_thinking = False
            self.should_stop = False

    async def trigger_think(self):
        async with self.lock:
            new_context = self._consolidate_context()
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
            if new_context:
                self.history.append({"role": "user", "content": new_context})
            self.should_stop = False
            self.is_thinking = True
            self.thinking_task = asyncio.create_task(self._continuous_thinking())

    async def trigger_wait(self):
        async with self.lock:
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

    def get_new_commands(self, sim_chunk_idx: int) -> str:
        if not self.output_commands:
            return ""
        if float(sim_chunk_idx) + 1e-9 < self._next_cmd_allowed_chunk:
            return ""
        cmd = self.output_commands.popleft()
        gap_s = random.uniform(S2_CMD_COOLDOWN_MIN_S, S2_CMD_COOLDOWN_MAX_S)
        sec_per_chunk = CHUNK_MS / 1000.0
        gap_chunks = gap_s / sec_per_chunk
        self._next_cmd_allowed_chunk = float(sim_chunk_idx) + gap_chunks
        return cmd


# ================= 会话 =================


class RealtimeSession:
    def __init__(self, websocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.conversation_id = f"omni-rt-{session_id}-{int(time.time() * 1000)}"
        self.running = False
        self.s1_client = AsyncOpenAI(base_url=S1_OMNI_BASE_URL, api_key=os.environ.get("S1_API_KEY", "EMPTY"))
        self.s1_messages: list = [
            {"role": "system", "content": S1_SYSTEM_PROMPT}
        ]
        self.s2_agent = System2Agent(on_s2_text=self._send_s2_event)
        self.input_audio_queue: asyncio.Queue = asyncio.Queue()
        self.playback_buffer = AudioBuffer()
        self.drop_agent_audio = False
        self.chunk_idx = 0
        self.ws_ready = asyncio.Event()
        self.ws_error: Optional[BaseException] = None

    async def _send_json(self, data: dict):
        try:
            await self.websocket.send(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logging.debug("send json failed: %s", e)

    async def _send_s2_event(self, data: dict):
        await self._send_json(data)

    async def _get_remote_active_session(self) -> Optional[str]:
        try:
            resp = await asyncio.to_thread(requests.get, S1_HEALTH_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            active = data.get("active_session_id")
            if isinstance(active, str) and active.strip():
                return active.strip()
        except Exception as e:
            logging.warning("query health failed: %s", e)
        return None

    async def _clear_remote_session(self, session_id: str):
        try:
            await asyncio.to_thread(
                requests.delete,
                f"{S1_CLEAR_SESSION_BASE}/{session_id}",
                timeout=30,
            )
        except Exception as e:
            logging.warning("clear session failed: %s", e)

    async def _prepare_remote_session(self, session_id: str):
        active_session_id = await self._get_remote_active_session()
        if active_session_id and active_session_id != session_id:
            logging.warning("清理旧 session: old=%s new=%s", active_session_id, session_id)
            await self._clear_remote_session(active_session_id)
            await asyncio.sleep(0.2)

    def _http_status_of_ws_exc(self, e: BaseException) -> Optional[int]:
        r = getattr(e, "status_code", None) or getattr(e, "status", None)
        if r is not None:
            return int(r)
        resp = getattr(e, "response", None)
        if resp is not None:
            s = getattr(resp, "status_code", None) or getattr(resp, "status", None)
            if s is not None:
                return int(s)
        return None

    async def _audio_ws_listener(self):
        ws_url = f"{S1_AUDIO_WS_BASE}/{self.conversation_id}"
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                async with websockets.connect(ws_url, max_size=None) as ws:
                    self.ws_error = None
                    self.ws_ready.set()
                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        if not isinstance(msg, (bytes, bytearray)):
                            continue
                        if self.drop_agent_audio:
                            logging.info("丢弃 talker 音频（STOP）")
                            continue
                        pcm = wav_bytes_to_pcm(bytes(msg))
                        if not pcm:
                            continue
                        pcm = stretch_pcm_to_chunk(pcm, target_ms=CHUNK_MS)
                        await self.playback_buffer.push(pcm)
                        n = await self.playback_buffer.bytes_len()
                        if n > MAX_PLAYBACK_BUFFER_BYTES:
                            await self.playback_buffer.drop_oldest_if_over(
                                MAX_PLAYBACK_BUFFER_BYTES, MAX_PLAYBACK_BUFFER_BYTES // 2
                            )
            except websockets.exceptions.InvalidStatus as e:
                st = self._http_status_of_ws_exc(e)
                if st == 409 and attempt < max_attempts:
                    active = await self._get_remote_active_session()
                    if active and active != self.conversation_id:
                        await self._clear_remote_session(active)
                    await asyncio.sleep(0.5)
                    continue
                self.ws_error = e
                self.ws_ready.set()
                raise
            except websockets.exceptions.ConnectionClosed as e:
                logging.info("talker WebSocket 关闭: %s", e)
                self.running = False
                return
            except Exception as e:
                self.ws_error = e
                self.ws_ready.set()
                raise
            # async with 正常结束（对端关连接且未抛 ConnectionClosed 时仍可能落在此路径）
            logging.info("talker WebSocket 会话结束")
            self.running = False
            return
        self.ws_error = RuntimeError("audio ws connect exhausted retries")
        self.ws_ready.set()

    def _trim_s1_history(self):
        if len(self.s1_messages) <= 1 + MAX_S1_TURNS * 2:
            return
        head = self.s1_messages[0:1]
        tail = self.s1_messages[-(MAX_S1_TURNS * 2) :]
        self.s1_messages = head + tail
        logging.info("裁剪 s1_messages 至最近 %d 轮", MAX_S1_TURNS)

    async def _receive_loop(self):
        try:
            async for message in self.websocket:
                if not self.running:
                    break
                if isinstance(message, bytes):
                    processed = apply_noise_gate(message, SILENCE_THRESHOLD)
                    await self.input_audio_queue.put(processed)
                elif isinstance(message, str):
                    try:
                        data = json.loads(message)
                        if data.get("type") == "ping":
                            await self._send_json({"type": "pong"})
                    except json.JSONDecodeError:
                        pass
        except websockets.exceptions.ConnectionClosed:
            logging.info("[%s] 连接关闭(收)", self.session_id)
        finally:
            self.running = False

    async def _output_loop(self):
        send_n = 0
        while self.running:
            try:
                out = await self.playback_buffer.pop(CHUNK_BYTES)
                send_n += 1
                await self.websocket.send(out)
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                logging.error("output: %s", e)
                break
            await asyncio.sleep(CHUNK_MS / 1000.0)
        self.running = False

    async def _process_loop(self):
        try:
            while self.running:
                acc = bytearray()
                while len(acc) < CHUNK_BYTES:
                    try:
                        part = await asyncio.wait_for(
                            self.input_audio_queue.get(), timeout=0.55
                        )
                        acc.extend(part)
                    except asyncio.TimeoutError:
                        need = CHUNK_BYTES - len(acc)
                        acc.extend(b"\x00" * need)
                        break
                if len(acc) > CHUNK_BYTES:
                    extra = bytes(acc[CHUNK_BYTES:])
                    acc = acc[:CHUNK_BYTES]
                    await self.input_audio_queue.put(extra)
                chunk_user = bytes(acc)
                i = self.chunk_idx
                s2_cmd = self.s2_agent.get_new_commands(i)
                b64_user = pcm_to_b64_wav(chunk_user)
                s1_content = [
                    {"type": "text", "text": "{'audio_input': '"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64_user, "format": "wav"},
                    },
                    {"type": "text", "text": f"', 'from_s2': '{s2_cmd}'}}"},
                ]
                self.s1_messages.append({"role": "user", "content": s1_content})
                self._trim_s1_history()

                s1_text = "{}"
                try:
                    completion = await self.s1_client.chat.completions.create(
                        model=S1_MODEL_NAME,
                        messages=self.s1_messages,
                        max_tokens=999,
                        temperature=0.8,
                        stream=False,
                        extra_body={
                            "vllm_xargs": {"conversation_id": self.conversation_id}
                        },
                    )
                    for ch in getattr(completion, "choices", []) or []:
                        cnt = getattr(ch.message, "content", None)
                        if cnt and isinstance(cnt, str) and cnt.strip():
                            s1_text = cnt.strip()
                            break
                except Exception as e:
                    logging.error("S1 HTTP: %s", e)
                self.s1_messages.append({"role": "assistant", "content": s1_text})
                self.chunk_idx += 1
                await self._parse_and_execute_s1(s1_text, i)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error("process: %s", e, exc_info=True)
        finally:
            self.running = False

    async def _parse_and_execute_s1(self, text: str, chunk_idx: int):
        try:
            clean = text.replace("```json", "").replace("```", "").strip()
            if "}" in clean:
                clean = clean[: clean.rindex("}") + 1]
            if "{" in clean:
                clean = clean[clean.index("{") :]
            data: dict = {}
            try:
                data = json.loads(clean)
            except Exception:
                try:
                    data = ast.literal_eval(clean)
                except Exception:
                    data = {}
            asr = data.get("asr", "")
            tts = data.get("tts", "")
            tts_ctrl = data.get("tts_control", "")
            s2_ctrl = data.get("system2_control", "")

            if asr:
                await self._send_json({"type": "asr", "text": asr})
                self.s2_agent.add_context("User", asr)
            if "[STOP]" in str(tts_ctrl):
                self.drop_agent_audio = True
                await self.playback_buffer.clear()
            if tts:
                self.s2_agent.add_context("Agent", tts)
                self.drop_agent_audio = False
                await self._send_json({"type": "tts", "text": tts})
            if "[THINK]" in str(s2_ctrl):
                await self._send_json({"type": "s2_status", "status": "思考中..."})
                await self.s2_agent.trigger_think()
            elif "[WAIT]" in str(s2_ctrl):
                await self._send_json({"type": "s2_status", "status": "等待"})
                await self.s2_agent.trigger_wait()
        except Exception as e:
            logging.error("parse chunk %s: %s", chunk_idx, e)

    async def run(self):
        self.running = True
        await self._prepare_remote_session(self.conversation_id)
        self.ws_ready.clear()
        self.ws_error = None
        ws_task = asyncio.create_task(self._audio_ws_listener())
        try:
            await asyncio.wait_for(self.ws_ready.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            self.running = False
            ws_task.cancel()
            try:
                await ws_task
            except Exception:
                pass
            return
        if self.ws_error is not None:
            self.running = False
            if not ws_task.done():
                ws_task.cancel()
            try:
                await ws_task
            except (asyncio.CancelledError, Exception):
                pass
            try:
                await self._send_json(
                    {
                        "type": "error",
                        "message": f"audio ws failed: {self.ws_error!s}",
                    }
                )
            except Exception:
                pass
            await self._clear_remote_session(self.conversation_id)
            return

        receive_task = asyncio.create_task(self._receive_loop())
        process_task = asyncio.create_task(self._process_loop())
        output_task = asyncio.create_task(self._output_loop())
        try:
            await asyncio.gather(receive_task, process_task, output_task, ws_task)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            for t in (receive_task, process_task, output_task, ws_task):
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            await self._clear_remote_session(self.conversation_id)
            logging.info("[%s] 会话结束，已 clear session", self.session_id)


class OmniRealtimeServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._counter = 0

    async def _handler(self, websocket):
        self._counter += 1
        sid = f"rt_{self._counter}"
        client = websocket.remote_address
        path = getattr(websocket, "path", None)
        logging.info("新连接 %s from %s path=%s", sid, client, path)
        session = RealtimeSession(websocket, sid)
        try:
            await session.run()
        finally:
            logging.info("连接结束 %s", sid)


async def main():
    p = argparse.ArgumentParser(
        description="Omni 实时服务（WebSocket，480ms 块）",
        epilog="若 bind 失败 EADDRINUSE: 换 --port 或 fuser 查看占用。",
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    srv = OmniRealtimeServer(args.host, args.port)
    logging.info(
        "Omni 实时服务: ws://%s:%s/ (CHUNK_MS=%sms)",
        args.host,
        args.port,
        CHUNK_MS,
    )
    # websockets >=12 请用顶层 websockets.serve，避免 websockets.server.serve 弃用
    async with websockets.serve(
        srv._handler,
        args.host,
        args.port,
        ping_interval=20,
        ping_timeout=60,
        max_size=10 * 1024 * 1024,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("已停止")
