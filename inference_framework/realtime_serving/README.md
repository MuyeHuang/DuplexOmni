# realtime_serving / 实时服务栈

## English

Purpose: provide the lightweight realtime DuplexOmni serving stack used by the
open-source inference framework.

Code role: this directory contains the thinker server, talker server, MTP worker,
orchestrator, realtime WebSocket bridge, local simulation clients, and shell
helpers for starting/stopping the stack.

Usage: set model paths and endpoint environment variables, then launch the stack
with `start_thinker_talker_stack.sh`.

Default services:

| Component | Default endpoint | Health check |
| --- | --- | --- |
| Thinker server | `http://127.0.0.1:19999` | `GET /health` |
| Talker server | `http://127.0.0.1:20000` | `GET /health` |
| Orchestrator | `http://127.0.0.1:21000` | `GET /health` |
| Realtime bridge | `ws://127.0.0.1:8765` | process-level health |

User-facing entrypoints:

- Text/OpenAI-compatible: `http://127.0.0.1:21000/v1/chat/completions`
- Realtime audio stream:
  `ws://127.0.0.1:21000/v1/audio/stream/{session_id}`

`simulate_v8.py` and `simulate_video_v8.py` are scripted local simulations that
exercise the orchestrator. `omni_realtime_server.py` plus
`omni_realtime_mac_client.py` is the real microphone/WebSocket interaction path.
The S2 thinking endpoint can be a local OpenAI-compatible service or an external
API; configure it explicitly through `S2_THINK_BASE_URL`, `S2_MODEL_NAME`, and
`S2_API_KEY`.

Realtime Mac client examples:

```bash
python3 omni_realtime_server.py --host 0.0.0.0 --port 8765
python3 omni_realtime_mac_client.py --server ws://127.0.0.1:8765

# Remote server through a local Mac tunnel.
ssh -L 28765:127.0.0.1:8765 user@your-server
python3 omni_realtime_mac_client.py --server ws://127.0.0.1:28765
```

Main files:

- `start_thinker_talker_stack.sh`: start thinker/talker/orchestrator services.
- `stop_thinker_talker_stack.sh`: stop local services started by the helper.
- `serving_core/server_thinker.py`: thinker-side model service.
- `serving_core/server_talker.py`: talker-side model service.
- `serving_core/server_orchestrator.py`: realtime orchestration service.
- `omni_realtime_server.py`: WebSocket bridge for realtime interaction.
- `omni_realtime_mac_client.py`: example client.
- `simulate_v8.py`: text/audio realtime simulation.
- `simulate_video_v8.py`: video-aware simulation.

Example:

```bash
export VLLM_ROOT=../vllm_qwen3_omni
export THINKER_MODEL=/path/to/duplexomni-thinker
export TALKER_MODEL=/path/to/duplexomni-talker
export S1_MODEL_NAME=/path/to/qwen3-omni
export S1_API_KEY=EMPTY
export S2_THINK_BASE_URL=http://localhost:8000/v1
export S2_MODEL_NAME=/path-or-name/of-thinking-model
export S2_API_KEY=EMPTY
./start_thinker_talker_stack.sh
```

Health checks:

```bash
curl http://127.0.0.1:19999/health
curl http://127.0.0.1:20000/health
curl http://127.0.0.1:21000/health
```

## 中文

目的：提供开源推理框架使用的轻量实时 DuplexOmni 服务栈。

代码作用：本目录包含 thinker server、talker server、MTP worker、orchestrator、实时 WebSocket bridge、本地仿真客户端，以及启动/停止服务的 shell helper。

使用方法：设置模型路径和 endpoint 环境变量，然后通过 `start_thinker_talker_stack.sh` 启动服务栈。

默认服务：

| 组件 | 默认 endpoint | 健康检查 |
| --- | --- | --- |
| Thinker server | `http://127.0.0.1:19999` | `GET /health` |
| Talker server | `http://127.0.0.1:20000` | `GET /health` |
| Orchestrator | `http://127.0.0.1:21000` | `GET /health` |
| Realtime bridge | `ws://127.0.0.1:8765` | 进程级检查 |

用户入口：

- 文本/OpenAI-compatible：`http://127.0.0.1:21000/v1/chat/completions`
- 实时音频流：`ws://127.0.0.1:21000/v1/audio/stream/{session_id}`

`simulate_v8.py` 和 `simulate_video_v8.py` 是面向 orchestrator 的本地 scripted simulation。`omni_realtime_server.py` 加 `omni_realtime_mac_client.py` 是真实麦克风/WebSocket 交互路径。S2 thinking endpoint 可以是本地 OpenAI-compatible 服务，也可以是外部 API；需要通过 `S2_THINK_BASE_URL`、`S2_MODEL_NAME` 和 `S2_API_KEY` 显式配置。

实时 Mac 客户端示例：

```bash
python3 omni_realtime_server.py --host 0.0.0.0 --port 8765
python3 omni_realtime_mac_client.py --server ws://127.0.0.1:8765

# 远端服务通过 Mac 本地端口转发访问。
ssh -L 28765:127.0.0.1:8765 user@your-server
python3 omni_realtime_mac_client.py --server ws://127.0.0.1:28765
```

主要文件：

- `start_thinker_talker_stack.sh`：启动 thinker/talker/orchestrator 服务。
- `stop_thinker_talker_stack.sh`：停止 helper 启动的本地服务。
- `serving_core/server_thinker.py`：thinker 侧模型服务。
- `serving_core/server_talker.py`：talker 侧模型服务。
- `serving_core/server_orchestrator.py`：实时编排服务。
- `omni_realtime_server.py`：实时交互 WebSocket bridge。
- `omni_realtime_mac_client.py`：示例客户端。
- `simulate_v8.py`：文本/音频实时仿真。
- `simulate_video_v8.py`：视频感知仿真。
