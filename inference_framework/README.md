# Inference Framework / 推理框架

## English

Purpose: stage the runtime serving stack, modified vLLM fork, and minimal
commands needed to launch realtime DuplexOmni inference.

Code role: `realtime_serving` provides thinker/talker/orchestrator serving and
clients; `vllm_qwen3_omni` provides the modified runtime used by those servers.

This directory packages the inference assets:

- `realtime_serving/`: realtime thinker/talker/orchestrator serving stack,
  realtime clients, and local simulation scripts. See
  `realtime_serving/README.md`.
- `vllm_qwen3_omni/`: modified vLLM fork used by serving and checkpoint/runtime
  workflows.

Datasets and model weights are not included. Download the thinker, talker, TTS,
and other model assets from the public model repositories and pass local paths
through environment variables.

Usage: install/build the modified vLLM runtime as needed, set model and endpoint
environment variables, then launch `realtime_serving`.

Example:

```bash
cd inference_framework/realtime_serving
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

Stop:

```bash
./stop_thinker_talker_stack.sh
```

Realtime bridge:

```bash
python3 omni_realtime_server.py --host 0.0.0.0 --port 8765
python3 omni_realtime_mac_client.py --server ws://127.0.0.1:8765
```

For a remote server, either use a reachable host directly or forward the remote
port to the Mac:

```bash
# Direct remote connection.
python3 omni_realtime_mac_client.py --server ws://your-server-host:8765

# Local Mac tunnel to the remote bridge.
ssh -L 28765:127.0.0.1:8765 user@your-server
python3 omni_realtime_mac_client.py --server ws://127.0.0.1:28765
```

## 中文

本目录打包推理框架资产：

目的：打包实时 DuplexOmni 推理所需的服务栈、修改版 vLLM fork 和最小启动命令。

代码作用：`realtime_serving` 提供 thinker/talker/orchestrator 服务和客户端；`vllm_qwen3_omni` 提供这些服务依赖的修改版运行时。

使用方法：按需安装/构建修改版 vLLM，设置模型和 endpoint 环境变量，然后启动 `realtime_serving`。

- `realtime_serving/`：实时 thinker/talker/orchestrator 服务栈、实时客户端和本地仿真脚本。
- 细节见 `realtime_serving/README.md`。
- `vllm_qwen3_omni/`：服务和 checkpoint/runtime 流程使用的修改版 vLLM fork。

数据和模型权重不包含在代码仓库中。请从公开模型仓库下载 thinker、talker、TTS 等资产，并通过环境变量传入本地路径。

启动、停止和实时桥接示例见上方命令。服务和 Mac 在同一台机器上时可使用
`ws://127.0.0.1:8765`；服务在远端时，请使用 Mac 能访问到的远端地址，或先通过
`ssh -L 28765:127.0.0.1:8765 user@your-server` 做本地转发，再运行
`python3 omni_realtime_mac_client.py --server ws://127.0.0.1:28765`。运行时需要替换模型路径、endpoint 和 key；`EMPTY` 只适用于本地无鉴权 OpenAI-compatible 服务。
