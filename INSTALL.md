# Installation / 安装

## English

Purpose: describe how to install the public DuplexOmni code tree from a clean
machine or container.

Code role: this document ties together the lightweight data/runtime Python
dependencies, the modified vLLM runtime, and the Qwen3-Omni training framework.

Usage: create separate environments for data preparation, vLLM serving, and
training when possible. The vLLM environment pins newer CUDA/Torch build
requirements than the lightweight data tools.

## Environment Requirements

Recommended baseline:

- Linux x86_64 with CUDA-capable NVIDIA GPUs.
- Python 3.10 or 3.11 for data tools and training.
- Python 3.10-3.13 for the vLLM fork, matching `inference_framework/vllm_qwen3_omni/pyproject.toml`.
- CUDA 12.x for GPU inference/training.
- Recent NVIDIA driver compatible with the selected CUDA/Torch build.
- `git`, `gcc/g++`, `cmake>=3.26.1`, `ninja`, `ffmpeg`/`ffprobe` for video/audio stages.

Use `INSTALL_ROOT` below as the path that contains this `open_source` directory.

```bash
cd open_source
git status --short
```

The public source tree does not require DuplexOmni datasets or trained
thinker/talker checkpoints to be present. Until those artifacts are published,
keep the Hugging Face repository fields as placeholders such as
`<duplexomni-dataset-repo>`, `<duplexomni-thinker-model-repo>`, and
`<duplexomni-talker-model-repo>`. Downloaded assets should stay under local
`data/`, `models/`, or `outputs/` directories and must not be committed.

## 1. Lightweight Data Pipeline Environment

This environment is enough for seed/content generation, OpenAI-compatible API
calls, JSONL processing, and many parquet utilities.

```bash
python -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
git status --short
```

Additional stage dependencies:

- TTS stage requires the Qwen3-TTS and forced-aligner packages/models used by
  `data_pipeline/04_tts_parquet/synthesize_align_tts_to_parquet.py`.
- Mimi stage requires `librosa`, `soundfile`, `scipy`, `torch`, and
  `transformers`, plus a local Mimi checkpoint.
- Video stage needs `ffprobe` and video decode libraries; `av` and `pillow` are
  included in `requirements.txt`.

## 2. Modified vLLM Runtime Installation

The realtime serving stack imports the local fork under
`inference_framework/vllm_qwen3_omni`. Install it from source in a dedicated
GPU environment.

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -U pip setuptools wheel

cd inference_framework/vllm_qwen3_omni
pip install -r requirements/build.txt
pip install -r requirements/cuda.txt
pip install -e .

python -c "import vllm; print(vllm.__version__)"
```

Notes:

- The fork currently declares `torch==2.9.1` and `torchaudio==2.9.1` in
  `requirements/cuda.txt`; keep Torch, CUDA, and driver versions aligned.
- If the host already provides a compatible Torch build, review
  `use_existing_torch.py` before replacing it.
- Building vLLM CUDA extensions can take a long time and requires a working C++
  and CUDA toolchain.

## 3. Realtime Serving Installation

Use the vLLM environment after the fork is installed:

```bash
cd open_source/inference_framework/realtime_serving
source ../../.venv-vllm/bin/activate

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

## 4. Qwen3-Omni Training Framework Installation

The training fork lives under `training_framework/qwen3_omni_training`.
Install it as an editable package in a dedicated training environment.

```bash
python -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -U pip setuptools wheel

cd training_framework/qwen3_omni_training
pip install -r requirements.txt
pip install -e .

python -c "import swift; print(swift.__version__)"
cd ../..
```

Set the bundled Megatron checkout explicitly so the trainer does not need to
auto-clone it:

```bash
export SWIFT_SRC=$(pwd)/training_framework/qwen3_omni_training
export MEGATRON_LM_PATH=$(pwd)/training_framework/Megatron-LM-core_v0.15.0
```

Recommended training order:

1. Train the thinker from the base Qwen3-Omni checkpoint.
2. Reuse the saved thinker checkpoint to train the talker.
3. Keep the two stages separate; the standard training path does not require a
   joint thinker+talker run.

Optional extras from the fork:

```bash
pip install -r requirements/ray.txt
pip install -r requirements/eval.txt
pip install -r requirements/swanlab.txt
```

Use the final parquet produced by the data pipeline as the training dataset, and
pass downloaded Hugging Face model checkpoints through the training recipe.

## 中文

目的：说明如何在干净机器或容器中安装公开 DuplexOmni 代码树。

代码作用：本文把轻量数据/运行依赖、修改版 vLLM 运行时、Qwen3-Omni 训练框架的安装方式串起来。

使用方法：尽量为数据准备、vLLM 推理、训练分别创建环境。vLLM 环境通常需要比轻量数据工具更新、更严格的 CUDA/Torch 组合。

公开源码仓库不要求 DuplexOmni 数据集或训练后的 thinker/talker checkpoint 已经存在。外部资产未发布前，可以保留 `<duplexomni-dataset-repo>`、`<duplexomni-thinker-model-repo>` 和 `<duplexomni-talker-model-repo>` 等占位符。下载后的资产应放在本地 `data/`、`models/` 或 `outputs/` 中，不要提交进源码仓库。

## 环境要求

推荐基线：

- Linux x86_64，带 NVIDIA CUDA GPU。
- 数据工具和训练使用 Python 3.10 或 3.11。
- vLLM fork 支持 Python 3.10-3.13，以 `inference_framework/vllm_qwen3_omni/pyproject.toml` 为准。
- GPU 推理/训练使用 CUDA 12.x。
- NVIDIA driver 需要与所选 CUDA/Torch 版本兼容。
- 需要 `git`、`gcc/g++`、`cmake>=3.26.1`、`ninja`、`ffmpeg`/`ffprobe`。

先在根目录做静态检查：

```bash
cd open_source
git status --short
```

## 1. 轻量数据管线环境

该环境用于 seed/content 生成、OpenAI-compatible API 调用、JSONL 处理和多数 parquet 工具。

```bash
python -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
git status --short
```

额外依赖：

- TTS 阶段需要 `synthesize_align_tts_to_parquet.py` 使用的 Qwen3-TTS、forced-aligner 包和模型。
- Mimi 阶段需要 `librosa`、`soundfile`、`scipy`、`torch`、`transformers` 和本地 Mimi checkpoint。
- 视频阶段需要 `ffprobe` 和视频解码库；`av`、`pillow` 已在 `requirements.txt` 中。

## 2. 修改版 vLLM 安装

实时服务会导入 `inference_framework/vllm_qwen3_omni` 下的本地 fork。建议在独立 GPU 环境中源码安装：

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -U pip setuptools wheel

cd inference_framework/vllm_qwen3_omni
pip install -r requirements/build.txt
pip install -r requirements/cuda.txt
pip install -e .

python -c "import vllm; print(vllm.__version__)"
```

注意：

- `requirements/cuda.txt` 当前声明 `torch==2.9.1` 和 `torchaudio==2.9.1`；需要保持 Torch、CUDA、driver 匹配。
- 如果宿主机已有兼容 Torch，替换前先阅读 `use_existing_torch.py`。
- 构建 vLLM CUDA extension 需要 C++/CUDA 工具链，耗时可能较长。

## 3. 实时服务安装

安装 vLLM fork 后，在同一环境中启动：

```bash
cd open_source/inference_framework/realtime_serving
source ../../.venv-vllm/bin/activate

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

停止：

```bash
./stop_thinker_talker_stack.sh
```

## 4. Qwen3-Omni 训练框架安装

训练 fork 位于 `training_framework/qwen3_omni_training`，建议用独立训练环境 editable 安装：

```bash
python -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -U pip setuptools wheel

cd training_framework/qwen3_omni_training
pip install -r requirements.txt
pip install -e .

python -c "import swift; print(swift.__version__)"
cd ../..
```

请显式指定本仓库内的 Megatron checkout，避免训练器自动联网克隆：

```bash
export SWIFT_SRC=$(pwd)/training_framework/qwen3_omni_training
export MEGATRON_LM_PATH=$(pwd)/training_framework/Megatron-LM-core_v0.15.0
```

推荐训练顺序：

1. 先用基础 Qwen3-Omni checkpoint 训练 thinker。
2. 再复用保存下来的 thinker checkpoint 训练 talker。
3. 两个阶段分开跑；标准训练流程不需要联合训练。

可选依赖：

```bash
pip install -r requirements/ray.txt
pip install -r requirements/eval.txt
pip install -r requirements/swanlab.txt
```

训练输入是数据管线产出的最终 parquet；模型 checkpoint 从 Hugging Face 下载后传入训练 recipe。
