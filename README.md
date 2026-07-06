# 🚀 DuplexOmni

🌐 Language: **English** | [中文说明](#中文说明)

**DuplexOmni** is a realtime multimodal full-duplex interaction system for
listening, seeing, thinking, and speaking in parallel. This repository contains
the public source code for the data pipeline, training framework, and realtime
serving stack described in the paper:

📄 Paper: [DuplexOmni: Real-Time Listening, Seeing, Thinking, and Speaking for
Full-Duplex Interaction](https://arxiv.org/abs/2606.09186)

## Overview

DuplexOmni targets realtime interactive agents that can continue processing
streaming audio/video input while producing speech output. The system separates
fast interaction from slower reasoning: the interaction model handles
low-latency listening, seeing, and speaking, while a pluggable System-2 layer can
perform heavier reasoning, retrieval, or tool use in the background.

This source release includes:

- data generation and cleaning pipelines for duplex dialogue sessions;
- TTS, alignment, parquet, Mimi-codec, and noise augmentation utilities;
- Qwen3-Omni E2E training code based on ms-swift and Megatron;
- a realtime inference stack with a modified vLLM runtime;
- documentation for external datasets, checkpoints, and runtime dependencies.

Generated datasets, trained model weights, checkpoints, logs, internal service
endpoints, generated parquet, and generated media are not included in this
repository. When public artifact repositories are available, download those
assets separately and pass their local paths to the relevant scripts.

Questions and contributions are welcome through
[GitHub Issues](https://github.com/MuyeHuang/DuplexOmni/issues). If you are
interested in this work, please consider starring the repository. The full
training data is about 9TB, so complete public upload may take time; thank you
for your patience.

## For Developers

Read the setup documents for the part of the stack you want to run:

```bash
git status --short
```

Recommended reading order:

1. [`CONCEPTS.md`](CONCEPTS.md) for terminology and training signals.
2. [`EXTERNAL_ASSETS.md`](EXTERNAL_ASSETS.md) for datasets, checkpoints, noise
   corpora, and placeholder artifact repositories.
3. [`INSTALL.md`](INSTALL.md) for environment setup.
4. [`DEPENDENCIES.md`](DEPENDENCIES.md) for file-level dependencies.
5. [`data_pipeline/README.md`](data_pipeline/README.md),
   [`training_framework/README.md`](training_framework/README.md), and
   [`inference_framework/README.md`](inference_framework/README.md) for
   component-specific workflows.

If you use an automated coding or runtime agent, provide
[`for_agent.md`](for_agent.md) first. It defines safe execution boundaries,
required assets, validation boundaries and the expected repository contract.

## Repository Layout

The code tree is organized as three implementation layers:

```text
open_source/
  data_pipeline/          seed/content generation, writer/director, cleaning, TTS, parquet, Mimi, noise, video branch
  training_framework/     Qwen3-Omni E2E training framework based on ms-swift/Megatron
  inference_framework/    realtime_serving serving stack, modified vLLM fork, export/runtime conventions
  assets/                 small bundled assets such as the reference voice prompt
  INSTALL.md              environment and installation guide
  DEPENDENCIES.md         file-level dependency map
  CONCEPTS.md             terminology for S1/S2, Thinker/Talker, MTP, and data splits
  EXTERNAL_ASSETS.md      datasets, checkpoints, TTS, Mimi, MUSAN, and FSD50K assets
  LICENSE                 Apache-2.0 license for DuplexOmni-owned source
  THIRD_PARTY_NOTICES.md  bundled upstream source and external asset notices
  for_agent.md            execution guide for coding/runtime agents
  requirements.txt
```

The only bundled audio asset is the small reference voice sample under
`assets/reference_audio/google_Leda.wav`.

## Release And Assets

This repository is intended for the public **source code** release. DuplexOmni
datasets, trained thinker/talker checkpoints, and optional TTS assets are
external artifacts. Until public Hugging Face repositories are available, the
documentation may use placeholder identifiers:

```text
<duplexomni-dataset-repo>
<duplexomni-thinker-model-repo>
<duplexomni-talker-model-repo>
<tts-model-repo-if-needed>
```

Do not commit local `data/`, `models/`, `outputs/`, logs, checkpoints, parquet,
or generated media. The top-level `.gitignore` blocks common generated files.
Third-party source notices are listed in `THIRD_PARTY_NOTICES.md`.

Purpose: provide the DuplexOmni data pipeline, training framework, and realtime
serving source code while keeping datasets and checkpoints as separate external
artifacts.

Code role: `data_pipeline` builds the dataset, `training_framework` trains the
Qwen3-Omni E2E model, and `inference_framework` serves the realtime system.

Usage: start with `git status --short`, then follow
`CONCEPTS.md`, `EXTERNAL_ASSETS.md`, `INSTALL.md`, `DEPENDENCIES.md`,
`data_pipeline/README.md`, `training_framework/README.md`, and
`inference_framework/README.md` in order.

## What This Repository Does

DuplexOmni separates realtime interaction from slower thinking. The interaction
model handles streaming audio/video input and realtime speech/text output. A
pluggable thinking layer can run heavier reasoning or tool-use in the
background. This repository contains the engineering pieces needed to reproduce
that stack:

- a Writer-Director data pipeline that creates continuous-interaction training
  data;
- audio, parquet, Mimi-codec, and noise stages that turn scripts into E2E
  training examples;
- a Qwen3-Omni/Megatron training framework for thinker-first, talker-second
  training;
- a realtime serving stack with a modified vLLM fork.

The central data idea is to synthesize multi-turn call/dialogue sessions where:

- `User` and `Agent` can overlap in time.
- User interruptions are represented with `^`, `[CUT]`, `[WAIT]`, and `[PENDXS]`.
- Some user requests trigger a System-2 thinking channel with `[THINK]`.
- System-2 messages are injected as `「...」` before the Agent speaks facts or
  conclusions that require reasoning, retrieval, or computation.
- The final training example contains text turns, aligned audio chunks, optional
  codec features, and metadata needed by the Qwen3-Omni E2E training stack.

The repository keeps the original pipeline logic but replaces private defaults
with public placeholders. Real runs require explicit paths, model checkpoints,
and OpenAI-compatible endpoints.

## Reference Voice

The agent voice is based on the `google_Leda` voice. The bundled
reference audio is:

```text
assets/reference_audio/google_Leda.wav
```

The reference text is:

```text
The weather is nice, and I speak calmly and clearly. 今天天气很好，我平静清楚地说话。
```

## Data Sources And Composition

The code is designed around a mixture of public text and video dialogue sources,
plus synthetic seed guidance:

| Source group | Purpose | Code |
| --- | --- | --- |
| Seed configs | Generate controllable scenario/style/interaction constraints. The repository includes config and generation/cleaning code, not generated `samples*.jsonl`. | `data_pipeline/00_seed_configs/` |
| General text chat | Convert public multi-turn or instruction chat into inbound/outbound voice-agent content. Examples include UltraChat-style and other open dialogue corpora expected under `DATASETS_ROOT`. | `data_pipeline/01_content/` |
| Video chat | Convert video event tags into video-grounded voice-agent dialogue content. | `data_pipeline/07_video_branch/` |
| Writer output | Use content + seed constraints to write natural scripts. | `data_pipeline/02_writer_director/` |
| Director output | Convert scripts into tag-rich training messages with duplex timing and System-2 control tags. | `data_pipeline/02_writer_director/` |
| TTS/audio | Synthesize and align User/Agent speech, then cut into session chunks. | `data_pipeline/04_tts_parquet/` |
| Codec | Extract Mimi codec features for E2E audio supervision. | `data_pipeline/05_mimi_codec/` |
| Noise | Add user-side noise augmentation for robust speech training. | `data_pipeline/06_noise/` |

Data and generated artifacts are intentionally excluded from this source tree.
When external dataset/model repositories are available, configure local paths
through CLI arguments or environment variables after download.

See `EXTERNAL_ASSETS.md` for the public content sources, expected Hugging Face
assets, MUSAN/FSD50K noise sources, and local directory layout.

## Authoritative DAG

The main text/audio DAG is:

```text
seed config/guidance
  -> generate_slot_permutations.py
  -> clean_samples_with_vllm.py
  -> samples.cleaned.jsonl

public content datasets
  -> clean_all_datasets.py / clean_ultrachat.py
  -> inbound/outbound content jsonl

samples.cleaned.jsonl + content jsonl
  -> voiceagent_text_pipeline.py --mode writer_only
  -> writer records

writer records
  -> classify_s1_s2.py
  -> clean_scripts.py
  -> hallucination_filter.py
  -> hallucination_repair_loop.py
  -> cleaned/repaired writer records

cleaned/repaired writer records
  -> voiceagent_text_pipeline.py --mode director_only
  -> director messages with [THINK], [WAIT], [PENDXS], ^, [CUT], and S2 `「...」`

director messages
  -> synthesize_align_tts_to_parquet.py
  -> build_training_parquet.py
  -> strip_self_audio_from_parquet.py
  -> training parquet

training parquet
  -> extract_mimi_codes_parquet.py
  -> inject_noise_parquet.py
  -> final E2E training parquet
```

The video branch shares the same writer/director and audio/parquet ideas but adds
video event tagging:

```text
video files + metadata
  -> auto_tag_nextqa_events.py / batch_tag_nextqa_videos.py / batch_tag_llava_videos.py
  -> clean_videochat.py
  -> voiceagent_video_pipeline.py
  -> video writer/director records
  -> extract_mimi_codes_video_parquet.py
```

## Directory Guide

### `data_pipeline/00_seed_configs`

Contains seed families:

- `seed_big`
- `seed_bc_big`
- `seed_xp_big`
- `seed_sparse`
- `seed_emotion`
- `seed_video`

Each family includes `config.json`, `guidance.json`, and generation/cleaning code
where applicable. Generated `samples*.jsonl` files are excluded.

### `data_pipeline/01_content`

Content conversion scripts:

- `clean_all_datasets.py`: unified cleaner for multiple public text datasets.
- `clean_ultrachat.py`: UltraChat-style conversion.
- `clean_videochat.py`: video-tag-to-dialogue content conversion.

### `data_pipeline/02_writer_director`

Main authoritative entrypoint:

```text
voiceagent_text_pipeline.py
```

It supports `writer_only`, `director_only`, and `full_pipeline`, resumable output,
inbound/outbound splits, multi-provider OpenAI-compatible endpoints, and
DeepSeek/Qwen-style thinking request bodies.

### `data_pipeline/03_between_writer_and_director`

Quality and repair steps between writer and director:

- `classify_s1_s2.py`: classify S1/S2 samples.
- `clean_scripts.py`: naturalize and normalize writer scripts.
- `hallucination_filter.py`: detect missing-context hallucination patterns.
- `hallucination_repair_loop.py`: iterative repair and recheck.

### `data_pipeline/04_tts_parquet`

Audio synthesis, alignment, and parquet conversion:

- `synthesize_align_tts_to_parquet.py`
- `build_training_parquet.py`
- `strip_self_audio_from_parquet.py`

### `data_pipeline/05_mimi_codec`

Mimi code extraction for text/audio E2E supervision:

- `config.py`
- `extract_mimi_codes_parquet.py`
- `extract_mimi_codes_video_parquet.py`

### `data_pipeline/06_noise`

Noise generation and injection:

- `build_background_noise.py`
- `inject_noise_parquet.py`

### `training_framework/qwen3_omni_training`

Training framework fork with Qwen3-Omni E2E changes, Megatron trainer changes,
recipes, and data reading schema. This directory is copied as a framework asset;
it retains upstream examples and documentation.

`training_framework/Megatron-LM-core_v0.15.0` is bundled alongside the training
fork and should be exported through `MEGATRON_LM_PATH` for training runs.

### `inference_framework`

- `realtime_serving`: thinker/talker/orchestrator serving stack and realtime client.
- `vllm_qwen3_omni`: modified vLLM fork used by serving and export/runtime workflows.

## Required Runtime Dependencies

The exact environment depends on which stage is run. At minimum:

- Python 3.10+
- `openai`
- `tqdm`
- `numpy`
- `pyarrow`
- `torch`
- `transformers`

Install the lightweight dependency file first:

```bash
pip install -r requirements.txt
```

Stage-specific optional dependencies:

- Content/video: `pandas` or `pyarrow`, `ffprobe`, video decoding libraries.
- TTS/audio: Qwen3-TTS dependencies, `scipy`, `soundfile`/audio stack as needed.
- Noise: `pedalboard` optional, with numpy fallback in some paths.
- Inference: modified `vllm_qwen3_omni`, FastAPI/Uvicorn, CUDA-capable GPUs.

## Configuration

Internal paths and keys have been replaced with placeholders. Configure real
values explicitly. Point these variables at Hugging Face repositories or at
local directories created from those repositories.

External dataset/checkpoint repository placeholders:

```bash
export HF_DATASET_REPO="<duplexomni-dataset-repo>"
export HF_THINKER_MODEL_REPO="<duplexomni-thinker-model-repo>"
export HF_TALKER_MODEL_REPO="<duplexomni-talker-model-repo>"
export HF_TTS_MODEL_REPO="<tts-model-repo-if-needed>"
```

Typical local layout after downloading Hugging Face assets:

```text
data/
  seed/
  content/
models/
  duplexomni-thinker/
  duplexomni-talker/
  qwen3-tts-custom/
  qwen3-tts-base/
  qwen3-forced-aligner/
  mimi/
outputs/
  pipeline_text/
  tts/
  parquet/
```

When external asset repositories are available, use `huggingface-cli download`
or your preferred artifact manager to populate these directories, then pass the
resulting local paths to the scripts below. Public content source URLs are listed in
`EXTERNAL_ASSETS.md`.

Common OpenAI-compatible settings:

```bash
export API_KEY=EMPTY
export API_BASES=http://localhost:8000/v1
export MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
```

Dataset and artifact roots:

```bash
export DATASETS_ROOT=/path/to/public/datasets
export VOICEAGENT_ROOT=/path/to/workspace
```

TTS/Mimi:

```bash
export QWEN3_TTS_CUSTOM_MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice
export QWEN3_TTS_BASE_MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-Base
export QWEN3_ALIGNER_MODEL=/path/to/Qwen3-ForcedAligner-0.6B
export MIMI_PATH=/path/to/mimi
```

Realtime serving:

```bash
export VLLM_ROOT=/path/to/open_source/inference_framework/vllm_qwen3_omni
export THINKER_MODEL=/path/to/qwen3-omni-thinker-checkpoint
export TALKER_MODEL=/path/to/qwen3-omni-talker-checkpoint
export S1_MODEL_NAME=/path/to/qwen3-omni-checkpoint
export S1_API_KEY=EMPTY
export S2_THINK_BASE_URL=http://localhost:8000/v1
export S2_MODEL_NAME=/path-or-name/of-thinking-model
export S2_API_KEY=EMPTY
```

## Basic Usage

Run commands from the `open_source/` directory unless noted otherwise.

### 1. Inspect the repository tree

```bash
git status --short
```

The command should not show generated data, logs, checkpoints, parquet files, or
model weights before you commit or share the repository.

### 2. Generate and clean seed samples

Example for one seed family:

```bash
cd data_pipeline/00_seed_configs/seed_big
python3 generate_slot_permutations.py \
  --sample-limit 10000 \
  --output-jsonl samples.jsonl

python3 clean_samples_with_vllm.py \
  --input-jsonl samples.jsonl \
  --output-jsonl samples.cleaned.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --model gpt-4.1
```

Repeat for the seed families you want to use. The resulting `samples.cleaned.jsonl`
files belong in an external Hugging Face dataset repository or local data
directory, not in the code repository.

### 3. Prepare content jsonl

Example:

```bash
python3 data_pipeline/01_content/clean_ultrachat.py \
  --input-jsonl /path/to/ultrachat/train_sft.jsonl \
  --output-jsonl data/content/train_sft.voiceagent.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --model gpt-4.1
```

For multi-dataset cleaning, configure `DATASETS_ROOT` and use
`clean_all_datasets.py`:

```bash
DATASETS_ROOT=/path/to/public/datasets \
python3 data_pipeline/01_content/clean_all_datasets.py \
  --dataset all \
  --split both \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --model gpt-4.1
```

### 4. Run writer/director

Single local provider example:

```bash
python3 data_pipeline/02_writer_director/voiceagent_text_pipeline.py \
  --mode full_pipeline \
  --split both \
  --api-key EMPTY \
  --api-base http://localhost:8000/v1 \
  --model-name deepseek-ai/DeepSeek-V4-Flash \
  --writer-provider-api-bases local=http://localhost:8000/v1 \
  --director-provider-api-bases local=http://localhost:8000/v1 \
  --seed-path data/seed/samples.cleaned.jsonl \
  --inbound-path data/content/inbound.jsonl \
  --outbound-path data/content/outbound.jsonl \
  --output-dir outputs/pipeline_text
```

Multi-provider format:

```bash
--director-provider-api-bases p0=http://host0:8000/v1,p1=http://host1:8000/v1
--director-provider-models p0=deepseek-ai/DeepSeek-V4-Flash,p1=deepseek-ai/DeepSeek-V4-Flash
```

### 5. Run between-writer-and-director cleaning

```bash
export API_BASES=http://localhost:8000/v1
export API_KEY=EMPTY
export MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash

python3 data_pipeline/03_between_writer_and_director/classify_s1_s2.py \
  --input outputs/pipeline_text/writer.jsonl \
  --output outputs/pipeline_text/writer.s1s2.jsonl

python3 data_pipeline/03_between_writer_and_director/clean_scripts.py \
  --input outputs/pipeline_text/writer.s1s2.jsonl \
  --output outputs/pipeline_text/writer.cleaned.jsonl

python3 data_pipeline/03_between_writer_and_director/hallucination_filter.py \
  --input outputs/pipeline_text/writer.cleaned.jsonl

python3 data_pipeline/03_between_writer_and_director/hallucination_repair_loop.py \
  --input outputs/pipeline_text/writer.cleaned.halluc.jsonl \
  --output outputs/pipeline_text/writer.repaired.jsonl
```

Adjust argument names as needed per script help output.

### 6. TTS, parquet, Mimi, noise

```bash
python3 data_pipeline/04_tts_parquet/synthesize_align_tts_to_parquet.py \
  --input-path outputs/pipeline_text/director.jsonl \
  --output-dir outputs/tts/output_dialogue_e2e \
  --cut-dir outputs/tts/finalcut_sessions

python3 data_pipeline/04_tts_parquet/build_training_parquet.py \
  --input-parquet-dir outputs/tts/finalcut_sessions \
  --output-dir outputs/parquet/training_v7

python3 data_pipeline/05_mimi_codec/extract_mimi_codes_parquet.py \
  --input-parquet-dir outputs/parquet/training_v7 \
  --output-dir outputs/parquet/training_v7_codec \
  --mimi-path /path/to/mimi

python3 data_pipeline/04_tts_parquet/strip_self_audio_from_parquet.py \
  --input-dir outputs/parquet/training_v7_codec \
  --output-dir outputs/parquet/training_v7_codec_noself

python3 data_pipeline/06_noise/inject_noise_parquet.py \
  --input-dir outputs/parquet/training_v7_codec_noself \
  --output-dir outputs/parquet/training_v7_codec_noself_noised
```

Some legacy scripts still expose defaults for the historical pipeline layout.
Use CLI arguments or environment variables to point them at your local paths.

### 7. Video branch

```bash
python3 data_pipeline/07_video_branch/batch_tag_nextqa_videos.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --model Qwen/Qwen3.5-397B-A17B

python3 data_pipeline/07_video_branch/clean_videochat.py \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY

python3 data_pipeline/07_video_branch/voiceagent_video_pipeline.py \
  --provider-api-bases local=http://localhost:8000/v1 \
  --model-name deepseek-ai/DeepSeek-V4-Flash
```

### 8. Training framework

The training framework lives under:

```text
training_framework/qwen3_omni_training/
```

Use its own `README.md`, examples, and Megatron/Qwen3-Omni E2E trainer code. The
dataset expected by training is the final parquet output produced by the data
pipeline.

### 9. Realtime inference stack

Start the thinker/talker/orchestrator stack:

```bash
cd inference_framework/realtime_serving
export VLLM_ROOT=../vllm_qwen3_omni
export THINKER_MODEL=/path/to/thinker/checkpoint
export TALKER_MODEL=/path/to/talker/checkpoint
./start_thinker_talker_stack.sh
```

Stop it:

```bash
./stop_thinker_talker_stack.sh
```

Local simulation:

```bash
python3 simulate_v8.py
python3 simulate_video_v8.py
```

Realtime WebSocket bridge:

```bash
python3 omni_realtime_server.py --host 0.0.0.0 --port 8765
python3 omni_realtime_mac_client.py --server ws://127.0.0.1:8765
```

If the server runs on a remote machine, either connect to the reachable remote
host directly:

```bash
python3 omni_realtime_mac_client.py --server ws://your-server-host:8765
```

or forward the remote bridge to a local Mac port:

```bash
ssh -L 28765:127.0.0.1:8765 user@your-server
python3 omni_realtime_mac_client.py --server ws://127.0.0.1:28765
```

## Repository Sanity Check

Before sharing the repository or running a full pipeline:

```bash
git status --short
find . -type f \( -name '*.pyc' -o -name '*.log' -o -name 'samples*.jsonl' \)
```

Expected:

- `git status --short` does not show generated data, logs, media, parquet, or
  cache files.
- No generated data, logs, media, parquet, or cache files are present.
- No private endpoint, private key, production path, or model checkpoint path is
  present.

## Expected Hugging Face Asset Layout

Expected external asset layout:

```text
Hugging Face dataset repo
  seed/samples.cleaned.jsonl
  content/inbound.jsonl
  content/outbound.jsonl
  writer_director/director.jsonl
  parquet/training_v7_codec_noself_noised/
  video/parquet/

Hugging Face model repos
  duplexomni-thinker/
  duplexomni-talker/
  qwen3-tts-custom/
  mimi/ or documented external Mimi dependency
```

The source tree uses placeholders like `data/...`, `models/...`, and
`outputs/...`; after downloading Hugging Face assets, either place files under
those directories or pass explicit CLI arguments.

## Important Limitations

- This source tree does not bundle weights or datasets directly; those assets
  are expected as separate Hugging Face repositories or local artifact
  directories.
- Hugging Face dataset/model repository names may remain placeholders until the
  public artifact uploads are complete.
- `localhost` endpoints are placeholders for OpenAI-compatible services.
- `models/...`, `data/...`, and `outputs/...` are placeholders.
- Large framework directories (`training_framework` and `vllm_qwen3_omni`) retain
  upstream examples and tests; custom data and model artifacts are excluded.

## Citation / BibTeX

If you use DuplexOmni in research, products, demos, or derivative open-source
projects, please cite the paper:

```bibtex
@misc{huang2026duplexomnirealtimelisteningseeing,
      title={DuplexOmni: Real-Time Listening, Seeing, Thinking, and Speaking for Full-Duplex Interaction},
      author={Muye Huang and Lingling Zhang and Xingyu Yu and Lei Shi and Zhanyu Ma and Jun Xu and Jiuchong Gao and Jinghua Hao and Renqing He and Jun Liu},
      year={2026},
      eprint={2606.09186},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2606.09186},
}
```

## 中文说明

🌐 语言：[English overview](#overview) | **中文**

本仓库提供 **DuplexOmni** 的公开源码。DuplexOmni 是一个面向实时多模态全双工交互的系统，目标是在同一交互过程中并行完成听、看、思考与说话。本仓库包含论文中数据管线、训练框架和实时推理服务栈对应的工程实现。

📄 论文：[DuplexOmni: Real-Time Listening, Seeing, Thinking, and Speaking for
Full-Duplex Interaction](https://arxiv.org/abs/2606.09186)

### 🚀 概览

DuplexOmni 面向实时全双工多模态交互：系统在生成语音输出的同时持续处理流式音频/视频输入，并通过可插拔的 System-2 层承载较慢的推理、检索或工具调用。本仓库覆盖从训练数据构建、模型训练到实时推理服务的主要工程组件。

本源码发布包含：

- duplex dialogue 数据生成、清洗与修复流程；
- TTS、对齐、parquet 构建、Mimi codec 提取和噪声增强工具；
- 基于 ms-swift/Megatron 的 Qwen3-Omni E2E 训练框架；
- 基于修改版 vLLM 的实时推理服务栈；
- 外部数据、checkpoint 和运行依赖的文档说明。

生成数据集、训练后的模型权重、checkpoint、日志、内部服务地址、生成 parquet 和生成媒体文件不包含在本仓库中。公开资产仓库可用后，请单独下载并通过脚本参数或环境变量传入本地路径。

如有问题或建议，欢迎通过 [GitHub Issues](https://github.com/MuyeHuang/DuplexOmni/issues) 反馈；如果你对本项目感兴趣，也欢迎点一个 star。完整训练数据约 9TB，公开完整上传需要较长时间，请耐心等待。

### 🛠️ 面向开发者

先阅读对应组件的安装和运行文档：

```bash
git status --short
```

推荐阅读顺序：

1. [`CONCEPTS.md`](CONCEPTS.md)：核心术语、训练信号和 S1/S2 控制约定。
2. [`EXTERNAL_ASSETS.md`](EXTERNAL_ASSETS.md)：数据集、checkpoint、噪声语料和资产仓库占位符。
3. [`INSTALL.md`](INSTALL.md)：环境安装和依赖准备。
4. [`DEPENDENCIES.md`](DEPENDENCIES.md)：文件级依赖关系。
5. [`data_pipeline/README.md`](data_pipeline/README.md)、[`training_framework/README.md`](training_framework/README.md) 和 [`inference_framework/README.md`](inference_framework/README.md)：各子系统的具体工作流。

如果使用自动化编码或运行工具，请先阅读 [`for_agent.md`](for_agent.md)。该文件说明仓库边界、安全执行顺序、外部资产、环境变量、验证边界和预期仓库约定。

目的：提供 DuplexOmni 的数据管线、训练框架和实时服务代码，同时把数据集和 checkpoint 作为独立外部资产管理。

代码作用：`data_pipeline` 构建数据，`training_framework` 训练 Qwen3-Omni E2E 模型，`inference_framework` 提供实时服务栈。

使用方法：先运行 `git status --short` 自检，再按 `INSTALL.md`、`DEPENDENCIES.md`、`data_pipeline/README.md`、`training_framework/README.md`、`inference_framework/README.md` 的说明执行。

### 📁 仓库结构

目录按三个实现层组织：

```text
open_source/
  data_pipeline/          seed/content 生成、writer/director、清洗修复、TTS、parquet、Mimi、噪声、视频分支
  training_framework/     基于 ms-swift/Megatron 的 Qwen3-Omni E2E 训练框架
  inference_framework/    realtime_serving 服务栈、修改版 vLLM、权重导出与运行约定
  assets/                 小型附带资产，例如参考音色音频
  INSTALL.md              环境和安装说明
  DEPENDENCIES.md         文件级依赖关系
  CONCEPTS.md             S1/S2、Thinker/Talker、MTP 和数据切分术语
  EXTERNAL_ASSETS.md      数据集、checkpoint、TTS、Mimi、MUSAN、FSD50K 等外部资产说明
  LICENSE                 DuplexOmni 自有源码的 Apache-2.0 许可证
  THIRD_PARTY_NOTICES.md  上游源码和外部资产说明
  for_agent.md            给自动化 Agent 的执行说明
  requirements.txt
```

仓库中唯一附带的音频资产是 `assets/reference_audio/google_Leda.wav`，用于参考音色示例。

### 📦 发布与资产状态

本仓库面向公开源码发布。DuplexOmni 数据集、Thinker/Talker checkpoint 和可选 TTS 资产是外部资产；在公开 Hugging Face 仓库可用前，文档中可能继续使用以下占位符：

```text
<duplexomni-dataset-repo>
<duplexomni-thinker-model-repo>
<duplexomni-talker-model-repo>
<tts-model-repo-if-needed>
```

不要提交本地 `data/`、`models/`、`outputs/`、日志、checkpoint、parquet 或生成媒体。顶层 `.gitignore` 会拦截常见生成文件。第三方源码说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

### 🧩 本仓库做什么

DuplexOmni 将实时交互和较慢思考分离。交互模型负责流式音频/视频输入，以及实时语音/文本输出；可插拔的 thinking 层可以在后台执行更重的推理、检索或工具调用。本仓库包含复现这套系统所需的主要工程组件：

- Writer-Director 数据管线，用于构造连续交互训练数据；
- 音频、parquet、Mimi codec 和噪声阶段，用于把脚本转换成 E2E 训练样本；
- Qwen3-Omni/Megatron 训练框架，用于 thinker-first、talker-second 训练；
- 基于修改版 vLLM 的实时服务栈。

核心数据思想是合成多轮通话/对话 session，其中：

- `User` 和 `Agent` 可以在时间上重叠说话；
- 用户打断通过 `^`、`[CUT]`、`[WAIT]` 和 `[PENDXS]` 表示；
- 部分用户请求会通过 `[THINK]` 触发 System-2 thinking channel；
- 当 Agent 的事实性回答、结论或工具结果依赖推理、检索或计算时，System-2 消息会以 `「...」` 形式注入到 Agent 发言前；
- 最终训练样本包含文本轮次、对齐后的语音片段、可选 codec 特征，以及 Qwen3-Omni E2E 训练栈需要的元信息。

仓库保留原始管线逻辑，但将私有默认路径替换成公开占位符。真实运行需要显式传入本地路径、模型 checkpoint 和 OpenAI-compatible endpoint。

### 🎙️ 参考音色

Agent 音色来源是 `google_Leda`。随代码树提供的参考音频为：

```text
assets/reference_audio/google_Leda.wav
```

参考文本为：

```text
The weather is nice, and I speak calmly and clearly. 今天天气很好，我平静清楚地说话。
```

### 🗂️ 数据来源与组成

代码围绕公开文本/视频对话来源和合成 seed guidance 的混合数据设计，并只包含配置与处理代码，不包含生产数据或生成样本。

| 数据组 | 作用 | 代码位置 |
| --- | --- | --- |
| Seed configs | 生成可控场景、风格和交互约束；仓库包含配置、生成和清洗代码，不包含生成后的 `samples*.jsonl`。 | `data_pipeline/00_seed_configs/` |
| 通用文本对话 | 将公开多轮对话或指令对话转换成 inbound/outbound 语音智能体内容，例如 UltraChat 风格数据和其他位于 `DATASETS_ROOT` 下的公开对话语料。 | `data_pipeline/01_content/` |
| 视频对话 | 将视频事件标签转换成 video-grounded 语音智能体对话内容。 | `data_pipeline/07_video_branch/` |
| Writer output | 使用 content + seed constraints 写自然剧本。 | `data_pipeline/02_writer_director/` |
| Director output | 把剧本转换成包含全双工时序和 System-2 控制标签的训练消息。 | `data_pipeline/02_writer_director/` |
| TTS/audio | 合成并对齐 User/Agent 语音，然后切成 session chunk。 | `data_pipeline/04_tts_parquet/` |
| Codec | 提取 Mimi codec 特征，用于 E2E 音频监督。 | `data_pipeline/05_mimi_codec/` |
| Noise | 添加用户侧噪声增强，提升语音训练鲁棒性。 | `data_pipeline/06_noise/` |

数据和生成产物不会放在源码树里。外部数据或模型仓库发布后，请下载到本地，并通过 CLI 参数或环境变量传给对应脚本。公开内容来源、Hugging Face 资产、MUSAN/FSD50K 噪声源和本地目录布局见 [`EXTERNAL_ASSETS.md`](EXTERNAL_ASSETS.md)。

### 🔁 权威数据 DAG

主线文本/音频 DAG：

```text
seed config/guidance
  -> generate_slot_permutations.py
  -> clean_samples_with_vllm.py
  -> samples.cleaned.jsonl

public content datasets
  -> clean_all_datasets.py / clean_ultrachat.py
  -> inbound/outbound content jsonl

samples.cleaned.jsonl + content jsonl
  -> voiceagent_text_pipeline.py --mode writer_only
  -> writer records

writer records
  -> classify_s1_s2.py
  -> clean_scripts.py
  -> hallucination_filter.py
  -> hallucination_repair_loop.py
  -> cleaned/repaired writer records

cleaned/repaired writer records
  -> voiceagent_text_pipeline.py --mode director_only
  -> 带 [THINK]、[WAIT]、[PENDXS]、^、[CUT] 和 S2 `「...」` 的 director messages

director messages
  -> synthesize_align_tts_to_parquet.py
  -> build_training_parquet.py
  -> strip_self_audio_from_parquet.py
  -> training parquet

training parquet
  -> extract_mimi_codes_parquet.py
  -> inject_noise_parquet.py
  -> final E2E training parquet
```

视频分支复用相同的 writer/director 和音频/parquet 思路，但增加视频事件标注：

```text
video files + metadata
  -> auto_tag_nextqa_events.py / batch_tag_nextqa_videos.py / batch_tag_llava_videos.py
  -> clean_videochat.py
  -> voiceagent_video_pipeline.py
  -> video writer/director records
  -> extract_mimi_codes_video_parquet.py
```

### 📚 目录指南

#### `data_pipeline/00_seed_configs`

包含以下 seed family：

- `seed_big`
- `seed_bc_big`
- `seed_xp_big`
- `seed_sparse`
- `seed_emotion`
- `seed_video`

每个 family 按需包含 `config.json`、`guidance.json` 以及对应的生成/清洗代码。生成后的 `samples*.jsonl` 被排除在仓库之外。

#### `data_pipeline/01_content`

内容转换脚本：

- `clean_all_datasets.py`：多个公开文本数据集的统一清洗入口；
- `clean_ultrachat.py`：UltraChat 风格数据转换；
- `clean_videochat.py`：视频标签到对话内容的转换。

#### `data_pipeline/02_writer_director`

主入口：

```text
voiceagent_text_pipeline.py
```

该脚本支持 `writer_only`、`director_only` 和 `full_pipeline`，支持可恢复输出、inbound/outbound split、多 provider OpenAI-compatible endpoint，以及 DeepSeek/Qwen 风格的 thinking request body。

#### `data_pipeline/03_between_writer_and_director`

Writer 与 Director 之间的质量控制和修复：

- `classify_s1_s2.py`：分类 S1/S2 样本；
- `clean_scripts.py`：自然化并规范化 writer 脚本；
- `hallucination_filter.py`：检测缺少上下文导致的幻觉模式；
- `hallucination_repair_loop.py`：迭代修复并复查。

#### `data_pipeline/04_tts_parquet`

音频合成、对齐和 parquet 转换：

- `synthesize_align_tts_to_parquet.py`
- `build_training_parquet.py`
- `strip_self_audio_from_parquet.py`

#### `data_pipeline/05_mimi_codec`

为文本/音频 E2E 监督提取 Mimi code：

- `config.py`
- `extract_mimi_codes_parquet.py`
- `extract_mimi_codes_video_parquet.py`

#### `data_pipeline/06_noise`

噪声生成和注入：

- `build_background_noise.py`
- `inject_noise_parquet.py`

#### `training_framework/qwen3_omni_training`

训练框架 fork，包含 Qwen3-Omni E2E 改动、Megatron trainer 改动、训练 recipe 和数据读取 schema。该目录作为框架资产发布，保留上游 examples 和文档。

`training_framework/Megatron-LM-core_v0.15.0` 与训练 fork 一起打包，训练时应通过 `MEGATRON_LM_PATH` 暴露给训练脚本。

#### `inference_framework`

- `realtime_serving`：thinker/talker/orchestrator 服务栈和实时客户端；
- `vllm_qwen3_omni`：用于服务、导出和运行流程的修改版 vLLM fork。

### 🧰 运行依赖

具体环境取决于你要运行哪个阶段。最小依赖包括：

- Python 3.10+
- `openai`
- `tqdm`
- `numpy`
- `pyarrow`
- `torch`
- `transformers`

先安装轻量依赖文件：

```bash
pip install -r requirements.txt
```

分阶段可选依赖：

- Content/video：`pandas` 或 `pyarrow`、`ffprobe`、视频解码库；
- TTS/audio：Qwen3-TTS 依赖、`scipy`、`soundfile` 或对应音频栈；
- Noise：可选 `pedalboard`，部分路径有 numpy fallback；
- Inference：修改版 `vllm_qwen3_omni`、FastAPI/Uvicorn、支持 CUDA 的 GPU。

### ⚙️ 配置

内部路径和 key 已替换成占位符。请显式配置真实值，并把这些变量指向 Hugging Face 仓库或从这些仓库下载得到的本地目录。

外部数据集/checkpoint 仓库占位符：

```bash
export HF_DATASET_REPO="<duplexomni-dataset-repo>"
export HF_THINKER_MODEL_REPO="<duplexomni-thinker-model-repo>"
export HF_TALKER_MODEL_REPO="<duplexomni-talker-model-repo>"
export HF_TTS_MODEL_REPO="<tts-model-repo-if-needed>"
```

下载 Hugging Face 资产后的典型本地布局：

```text
data/
  seed/
  content/
models/
  duplexomni-thinker/
  duplexomni-talker/
  qwen3-tts-custom/
  qwen3-tts-base/
  qwen3-forced-aligner/
  mimi/
outputs/
  pipeline_text/
  tts/
  parquet/
```

外部资产仓库可用后，可使用 `huggingface-cli download` 或你自己的 artifact manager 填充这些目录，再把本地路径传给下面的脚本。公开内容源 URL 见 [`EXTERNAL_ASSETS.md`](EXTERNAL_ASSETS.md)。

常用 OpenAI-compatible 设置：

```bash
export API_KEY=EMPTY
export API_BASES=http://localhost:8000/v1
export MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
```

数据集和产物根目录：

```bash
export DATASETS_ROOT=/path/to/public/datasets
export VOICEAGENT_ROOT=/path/to/workspace
```

TTS/Mimi：

```bash
export QWEN3_TTS_CUSTOM_MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice
export QWEN3_TTS_BASE_MODEL=/path/to/Qwen3-TTS-12Hz-1.7B-Base
export QWEN3_ALIGNER_MODEL=/path/to/Qwen3-ForcedAligner-0.6B
export MIMI_PATH=/path/to/mimi
```

实时服务：

```bash
export VLLM_ROOT=/path/to/open_source/inference_framework/vllm_qwen3_omni
export THINKER_MODEL=/path/to/qwen3-omni-thinker-checkpoint
export TALKER_MODEL=/path/to/qwen3-omni-talker-checkpoint
export S1_MODEL_NAME=/path/to/qwen3-omni-checkpoint
export S1_API_KEY=EMPTY
export S2_THINK_BASE_URL=http://localhost:8000/v1
export S2_MODEL_NAME=/path-or-name/of-thinking-model
export S2_API_KEY=EMPTY
```

### 🚦 基础使用

除非特别说明，以下命令都从 `open_source/` 目录运行。

#### 1. 检查仓库状态

```bash
git status --short
```

该命令会显示当前工作区是否有未提交改动。生成数据、日志、parquet、checkpoint 和模型权重不应进入仓库。

#### 2. 生成并清洗 seed samples

以一个 seed family 为例：

```bash
cd data_pipeline/00_seed_configs/seed_big
python3 generate_slot_permutations.py \
  --sample-limit 10000 \
  --output-jsonl samples.jsonl

python3 clean_samples_with_vllm.py \
  --input-jsonl samples.jsonl \
  --output-jsonl samples.cleaned.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --model gpt-4.1
```

对需要使用的 seed family 重复上述步骤。生成的 `samples.cleaned.jsonl` 应放入外部 Hugging Face 数据仓库或本地数据目录，不应提交到代码仓库。

#### 3. 准备 content jsonl

示例：

```bash
python3 data_pipeline/01_content/clean_ultrachat.py \
  --input-jsonl /path/to/ultrachat/train_sft.jsonl \
  --output-jsonl data/content/train_sft.voiceagent.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --model gpt-4.1
```

多数据集清洗时，配置 `DATASETS_ROOT` 并使用 `clean_all_datasets.py`：

```bash
DATASETS_ROOT=/path/to/public/datasets \
python3 data_pipeline/01_content/clean_all_datasets.py \
  --dataset all \
  --split both \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --model gpt-4.1
```

#### 4. 运行 writer/director

单个本地 provider 示例：

```bash
python3 data_pipeline/02_writer_director/voiceagent_text_pipeline.py \
  --mode full_pipeline \
  --split both \
  --api-key EMPTY \
  --api-base http://localhost:8000/v1 \
  --model-name deepseek-ai/DeepSeek-V4-Flash \
  --writer-provider-api-bases local=http://localhost:8000/v1 \
  --director-provider-api-bases local=http://localhost:8000/v1 \
  --seed-path data/seed/samples.cleaned.jsonl \
  --inbound-path data/content/inbound.jsonl \
  --outbound-path data/content/outbound.jsonl \
  --output-dir outputs/pipeline_text
```

多 provider 格式：

```bash
--director-provider-api-bases p0=http://host0:8000/v1,p1=http://host1:8000/v1
--director-provider-models p0=deepseek-ai/DeepSeek-V4-Flash,p1=deepseek-ai/DeepSeek-V4-Flash
```

#### 5. 运行 writer/director 中间清洗

```bash
export API_BASES=http://localhost:8000/v1
export API_KEY=EMPTY
export MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash

python3 data_pipeline/03_between_writer_and_director/classify_s1_s2.py \
  --input outputs/pipeline_text/writer.jsonl \
  --output outputs/pipeline_text/writer.s1s2.jsonl

python3 data_pipeline/03_between_writer_and_director/clean_scripts.py \
  --input outputs/pipeline_text/writer.s1s2.jsonl \
  --output outputs/pipeline_text/writer.cleaned.jsonl

python3 data_pipeline/03_between_writer_and_director/hallucination_filter.py \
  --input outputs/pipeline_text/writer.cleaned.jsonl

python3 data_pipeline/03_between_writer_and_director/hallucination_repair_loop.py \
  --input outputs/pipeline_text/writer.cleaned.halluc.jsonl \
  --output outputs/pipeline_text/writer.repaired.jsonl
```

不同脚本的参数名可能随实现略有变化，请以 `--help` 输出为准。

#### 6. TTS、parquet、Mimi、noise

```bash
python3 data_pipeline/04_tts_parquet/synthesize_align_tts_to_parquet.py \
  --input-path outputs/pipeline_text/director.jsonl \
  --output-dir outputs/tts/output_dialogue_e2e \
  --cut-dir outputs/tts/finalcut_sessions

python3 data_pipeline/04_tts_parquet/build_training_parquet.py \
  --input-parquet-dir outputs/tts/finalcut_sessions \
  --output-dir outputs/parquet/training_v7

python3 data_pipeline/05_mimi_codec/extract_mimi_codes_parquet.py \
  --input-parquet-dir outputs/parquet/training_v7 \
  --output-dir outputs/parquet/training_v7_codec \
  --mimi-path /path/to/mimi

python3 data_pipeline/04_tts_parquet/strip_self_audio_from_parquet.py \
  --input-dir outputs/parquet/training_v7_codec \
  --output-dir outputs/parquet/training_v7_codec_noself

python3 data_pipeline/06_noise/inject_noise_parquet.py \
  --input-dir outputs/parquet/training_v7_codec_noself \
  --output-dir outputs/parquet/training_v7_codec_noself_noised
```

部分历史脚本仍保留旧管线布局的默认值。公开运行时请通过 CLI 参数或环境变量明确指向你的本地路径。

#### 7. 视频分支

```bash
python3 data_pipeline/07_video_branch/batch_tag_nextqa_videos.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --model Qwen/Qwen3.5-397B-A17B

python3 data_pipeline/07_video_branch/clean_videochat.py \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY

python3 data_pipeline/07_video_branch/voiceagent_video_pipeline.py \
  --provider-api-bases local=http://localhost:8000/v1 \
  --model-name deepseek-ai/DeepSeek-V4-Flash
```

#### 8. 训练框架

训练框架位于：

```text
training_framework/qwen3_omni_training/
```

请使用该目录自己的 `README.md`、examples 和 Megatron/Qwen3-Omni E2E trainer 代码。训练期望的数据集是 data pipeline 产出的最终 parquet。

#### 9. 实时推理服务栈

启动 thinker/talker/orchestrator：

```bash
cd inference_framework/realtime_serving
export VLLM_ROOT=../vllm_qwen3_omni
export THINKER_MODEL=/path/to/thinker/checkpoint
export TALKER_MODEL=/path/to/talker/checkpoint
./start_thinker_talker_stack.sh
```

停止服务：

```bash
./stop_thinker_talker_stack.sh
```

本地仿真：

```bash
python3 simulate_v8.py
python3 simulate_video_v8.py
```

实时 WebSocket bridge 和 Mac 命令行客户端：

```bash
python3 omni_realtime_server.py --host 0.0.0.0 --port 8765
python3 omni_realtime_mac_client.py --server ws://127.0.0.1:8765
```

如果服务在远端机器上，可以直接连接 Mac 能访问到的远端地址：

```bash
python3 omni_realtime_mac_client.py --server ws://your-server-host:8765
```

也可以先把远端 8765 转发到 Mac 本地端口，再让客户端连接本地转发端口：

```bash
ssh -L 28765:127.0.0.1:8765 user@your-server
python3 omni_realtime_mac_client.py --server ws://127.0.0.1:28765
```

### 🧪 仓库卫生检查

公开分享仓库或运行完整管线前，请执行：

```bash
git status --short
find . -type f \( -name '*.pyc' -o -name '*.log' -o -name 'samples*.jsonl' \)
```

期望结果：

- `git status --short` 不显示生成数据、日志、媒体、parquet 或 cache 文件；
- 仓库内没有生成数据、日志、媒体、parquet 或 cache 文件；
- 仓库内没有内部 endpoint、私有 key、生产路径或模型 checkpoint 路径。

### 🤗 预期 Hugging Face 资产布局

预期外部资产布局：

```text
Hugging Face dataset repo
  seed/samples.cleaned.jsonl
  content/inbound.jsonl
  content/outbound.jsonl
  writer_director/director.jsonl
  parquet/training_v7_codec_noself_noised/
  video/parquet/

Hugging Face model repos
  duplexomni-thinker/
  duplexomni-talker/
  qwen3-tts-custom/
  mimi/ or documented external Mimi dependency
```

源码树使用 `data/...`、`models/...` 和 `outputs/...` 等占位路径。下载 Hugging Face 资产后，可以放在这些目录下，也可以通过显式 CLI 参数传入。

### ⚠️ 重要限制

- 本源码树不直接包含权重或数据集；这些资产预期以独立 Hugging Face 仓库或本地 artifact 目录提供。
- Hugging Face 数据集/模型仓库名称在公开资产上传完成前可能继续使用占位符。
- `localhost` endpoint 是 OpenAI-compatible 服务占位符。
- `models/...`、`data/...` 和 `outputs/...` 是占位路径。
- 大型框架目录（`training_framework` 和 `vllm_qwen3_omni`）保留上游 examples 和 tests；自定义数据和模型产物已排除。

### 📎 引用 / BibTeX

如果你在研究、产品、demo 或二次开源项目中使用 DuplexOmni，请引用论文：

```bibtex
@misc{huang2026duplexomnirealtimelisteningseeing,
      title={DuplexOmni: Real-Time Listening, Seeing, Thinking, and Speaking for Full-Duplex Interaction},
      author={Muye Huang and Lingling Zhang and Xingyu Yu and Lei Shi and Zhanyu Ma and Jun Xu and Jiuchong Gao and Jinghua Hao and Renqing He and Jun Liu},
      year={2026},
      eprint={2606.09186},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2606.09186},
}
```

## Third-Party Sources / 第三方来源

This repository vendors or adapts code from the following upstream projects:

- vLLM: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- ms-swift: [modelscope/ms-swift](https://github.com/modelscope/ms-swift)
- Megatron-LM: [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

This repository includes or adapts code from the upstream projects listed
above. Use, distribution, and redistribution must also comply with the
corresponding upstream licenses.

This directory was organized and released by GPT-5.5. If you have questions,
please open an [issue](https://github.com/MuyeHuang/DuplexOmni/issues). If you
are interested in this project, please consider giving it a star. The full
training data is about 9TB, so uploading the complete dataset is difficult;
thank you for your patience.

本仓库包含或改造了以上开源项目的代码。使用、分发或再发布时请同时遵循对应上游项目的许可证要求。

本目录由 GPT-5.5 整理并发布。有疑问请提
[issue](https://github.com/MuyeHuang/DuplexOmni/issues)；如果对本项目感兴趣，欢迎点一个
star，谢谢。训练数据约 9TB，完整上传比较困难，请耐心等待。
