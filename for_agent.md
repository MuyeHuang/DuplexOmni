# Guide For Automated Agents / 自动化 Agent 指南

## English

Purpose: give coding/runtime agents a compact contract for working with the
DuplexOmni open-source code tree without relying on private paths, private
services, or generated artifacts.

Code role: this file is not executable code. It is an operating guide for
agents that install dependencies, run validation, prepare data, train models, or
start local serving on behalf of a user.

Usage: read this file before modifying or running anything. Prefer explicit
paths supplied by the user or downloaded Hugging Face assets. Do not infer
private production paths.

## Repository Contract

- This is the source-code repository for DuplexOmni:
  <https://arxiv.org/abs/2606.09186>.
- The repository contains code only. Datasets, generated audio/video/parquet
  files, model weights, logs, and service endpoints live outside the source
  tree.
- Use `README.md` for the human overview, `INSTALL.md` for environments, and
  `DEPENDENCIES.md` for file-level dependencies.
- Use `CONCEPTS.md` for terminology and `EXTERNAL_ASSETS.md` for public dataset,
  checkpoint, TTS, Mimi, MUSAN, and FSD50K assets.
- Run commands from the `open_source/` directory unless a stage README says
  otherwise.
- Never write generated data, logs, media, parquet, caches, model checkpoints,
  or API keys into this repository.

## First Commands

```bash
cd /path/to/open_source
git status --short
git status --short
```

If validation fails, report the exact failing paths before changing files.

## External Assets

Full pipeline runs require these assets outside the source tree:

```text
data/seed/*.cleaned.jsonl
data/content/inbound.jsonl
data/content/outbound.jsonl
outputs/pipeline_text/*.jsonl
outputs/parquet/final_training_parquet/
models/base-qwen3-omni/
models/duplexomni-thinker/
models/duplexomni-talker/
models/qwen3-tts/
models/mimi/
```

Optional asset repository variables:

```bash
export HF_DATASET_REPO="<duplexomni-dataset-repo>"
export HF_THINKER_MODEL_REPO="<duplexomni-thinker-model-repo>"
export HF_TALKER_MODEL_REPO="<duplexomni-talker-model-repo>"
```

Do not treat placeholder repository names as real download targets. Use
`EXTERNAL_ASSETS.md` for the current public content dataset URLs.

## Environment Variables

Use placeholders until the user supplies real local paths:

```bash
export API_KEY=EMPTY
export API_BASES=http://localhost:8000/v1
export MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
export DATASETS_ROOT=/path/to/public/datasets
export VLLM_ROOT=/path/to/open_source/inference_framework/vllm_qwen3_omni
export SWIFT_SRC=/path/to/open_source/training_framework/qwen3_omni_training
export MEGATRON_LM_PATH=/path/to/open_source/training_framework/Megatron-LM-core_v0.15.0
export MIMI_PATH=/path/to/mimi
```

## Safe Execution Order

1. Validate the repository tree.
2. Install only the environment needed for the requested stage.
3. Download or locate external Hugging Face assets.
4. Run seed/content preparation if raw public data is provided.
5. Run writer, then quality cleaning/repair, then director.
6. Run TTS/alignment, parquet construction, Mimi codec extraction, and noise.
7. Train thinker first.
8. Train talker from the saved thinker checkpoint.
9. Start realtime serving with thinker/talker checkpoints.

## Key Entrypoints

```text
data_pipeline/02_writer_director/voiceagent_text_pipeline.py
data_pipeline/07_video_branch/voiceagent_video_pipeline.py
data_pipeline/04_tts_parquet/synthesize_align_tts_to_parquet.py
data_pipeline/04_tts_parquet/build_training_parquet.py
data_pipeline/05_mimi_codec/extract_mimi_codes_parquet.py
data_pipeline/06_noise/inject_noise_parquet.py
training_framework/qwen3_omni_training/
training_framework/Megatron-LM-core_v0.15.0/
inference_framework/realtime_serving/
inference_framework/vllm_qwen3_omni/
```

## Agent Safety Rules

- Do not bulk-scan user disks. Stay under `open_source/` unless the user gives
  an explicit asset path.
- Do not replace placeholders with private paths in committed files.
- Do not commit generated artifacts.
- Do not silently skip failed samples in the data pipeline.
- Do not assume a remote endpoint is public; treat all service URLs as user
  configuration.
- Prefer `python <script> --help` before running a stage with real inputs.

## 中文

目的：给自动化编码/运行 Agent 一个简洁约定，帮助它在没有私有路径、私有服务和生成产物的前提下操作 DuplexOmni 开源代码树。

代码作用：本文不是可执行代码，而是给 Agent 的操作说明，用于安装依赖、运行自检、准备数据、训练模型或启动本地服务。

使用方法：改动或运行任何命令前先读本文。优先使用用户显式提供的路径或 Hugging Face 下载资产，不要猜测内部生产路径。

## 仓库约定

- 这是 DuplexOmni 的开源代码仓库：<https://arxiv.org/abs/2606.09186>。
- 仓库只放代码；数据集、生成音频/视频/parquet、模型权重、日志、服务 endpoint 都在源码树之外。
- 人类总览看 `README.md`，安装看 `INSTALL.md`，文件级依赖看 `DEPENDENCIES.md`。
- 术语看 `CONCEPTS.md`，公开数据源、checkpoint、TTS、Mimi、MUSAN 和 FSD50K 资产看 `EXTERNAL_ASSETS.md`。
- 除非阶段 README 特别说明，命令都从 `open_source/` 目录执行。
- 不要把生成数据、日志、媒体、parquet、缓存、checkpoint 或 API key 写进仓库。

## 第一步命令

```bash
cd /path/to/open_source
git status --short
git status --short
```

如果自检失败，先报告失败路径，再决定是否修改文件。

## 外部资产

完整 pipeline 运行需要在源码树之外准备这些资产：

```text
data/seed/*.cleaned.jsonl
data/content/inbound.jsonl
data/content/outbound.jsonl
outputs/pipeline_text/*.jsonl
outputs/parquet/final_training_parquet/
models/base-qwen3-omni/
models/duplexomni-thinker/
models/duplexomni-talker/
models/qwen3-tts/
models/mimi/
```

可选的资产仓库变量：

```bash
export HF_DATASET_REPO="<duplexomni-dataset-repo>"
export HF_THINKER_MODEL_REPO="<duplexomni-thinker-model-repo>"
export HF_TALKER_MODEL_REPO="<duplexomni-talker-model-repo>"
```

不要把占位仓库名当成真实下载地址。当前公开 content 数据集地址见 `EXTERNAL_ASSETS.md`。

## 环境变量

用户给真实路径前，使用占位符：

```bash
export API_KEY=EMPTY
export API_BASES=http://localhost:8000/v1
export MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
export DATASETS_ROOT=/path/to/public/datasets
export VLLM_ROOT=/path/to/open_source/inference_framework/vllm_qwen3_omni
export SWIFT_SRC=/path/to/open_source/training_framework/qwen3_omni_training
export MEGATRON_LM_PATH=/path/to/open_source/training_framework/Megatron-LM-core_v0.15.0
export MIMI_PATH=/path/to/mimi
```

## 安全执行顺序

1. 先验证仓库树。
2. 只安装当前任务需要的环境。
3. 下载或定位外部 Hugging Face 资产。
4. 如有原始公开数据，运行 seed/content 准备。
5. 运行 writer，再做质量清洗/修复，再运行 director。
6. 运行 TTS/alignment、parquet 构建、Mimi codec 提取和噪声注入。
7. 先训练 thinker。
8. 从保存的 thinker checkpoint 继续训练 talker。
9. 用 thinker/talker checkpoint 启动实时服务。

## 关键入口

```text
data_pipeline/02_writer_director/voiceagent_text_pipeline.py
data_pipeline/07_video_branch/voiceagent_video_pipeline.py
data_pipeline/04_tts_parquet/synthesize_align_tts_to_parquet.py
data_pipeline/04_tts_parquet/build_training_parquet.py
data_pipeline/05_mimi_codec/extract_mimi_codes_parquet.py
data_pipeline/06_noise/inject_noise_parquet.py
training_framework/qwen3_omni_training/
training_framework/Megatron-LM-core_v0.15.0/
inference_framework/realtime_serving/
inference_framework/vllm_qwen3_omni/
```

## Agent 安全规则

- 不要大规模扫描用户磁盘；除非用户明确给路径，否则只在 `open_source/` 内操作。
- 不要把私有路径写进提交文件。
- 不要提交生成产物。
- 数据管线失败样本不能静默跳过。
- 不要假设远端 endpoint 是公共服务；所有服务地址都视为用户配置。
- 真实运行某阶段前，优先执行 `python <script> --help`。
