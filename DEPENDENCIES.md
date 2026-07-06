# Dependency Map / 依赖关系

## English

Purpose: document the dependency relationships between repository files so users
can install only the parts they need.

Code role: this file is a human-readable map from each pipeline/runtime stage to
Python packages, system tools, model assets, and upstream/downstream files.

Usage: read this together with `INSTALL.md` before running a stage. If a stage is
not used, its optional dependencies can usually be skipped.

```bash
cd open_source
git status --short
```

## Stage Dependency Matrix

| Stage | Main files | Python dependencies | System/model dependencies | Produces |
| --- | --- | --- | --- | --- |
| Seed configs | `00_seed_configs/*/generate_slot_permutations.py`, `clean_samples_with_vllm.py` | `openai`, `tqdm` through client usage, stdlib | OpenAI-compatible endpoint for cleaning | cleaned seed JSONL |
| Content cleaning | `01_content/clean_all_datasets.py`, `clean_ultrachat.py`, `clean_videochat.py` | `openai`, `pyarrow`, `pandas`, `tqdm` | public datasets under `DATASETS_ROOT`; see `EXTERNAL_ASSETS.md` | inbound/outbound content JSONL |
| Writer/director | `02_writer_director/voiceagent_text_pipeline.py`, `api_client.py` | `openai`, `httpx`, `tqdm` | OpenAI-compatible text model endpoints | writer/director JSONL |
| Quality repair | `03_between_writer_and_director/*.py` | `openai`, `tqdm` | OpenAI-compatible text model endpoints | cleaned/repaired writer JSONL |
| TTS/parquet | `04_tts_parquet/synthesize_align_tts_to_parquet.py`, `build_training_parquet.py`, `strip_self_audio_from_parquet.py` | `torch`, `numpy`, `pyarrow`, `pandas`, `tqdm` | Qwen3-TTS custom/base model, forced aligner, audio runtime | training parquet |
| Mimi codec | `05_mimi_codec/extract_mimi_codes_parquet.py`, `extract_mimi_codes_video_parquet.py` | `torch`, `transformers`, `librosa`, `soundfile`, `scipy`, `pyarrow` | Mimi checkpoint, GPUs | codec parquet |
| Noise | `06_noise/inject_noise_parquet.py`, `build_background_noise.py` | `numpy`, `pyarrow`, `pedalboard`, `tqdm` | MUSAN and/or FSD50K folders under `NOISE_ROOT`; see `EXTERNAL_ASSETS.md` | noised parquet |
| Video branch | `07_video_branch/*.py` | `openai`, `pandas`, `pyarrow`, `av`, `pillow`, `tqdm` | video files, `ffprobe`, vision-language endpoint | video writer/director JSONL |
| Realtime serving | `inference_framework/realtime_serving/*` | `fastapi`, `uvicorn`, `websockets`, `requests`, `torch`, `transformers`, local `vllm` | thinker/talker checkpoints, CUDA GPUs | local HTTP/WebSocket services |
| Training | `training_framework/qwen3_omni_training` + `training_framework/Megatron-LM-core_v0.15.0` | see `requirements.txt` and `requirements/*.txt` | final parquet, Qwen3-Omni checkpoints, GPUs, bundled `MEGATRON_LM_PATH` | trained checkpoints |

## File-Level Edges

```text
00_seed_configs/*/clean_samples_with_vllm.py
  -> data/seed/*.cleaned.jsonl
  -> 02_writer_director/voiceagent_text_pipeline.py --seed-path

01_content/*.py
  -> data/content/inbound.jsonl + data/content/outbound.jsonl
  -> 02_writer_director/voiceagent_text_pipeline.py --inbound-path/--outbound-path

02_writer_director/voiceagent_text_pipeline.py --mode writer_only
  -> writer JSONL
  -> 03_between_writer_and_director/*.py
  -> 02_writer_director/voiceagent_text_pipeline.py --mode director_only

02_writer_director/voiceagent_text_pipeline.py --mode director_only
  -> director JSONL
  -> 04_tts_parquet/synthesize_align_tts_to_parquet.py

04_tts_parquet/synthesize_align_tts_to_parquet.py
  -> 04_tts_parquet/build_training_parquet.py
  -> 04_tts_parquet/strip_self_audio_from_parquet.py
  -> 05_mimi_codec/extract_mimi_codes_parquet.py
  -> 06_noise/inject_noise_parquet.py

07_video_branch/voiceagent_video_pipeline.py
  -> video director JSONL
  -> 05_mimi_codec/extract_mimi_codes_video_parquet.py
```

## Runtime Import Notes

- `02_writer_director/voiceagent_text_pipeline.py` imports local `api_client.py`.
- `07_video_branch/voiceagent_video_pipeline.py` imports local `video_pipeline_base.py`.
- `06_noise/inject_noise_parquet.py` imports local `build_background_noise.py`.
- `inference_framework/realtime_serving/serving_core/server_talker.py` and
  `server_thinker.py` add `VLLM_ROOT` to `sys.path`; by default this resolves to
  `../vllm_qwen3_omni`.
- `training_framework/qwen3_omni_training` expects `SWIFT_SRC` to point at the
  local training fork and `MEGATRON_LM_PATH` to point at the bundled
  `training_framework/Megatron-LM-core_v0.15.0` checkout. The framework will
  otherwise try to initialize Megatron automatically.
- `training_framework/qwen3_omni_training` is an editable install of the local
  ms-swift fork and provides the `swift` Python package.

## External Assets

Generated DuplexOmni data and checkpoints are external assets. Public content
dataset URLs, MUSAN/FSD50K download sources, and expected local layouts are
listed in `EXTERNAL_ASSETS.md`.

## 中文

目的：记录仓库文件之间的依赖关系，方便用户只安装自己需要的部分。

代码作用：本文把每个 pipeline/runtime 阶段映射到 Python 包、系统工具、模型资产和上下游文件。

使用方法：运行某一阶段前，先结合 `INSTALL.md` 阅读本文件；不用的阶段通常可以跳过对应可选依赖。

```bash
cd open_source
git status --short
```

## 阶段依赖矩阵

| 阶段 | 主文件 | Python 依赖 | 系统/模型依赖 | 产物 |
| --- | --- | --- | --- | --- |
| Seed 配置 | `00_seed_configs/*/generate_slot_permutations.py`, `clean_samples_with_vllm.py` | `openai`、标准库 | 清洗用 OpenAI-compatible endpoint | cleaned seed JSONL |
| Content 清洗 | `01_content/clean_all_datasets.py`, `clean_ultrachat.py`, `clean_videochat.py` | `openai`, `pyarrow`, `pandas`, `tqdm` | `$DATASETS_ROOT` 下的公开数据集；见 `EXTERNAL_ASSETS.md` | inbound/outbound content JSONL |
| Writer/director | `02_writer_director/voiceagent_text_pipeline.py`, `api_client.py` | `openai`, `httpx`, `tqdm` | OpenAI-compatible 文本模型 endpoint | writer/director JSONL |
| 质量修复 | `03_between_writer_and_director/*.py` | `openai`, `tqdm` | OpenAI-compatible 文本模型 endpoint | cleaned/repaired writer JSONL |
| TTS/parquet | `04_tts_parquet/synthesize_align_tts_to_parquet.py`, `build_training_parquet.py`, `strip_self_audio_from_parquet.py` | `torch`, `numpy`, `pyarrow`, `pandas`, `tqdm` | Qwen3-TTS custom/base 模型、forced aligner、音频运行环境 | training parquet |
| Mimi codec | `05_mimi_codec/extract_mimi_codes_parquet.py`, `extract_mimi_codes_video_parquet.py` | `torch`, `transformers`, `librosa`, `soundfile`, `scipy`, `pyarrow` | Mimi checkpoint、GPU | codec parquet |
| 噪声 | `06_noise/inject_noise_parquet.py`, `build_background_noise.py` | `numpy`, `pyarrow`, `pedalboard`, `tqdm` | `NOISE_ROOT` 下的 MUSAN 和/或 FSD50K 目录；见 `EXTERNAL_ASSETS.md` | noised parquet |
| 视频分支 | `07_video_branch/*.py` | `openai`, `pandas`, `pyarrow`, `av`, `pillow`, `tqdm` | 视频文件、`ffprobe`、视觉语言 endpoint | video writer/director JSONL |
| 实时推理 | `inference_framework/realtime_serving/*` | `fastapi`, `uvicorn`, `websockets`, `requests`, `torch`, `transformers`, 本地 `vllm` | thinker/talker checkpoint、CUDA GPU | 本地 HTTP/WebSocket 服务 |
| 训练 | `training_framework/qwen3_omni_training` + `training_framework/Megatron-LM-core_v0.15.0` | 见 `requirements.txt` 与 `requirements/*.txt` | final parquet、Qwen3-Omni checkpoint、GPU、本仓库内的 `MEGATRON_LM_PATH` | 训练 checkpoint |

## 文件级依赖

```text
00_seed_configs/*/clean_samples_with_vllm.py
  -> data/seed/*.cleaned.jsonl
  -> 02_writer_director/voiceagent_text_pipeline.py --seed-path

01_content/*.py
  -> data/content/inbound.jsonl + data/content/outbound.jsonl
  -> 02_writer_director/voiceagent_text_pipeline.py --inbound-path/--outbound-path

02_writer_director/voiceagent_text_pipeline.py --mode writer_only
  -> writer JSONL
  -> 03_between_writer_and_director/*.py
  -> 02_writer_director/voiceagent_text_pipeline.py --mode director_only

02_writer_director/voiceagent_text_pipeline.py --mode director_only
  -> director JSONL
  -> 04_tts_parquet/synthesize_align_tts_to_parquet.py

04_tts_parquet/synthesize_align_tts_to_parquet.py
  -> 04_tts_parquet/build_training_parquet.py
  -> 04_tts_parquet/strip_self_audio_from_parquet.py
  -> 05_mimi_codec/extract_mimi_codes_parquet.py
  -> 06_noise/inject_noise_parquet.py

07_video_branch/voiceagent_video_pipeline.py
  -> video director JSONL
  -> 05_mimi_codec/extract_mimi_codes_video_parquet.py
```

## 运行时 import 说明

- `02_writer_director/voiceagent_text_pipeline.py` 导入同目录 `api_client.py`。
- `07_video_branch/voiceagent_video_pipeline.py` 导入同目录 `video_pipeline_base.py`。
- `06_noise/inject_noise_parquet.py` 导入同目录 `build_background_noise.py`。
- `inference_framework/realtime_serving/serving_core/server_talker.py` 和
  `server_thinker.py` 会把 `VLLM_ROOT` 加入 `sys.path`；默认解析到 `../vllm_qwen3_omni`。
- `training_framework/qwen3_omni_training` 需要 `SWIFT_SRC` 指向本地训练 fork，
  以及 `MEGATRON_LM_PATH` 指向本仓库内的
  `training_framework/Megatron-LM-core_v0.15.0` checkout；否则框架会尝试自动初始化 Megatron。
- `training_framework/qwen3_omni_training` 是本地 ms-swift fork 的 editable install，提供 `swift` Python 包。

## 外部资产

生成后的 DuplexOmni 数据和 checkpoint 是外部资产。公开 content 数据集地址、MUSAN/FSD50K 下载来源和期望本地布局见 `EXTERNAL_ASSETS.md`。
