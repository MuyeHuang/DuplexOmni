# Data Pipeline / 数据管线

## English

Purpose: define the end-to-end source data construction DAG for the public
DuplexOmni release.

Code role: each numbered subdirectory owns one stage of the dataset build, from
seed/content preparation to writer/director generation, audio conversion, codec
extraction, and noise augmentation.

This directory contains the authoritative data pipeline for building full duplex
DuplexOmni training data. It starts from seed guidance and public content,
generates writer scripts, cleans and repairs them, runs director annotation, and
then converts the result into audio/parquet/Mimi/noise training assets.

Usage: run the numbered stages in order and pass explicit local paths for
datasets, models, endpoints, and outputs.

Run stages in numeric order:

```text
00_seed_configs
  -> 01_content
  -> 02_writer_director --mode writer_only
  -> 03_between_writer_and_director
  -> 02_writer_director --mode director_only
  -> 04_tts_parquet
  -> 05_mimi_codec
  -> 06_noise
```

`07_video_branch` is an optional branch for video-grounded dialogue. It shares
the same writer/director and downstream audio/parquet design, but adds video
event tagging before dialogue generation.

Data, weights, generated JSONL, audio, parquet, and logs are intentionally not
stored here. Use Hugging Face dataset/model repositories for those assets and
pass local paths through CLI arguments.

## 中文

目的：定义公开 DuplexOmni 发布所需的端到端数据构建 DAG。

代码作用：每个数字子目录负责一个数据构建阶段，从 seed/content 准备，到 writer/director 生成、音频转换、codec 提取和噪声增强。

使用方法：按数字目录顺序运行，并显式传入本地数据、模型、endpoint 和输出路径。

本目录是全双工 DuplexOmni 训练数据的权威数据管线。流程从 seed guidance 和公开 content 开始，生成 writer 剧本，经过清洗修复后进入 director 标注，再转换为音频、parquet、Mimi codec 和噪声增强后的训练资产。

主线请按数字目录顺序运行：

```text
00_seed_configs
  -> 01_content
  -> 02_writer_director --mode writer_only
  -> 03_between_writer_and_director
  -> 02_writer_director --mode director_only
  -> 04_tts_parquet
  -> 05_mimi_codec
  -> 06_noise
```

`07_video_branch` 是视频 grounded 对话的小分支。它复用主线 writer/director 和后续音频/parquet 设计，但在对话生成前增加视频事件标注。

这里不保存数据、权重、生成 JSONL、音频、parquet 或日志。开源发布时应把这些资产放到 Hugging Face dataset/model repo，再通过命令行参数传入本地路径。

## Shared Runtime / 通用运行约定

```bash
export API_KEY=EMPTY
export API_BASES=http://localhost:8000/v1
export MODEL_NAME=deepseek-ai/DeepSeek-V4-Flash
export DATASETS_ROOT=/path/to/public/datasets
export VOICEAGENT_ROOT=/path/to/workspace
```

Most scripts expose `--help`. Prefer explicit CLI paths over defaults when
running outside the original production environment.

多数脚本支持 `--help`。在开源环境中运行时，优先显式传入路径参数，不依赖历史默认路径。
