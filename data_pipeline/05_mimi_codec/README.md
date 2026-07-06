# 05 Mimi Codec / Mimi 编码

## English

Purpose: extract Mimi codec tokens/features from training parquet audio fields
for Qwen3-Omni E2E supervision.

Code role: the extraction scripts stream parquet shards, batch audio by GPU, run
the Mimi encoder, and write codec features back to parquet.

Main files:

- `config.py`: local path and codec constants. Configure real paths with
  environment variables such as `MIMI_PATH`, `TRAIN_JSONL`, and
  `CHECKPOINT_DIR`.
- `extract_mimi_codes_parquet.py`: main text/audio parquet
  Mimi extraction pipeline.
- `extract_mimi_codes_video_parquet.py`: video-branch variant.

Usage: run after `04_tts_parquet`, set `--mimi-path`, and choose GPU/batch
settings according to local hardware.

Example:

```bash
python data_pipeline/05_mimi_codec/extract_mimi_codes_parquet.py \
  --input-parquet-dir outputs/parquet/training_v7_noself \
  --output-dir outputs/parquet/training_v7_mimi \
  --mimi-path models/mimi \
  --n-gpus 8 \
  --resume
```

Video branch:

```bash
python data_pipeline/05_mimi_codec/extract_mimi_codes_video_parquet.py \
  --input-parquet-dir outputs/video/parquet/training_v7_noself \
  --output-dir outputs/video/parquet/training_v7_mimi \
  --mimi-path models/mimi \
  --n-gpus 8 \
  --resume
```

## 中文

目的：从训练 parquet 的音频字段中提取 Mimi codec token/feature，供 Qwen3-Omni E2E 音频监督使用。

代码作用：提取脚本流式读取 parquet shard，按 GPU 批量处理音频，运行 Mimi encoder，并把 codec feature 写回 parquet。

使用方法：在 `04_tts_parquet` 后运行，设置 `--mimi-path`，并根据本地硬件选择 GPU 和 batch 参数。

主要文件：

- `config.py`：本地路径和 codec 常量。真实路径通过 `MIMI_PATH`、`TRAIN_JSONL`、`CHECKPOINT_DIR` 等环境变量配置。
- `extract_mimi_codes_parquet.py`：主线文本/音频 parquet 的 Mimi 提取流程。
- `extract_mimi_codes_video_parquet.py`：视频分支版本。

输出 parquet 属于数据资产，不进入代码仓库。运行前确认 Mimi 模型路径、GPU 数量和输入 parquet schema。
