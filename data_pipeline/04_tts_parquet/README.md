# 04 TTS Parquet / 语音合成与 Parquet

## English

Purpose: convert director messages into aligned audio sessions and training
parquet files. This stage synthesizes speech, aligns text/audio, cuts sessions,
and prepares parquet shards for E2E training.

Code role: these scripts run Qwen3-TTS, forced alignment, finalcut session
assembly, parquet packing, and self-audio stripping.

Main files:

- `synthesize_align_tts_to_parquet.py`: Qwen3-TTS generation, forced
  alignment, session cutting, and intermediate parquet/session outputs.
- `build_training_parquet.py`: convert finalcut session output into the
  training parquet schema.
- `strip_self_audio_from_parquet.py`: remove
  self-audio fields before the final downstream training/noise layout.

Usage: pass explicit model paths and output directories; treat generated audio
and parquet as dataset assets, not source files.

Example:

```bash
python data_pipeline/04_tts_parquet/synthesize_align_tts_to_parquet.py \
  --input-path outputs/pipeline_text/director.jsonl \
  --output-dir outputs/tts/output_dialogue_e2e \
  --cut-dir outputs/tts/finalcut_sessions \
  --custom-model-path models/qwen3-tts-custom \
  --base-model-path models/qwen3-tts-base \
  --aligner-model-path models/qwen3-forced-aligner \
  --resume

python data_pipeline/04_tts_parquet/build_training_parquet.py \
  --input-parquet-dir outputs/tts/finalcut_sessions \
  --output-dir outputs/parquet/training_v7

python data_pipeline/04_tts_parquet/strip_self_audio_from_parquet.py \
  --input-dir outputs/parquet/training_v7 \
  --output-dir outputs/parquet/training_v7_noself \
  --resume
```

## 中文

目的：把 director messages 转成对齐后的音频 session 和训练 parquet。本阶段负责 TTS 合成、强制对齐、session 切分，以及 E2E 训练所需 parquet shard。

代码作用：这些脚本运行 Qwen3-TTS、forced alignment、finalcut session 组装、parquet 打包和 self-audio 移除。

使用方法：显式传入模型路径和输出目录；生成音频与 parquet 属于数据资产，不提交到源码仓库。

主要文件：

- `synthesize_align_tts_to_parquet.py`：Qwen3-TTS 生成、forced aligner 对齐、session 切分和中间 parquet/session 输出。
- `build_training_parquet.py`：把 finalcut session 输出转换为训练 parquet schema。
- `strip_self_audio_from_parquet.py`：移除 self-audio 字段，得到后续训练/噪声增强使用的布局。

运行时需要显式指定 TTS custom/base 模型、forced aligner 模型、输入 director JSONL 和输出目录。生成音频和 parquet 属于数据资产，不应提交到代码仓库。
