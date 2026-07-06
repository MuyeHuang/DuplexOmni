# 06 Noise / 噪声增强

## English

Purpose: build background noise assets and inject user-side noise augmentation
into final training parquet. This improves robustness for real-time voice-agent
training.

Code role: `build_background_noise.py` creates optional preview/noise material;
`inject_noise_parquet.py` applies noise augmentation to parquet rows with
streaming parquet reads/writes, file-level concurrency, and row-level worker
parallelism.

Main files:

- `build_background_noise.py`: optional preview/background-noise construction
  utility using MUSAN/FSD50K-style local noise folders.
- `inject_noise_parquet.py`: parquet-level noise injection pipeline. It streams
  input parquet in batches, writes output parquet incrementally, can process
  multiple parquet files concurrently, and injects background noise into both
  normal user-audio chunks and RIFF/WAV silence chunks.

Usage: run after Mimi extraction or after the parquet layout you choose as the
final training input; keep noise source media and output parquet outside git.
Use MUSAN, FSD50K, or both. If neither source contains wav files, the scripts
raise an explicit error instead of silently writing un-noised data.

Noise sources:

- MUSAN: <https://www.openslr.org/17/>
- FSD50K: <https://zenodo.org/records/4060432>
- FSD50K companion site: <https://fsannotator.upf.edu/fsd/release/FSD50K/>

Expected layout:

```text
$NOISE_ROOT/
  MUSAN/raw/musan/noise/free-sound/*.wav
  MUSAN/raw/musan/noise/sound-bible/*.wav
  FSD50K/raw/FSD50K/FSD50K.dev_audio/*.wav
```

Example:

```bash
python data_pipeline/06_noise/inject_noise_parquet.py \
  --input-dir outputs/parquet/training_v7_mimi \
  --output-dir outputs/parquet/training_v7_mimi_noised \
  --musan-noise-dirs "$NOISE_ROOT/MUSAN/raw/musan/noise/free-sound,$NOISE_ROOT/MUSAN/raw/musan/noise/sound-bible" \
  --fsd50k-audio-dir "$NOISE_ROOT/FSD50K/raw/FSD50K/FSD50K.dev_audio" \
  --file-workers 4 \
  --workers 64 \
  --batch-rows 128 \
  --compression snappy
```

Useful controls:

- `--input-glob`: select parquet files under `--input-dir`, default
  `*.parquet`.
- `--file-workers`: number of parquet files processed concurrently.
- `--workers`: global row-level noise worker process count.
- `--batch-rows`: streaming parquet batch size.
- `--max-files`: smoke-test only the first N input files.
- `--resume`: skip output files that already exist.

Optional background-noise preview:

```bash
VOICEAGENT_ROOT=/path/to/workspace \
NOISE_ROOT=/path/to/noise \
python data_pipeline/06_noise/build_background_noise.py \
  --output-dir outputs/noise_preview
```

## 中文

目的：构造背景噪声资产，并把用户侧噪声增强注入最终训练 parquet，提高实时语音智能体在真实环境中的鲁棒性。

代码作用：`build_background_noise.py` 构造可选试听/噪声素材；`inject_noise_parquet.py` 对 parquet 行执行噪声增强，使用流式 parquet 读写、文件级并发和 row 级 worker 并行。

使用方法：在 Mimi 提取后，或在你选择的最终训练 parquet 布局后运行；噪声源媒体和输出 parquet 不进入 git。MUSAN、FSD50K 可二选一，也可同时使用。如果两个来源都找不到 wav，脚本会明确报错，不会静默产出“无噪声”数据。

噪声来源：

- MUSAN：<https://www.openslr.org/17/>
- FSD50K：<https://zenodo.org/records/4060432>
- FSD50K companion site：<https://fsannotator.upf.edu/fsd/release/FSD50K/>

期望目录布局：

```text
$NOISE_ROOT/
  MUSAN/raw/musan/noise/free-sound/*.wav
  MUSAN/raw/musan/noise/sound-bible/*.wav
  FSD50K/raw/FSD50K/FSD50K.dev_audio/*.wav
```

主要文件：

- `build_background_noise.py`：可选的背景噪声/试听构造工具，依赖本地 MUSAN/FSD50K 风格噪声目录。
- `inject_noise_parquet.py`：parquet 级噪声注入主流程。它按 batch 流式读取和写出 parquet，可以并发处理多个 parquet 文件，并会给普通 user-audio chunk 和 RIFF/WAV 静音 chunk 都铺背景噪。

噪声源、试听 wav 和增强后的 parquet 都属于数据资产，不进入代码仓库。运行时通过显式路径或 `VOICEAGENT_ROOT` 指向本地资产。
