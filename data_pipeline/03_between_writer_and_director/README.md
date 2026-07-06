# 03 Between Writer And Director / Writer 与 Director 之间的质量处理

## English

Purpose: clean, classify, filter, and repair writer scripts before they enter
director annotation. This stage improves script naturalness and reduces
hallucinated facts that the director would otherwise preserve.

Code role: the scripts assign S1/S2 labels, normalize the writer text, detect
hallucination patterns, and iteratively repair failed samples.

Main files:

- `classify_s1_s2.py`: label samples as S1 or S2.
- `clean_scripts.py`: normalize writer scripts and improve conversational
  quality, optionally using S1/S2 labels.
- `hallucination_filter.py`: detect likely missing-context hallucinations.
- `hallucination_repair_loop.py`: repair hallucinated samples and recheck them.

Usage: run this stage after `writer_only` and before `director_only`. Keep all
intermediate JSONL outputs so failed or repaired samples can be audited.

Typical order:

```bash
python data_pipeline/03_between_writer_and_director/classify_s1_s2.py \
  --input outputs/pipeline_text/inbound.writer.jsonl \
  --output outputs/pipeline_text/inbound.s1s2.jsonl

python data_pipeline/03_between_writer_and_director/clean_scripts.py \
  --input outputs/pipeline_text/inbound.writer.jsonl \
  --output outputs/pipeline_text/inbound.writer.cleaned.jsonl \
  --s1s2-labels outputs/pipeline_text/inbound.s1s2.jsonl \
  --max-workers 200

python data_pipeline/03_between_writer_and_director/hallucination_filter.py \
  --input outputs/pipeline_text/inbound.writer.cleaned.jsonl \
  --output outputs/pipeline_text/inbound.writer.halluc.jsonl \
  --max-workers 200

python data_pipeline/03_between_writer_and_director/hallucination_repair_loop.py \
  --input outputs/pipeline_text/inbound.writer.halluc.jsonl \
  --output outputs/pipeline_text/inbound.writer.repaired.jsonl \
  --max-workers 200 \
  --max-rounds 5
```

These scripts use OpenAI-compatible endpoints configured by environment
variables in the code or by your local wrapper. Run `--help` before a full run
and keep output paths explicit.

## 中文

目的：在 writer 剧本进入 director 标注前，做分类、清洗、幻觉检测和修复。这个阶段用于提高剧本自然度，并减少 director 会继承下去的缺上下文幻觉。

代码作用：这些脚本负责 S1/S2 标签、writer 文本规范化、幻觉模式检测，以及失败样本的循环修复。

使用方法：在 `writer_only` 后、`director_only` 前运行；保留中间 JSONL，便于审计失败和修复样本。

主要文件：

- `classify_s1_s2.py`：将样本标为 S1 或 S2。
- `clean_scripts.py`：规范化 writer 剧本，提高对话质量，可接入 S1/S2 标签。
- `hallucination_filter.py`：检测疑似缺上下文幻觉。
- `hallucination_repair_loop.py`：对幻觉样本做循环修复和复查。

典型运行顺序见上方命令。完整运行前请先查看各脚本 `--help`，并显式传入输入输出路径和并发参数。
