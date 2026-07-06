# 02 Writer Director / 编剧与导演

## English

Purpose: this is the authoritative main entrypoint for text-side generation.
`voiceagent_text_pipeline.py` first writes natural scripts
from content + seed guidance, then converts scripts into director messages with
full duplex timing and reasoning tags.

Code role: writer mode generates playable scripts; director mode translates
those scripts into supervised messages containing timing, interruption, wait,
cut, pending, and System-2 tags.

Main files:

- `voiceagent_text_pipeline.py`: main writer/director
  pipeline.
- `api_client.py`: small OpenAI-compatible API wrapper.

Key modes:

- `--mode writer_only`: generate writer scripts.
- `--mode director_only`: generate director records from existing writer data.
- `--mode full_pipeline`: run both stages in one invocation.

Key inputs:

- `--seed-path`: comma-separated cleaned seed JSONL files.
- `--inbound-path`: inbound content JSONL.
- `--outbound-path`: outbound content JSONL.
- `--output-dir`: resumable pipeline output directory.

Usage: run `writer_only`, run the quality stage in `03_between_writer_and_director`,
then resume with `director_only`; use `full_pipeline` only for small or direct
end-to-end runs.

Single-provider example:

```bash
python data_pipeline/02_writer_director/voiceagent_text_pipeline.py \
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

Multi-provider example:

```bash
python data_pipeline/02_writer_director/voiceagent_text_pipeline.py \
  --mode director_only \
  --split both \
  --director-provider-api-bases p0=http://host0:8000/v1,p1=http://host1:8000/v1 \
  --director-provider-models p0=deepseek-ai/DeepSeek-V4-Flash,p1=deepseek-ai/DeepSeek-V4-Flash \
  --director-provider-workers 200 \
  --director-provider-rpm 0 \
  --output-dir outputs/pipeline_text
```

## 中文

目的：这是文本侧生成的权威主入口。`voiceagent_text_pipeline.py` 先用 content + seed guidance 生成自然剧本，再把剧本转换成带全双工时序和推理标签的 director messages。

代码作用：writer 模式生成可播放剧本；director 模式把剧本转换为带时序、打断、等待、截断、pending 和 System-2 标签的监督消息。

使用方法：先运行 `writer_only`，再运行 `03_between_writer_and_director` 质量处理，最后用 `director_only` 断点续跑；小规模端到端可以使用 `full_pipeline`。

主要文件：

- `voiceagent_text_pipeline.py`：主线 writer/director pipeline。
- `api_client.py`：OpenAI-compatible API 封装。

关键模式：

- `--mode writer_only`：只生成 writer 剧本。
- `--mode director_only`：基于已有 writer 结果生成 director 记录。
- `--mode full_pipeline`：一次运行 writer + director。

关键输入：

- `--seed-path`：逗号分隔的 cleaned seed JSONL。
- `--inbound-path`：inbound content JSONL。
- `--outbound-path`：outbound content JSONL。
- `--output-dir`：支持断点续传的输出目录。

输出会写入 `--output-dir` 下的 inbound/outbound writer/director JSONL。pipeline 会拒绝静默完成；如存在失败任务，会报错退出，便于后续重跑补齐。
