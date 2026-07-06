# 07 Video Branch / 视频分支

## English

Purpose: generate video-grounded VoiceAgent data. This branch tags video events,
converts them into dialogue content, then runs a video-aware writer/director
pipeline. Most downstream audio/parquet/Mimi stages are shared with the main
pipeline.

Code role: the taggers extract structured video events, `clean_videochat.py`
turns them into content JSONL, and the multi-provider pipeline generates
video-grounded writer/director records.

Main files:

- `auto_tag_nextqa_events.py`: sample or auto-tag NextQA-style video events.
- `batch_tag_nextqa_videos.py`: batch tag NextQA videos with a vision-language
  OpenAI-compatible endpoint.
- `batch_tag_llava_videos.py`: batch tag LLaVA/ActivityNet/NextQA-style video
  folders.
- `clean_videochat.py`: convert video tags/content into dialogue JSONL.
- `voiceagent_video_pipeline.py`: multi-provider video
  writer/director pipeline.
- `video_pipeline_base.py`: legacy/shared video pipeline dependency.
- `api_client.py`: local API wrapper used by the video branch.

Usage: tag videos first, clean the tags into content JSONL, then run
`voiceagent_video_pipeline.py` and continue with the shared
audio/parquet/Mimi stages.

Example:

```bash
python data_pipeline/07_video_branch/batch_tag_nextqa_videos.py \
  --nextqa-root /path/to/NextQA \
  --video-root /path/to/NextQA/videos \
  --output-dir outputs/video/tags_nextqa \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --model Qwen/Qwen3.5-397B-A17B \
  --workers 100

python data_pipeline/07_video_branch/clean_videochat.py \
  --input-jsonl outputs/video/tags_nextqa/tags.jsonl \
  --output-jsonl data/content/videochat.voiceagent.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --model gpt-4.1

python data_pipeline/07_video_branch/voiceagent_video_pipeline.py \
  --provider-api-bases local=http://localhost:8000/v1 \
  --provider-workers 100 \
  --model-name deepseek-ai/DeepSeek-V4-Flash \
  --seed-path data/seed/video.samples.cleaned.jsonl \
  --inbound-path data/content/video.inbound.jsonl \
  --outbound-path data/content/video.outbound.jsonl \
  --output-dir outputs/video/pipeline
```

## 中文

目的：生成视频 grounded 的 VoiceAgent 数据。该分支先对视频事件打标签，再转成对话 content，然后运行视频感知的 writer/director pipeline。多数后续音频、parquet、Mimi 阶段与主线共用。

代码作用：tagger 提取结构化视频事件，`clean_videochat.py` 把事件转成 content JSONL，多 provider pipeline 生成视频 grounded writer/director 记录。

使用方法：先标注视频，再把标签清洗为 content JSONL，然后运行 `voiceagent_video_pipeline.py`，后续接共享的音频/parquet/Mimi 阶段。

主要文件：

- `auto_tag_nextqa_events.py`：对 NextQA 风格视频事件做抽样或自动标注。
- `batch_tag_nextqa_videos.py`：通过 OpenAI-compatible 视觉语言 endpoint 批量标注 NextQA 视频。
- `batch_tag_llava_videos.py`：批量标注 LLaVA/ActivityNet/NextQA 风格视频目录。
- `clean_videochat.py`：把视频标签/content 转成对话 JSONL。
- `voiceagent_video_pipeline.py`：多 provider 视频 writer/director pipeline。
- `video_pipeline_base.py`：视频分支 legacy/shared 依赖，不作为主线入口。
- `api_client.py`：视频分支使用的本地 API wrapper。

视频文件、抽帧、标签结果和视频 parquet 都属于数据资产，不进入代码仓库。运行时显式传入 `--video-root`、`--output-dir`、endpoint 和模型名。
