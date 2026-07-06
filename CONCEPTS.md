# Concepts / 核心概念

## English

Purpose: define the DuplexOmni terms used across the data pipeline, training
framework, and realtime serving stack.

Code role: this document is a terminology contract for readers and automated
agents. It is not executable code.

Usage: read this before running the pipeline or interpreting generated records.
The terms below are used consistently by `data_pipeline`, `training_framework`,
and `inference_framework`.

## S1 And S2

S1 is the low-latency interaction path. It is responsible for continuous
listening, seeing, short response planning, rhythm control, and streaming
Assistant output.

S2 is the asynchronous thinking/tool path. It can be a stronger LLM, a
retrieval system, a tool agent, or any OpenAI-compatible endpoint that performs
slower reasoning in the background. S2 output is returned to S1 without stopping
the realtime interaction.

## Interaction Layer, Thinking Layer, Thinker, And Talker

The paper-level architecture has two layers:

- Interaction layer: the realtime DuplexOmni model that keeps receiving
  streaming audio/video and keeps producing speech/text.
- Thinking layer: a pluggable background reasoning or tool-use module.

The model-level Qwen-Omni components are different:

- Thinker: the internal multimodal language backbone that reads context and
  produces Assistant text tokens and hidden states.
- Talker: the internal speech generation component that converts Thinker
  states into streaming speech codec tokens and waveform.

Therefore, the paper's "thinking layer" is not the same thing as the model's
"Thinker" module.

## Inbound And Outbound

Inbound samples model user-initiated conversations: the user starts the
interaction and the Agent responds.

Outbound samples model Agent-initiated conversations: the Agent starts with a
task-oriented opening and the user responds.

Both splits are kept because realtime agents need to behave naturally in both
user-initiated and agent-initiated scenarios.

## Self-Audio

Self-audio means Assistant-side audio stored as context or reconstruction
material inside intermediate parquet records. It can be useful for alignment,
debugging, or downstream codec construction.

Before final E2E training, `strip_self_audio_from_parquet.py` removes the
self-audio fields so the model does not receive target-side audio as an input
shortcut.

## MTP

MTP means multi-token prediction in the Talker path. In this repository it refers
to the Talker-side module that predicts residual RVQ codec codebooks after the
layer-0 codec token. It is an internal model/runtime component, not an external
API service.

## 中文

目的：定义 DuplexOmni 数据管线、训练框架和实时服务栈中反复出现的术语。

代码作用：本文是给读者和自动化 Agent 的术语约定，不是可执行代码。

使用方法：运行 pipeline 或理解生成记录前先读本文。下面这些术语会在
`data_pipeline`、`training_framework` 和 `inference_framework` 中保持一致。

## S1 和 S2

S1 是低延迟 interaction path，负责连续听、看、短程响应规划、节奏控制，以及流式输出 Assistant 内容。

S2 是异步 thinking/tool path，可以是更强的 LLM、检索系统、工具 Agent，或任何 OpenAI-compatible endpoint。S2 在后台执行较慢推理，结果返回给 S1，但不阻塞实时交互。

## Interaction Layer、Thinking Layer、Thinker 和 Talker

论文层面的架构有两层：

- Interaction layer：实时 DuplexOmni 模型，持续接收流式音频/视频并持续产生语音/文本。
- Thinking layer：可插拔的后台推理或工具调用模块。

模型内部的 Qwen-Omni 组件是另一组概念：

- Thinker：内部多模态语言骨干，读取上下文并生成 Assistant 文本 token 和 hidden states。
- Talker：内部语音生成组件，把 Thinker 状态转换成流式 speech codec token 和 waveform。

因此，论文中的 “thinking layer” 不等于模型内部的 “Thinker” 模块。

## Inbound 和 Outbound

Inbound 表示用户主动发起的对话：用户先说，Agent 响应。

Outbound 表示 Agent 主动发起的对话：Agent 以任务型开场先说，用户随后响应。

保留两个 split 是因为实时 Agent 需要同时覆盖用户主动进入和 Agent 主动触达两类场景。

## Self-Audio

Self-audio 指中间 parquet 记录里保存的 Assistant 侧音频上下文或重建材料。它对对齐、调试或后续 codec 构造有用。

最终 E2E 训练前，`strip_self_audio_from_parquet.py` 会删除 self-audio 字段，避免模型把目标侧音频当成输入捷径。

## MTP

MTP 指 Talker 路径里的 multi-token prediction。在本仓库中，它是 Talker 侧预测 layer-0 codec token 之后残差 RVQ codebooks 的内部模块，不是外部 API 服务。
