# 01 Content / 内容清洗

## English

Purpose: convert public dialogue or instruction datasets into DuplexOmni content
JSONL. The output is split into inbound and outbound style samples used by the
writer stage.

Code role: these scripts normalize public datasets into the content schema used
by `02_writer_director`, optionally rewriting text through an OpenAI-compatible
model.

Main files:

- `clean_all_datasets.py`: unified cleaner for multiple public text datasets
  under `DATASETS_ROOT`.
- `clean_ultrachat.py`: rewrite UltraChat-style JSONL into spoken dialogue.
- `clean_videochat.py`: convert video-chat style content into dialogue content.

Expected input: public dataset files downloaded separately, usually under
`$DATASETS_ROOT`.

Public Hugging Face sources used by `clean_all_datasets.py`:

| Key | Source |
| --- | --- |
| `wildchat` | <https://huggingface.co/datasets/allenai/WildChat-4.8M> |
| `mt_sft` | <https://huggingface.co/datasets/thomas-yanxin/MT-SFT-ShareGPT> |
| `belle` | <https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M> |
| `coig` | <https://huggingface.co/datasets/BAAI/COIG> |
| `coig_cqia` | <https://huggingface.co/datasets/m-a-p/COIG-CQIA> |
| `oasst2` | <https://huggingface.co/datasets/OpenAssistant/oasst2> |
| `no_robots` | <https://huggingface.co/datasets/HuggingFaceH4/no_robots> |

`clean_ultrachat.py` can also consume UltraChat-style input from
<https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k>. See
`../../EXTERNAL_ASSETS.md` for the expected local layout.

Expected output: content JSONL files such as:

```text
data/content/inbound.jsonl
data/content/outbound.jsonl
```

Usage: download public datasets separately, set `DATASETS_ROOT` or pass explicit
input/output paths, and keep generated content JSONL outside the source tree.

Example for UltraChat-style data:

```bash
python data_pipeline/01_content/clean_ultrachat.py \
  --input-jsonl /path/to/ultrachat/train_sft.jsonl \
  --output-jsonl data/content/train_sft.voiceagent.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --model gpt-4.1 \
  --concurrency 8 \
  --rpm-limit 0
```

Example for the unified public dataset cleaner:

```bash
DATASETS_ROOT=/path/to/public/datasets \
python data_pipeline/01_content/clean_all_datasets.py \
  --dataset all \
  --split both \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --model gpt-4.1
```

## 中文

目的：把公开对话或指令数据转换为 DuplexOmni content JSONL。输出会区分 inbound/outbound 风格，作为 writer 阶段输入。

代码作用：这些脚本把公开数据集规范化为 `02_writer_director` 使用的 content schema，并可通过 OpenAI-compatible 模型做文本改写。

使用方法：单独下载公开数据集，设置 `DATASETS_ROOT` 或显式传入输入输出路径；生成的 content JSONL 不进入源码仓库。

主要文件：

- `clean_all_datasets.py`：统一清洗 `$DATASETS_ROOT` 下的多个公开文本数据集。
- `clean_ultrachat.py`：把 UltraChat 风格 JSONL 改写成适合语音智能体的口语对话。
- `clean_videochat.py`：把 video-chat 风格内容转换成对话 content。

输入：单独下载的公开数据集，通常放在 `$DATASETS_ROOT`。

`clean_all_datasets.py` 使用的公开 Hugging Face 来源：

| Key | 来源 |
| --- | --- |
| `wildchat` | <https://huggingface.co/datasets/allenai/WildChat-4.8M> |
| `mt_sft` | <https://huggingface.co/datasets/thomas-yanxin/MT-SFT-ShareGPT> |
| `belle` | <https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M> |
| `coig` | <https://huggingface.co/datasets/BAAI/COIG> |
| `coig_cqia` | <https://huggingface.co/datasets/m-a-p/COIG-CQIA> |
| `oasst2` | <https://huggingface.co/datasets/OpenAssistant/oasst2> |
| `no_robots` | <https://huggingface.co/datasets/HuggingFaceH4/no_robots> |

`clean_ultrachat.py` 也可以处理 UltraChat 风格输入：
<https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k>。期望本地布局见
`../../EXTERNAL_ASSETS.md`。

输出：content JSONL，例如：

```text
data/content/inbound.jsonl
data/content/outbound.jsonl
```

使用方法见上方命令。运行时应显式传入 `--input-jsonl`、`--output-jsonl`、`--api-base`、`--api-key` 和 `--model`。
