# External Assets / 外部资产

## English

Purpose: list the datasets, model checkpoints, noise corpora, and reference
assets expected by the DuplexOmni repository.

Code role: this document describes external resources only. These assets are
not committed to the source tree, except for the small reference voice sample
`assets/reference_audio/google_Leda.wav`.

Usage: download assets into local directories, then pass those paths through
CLI arguments or environment variables. Do not commit downloaded datasets,
checkpoints, generated parquet, logs, or media into this repository.

## DuplexOmni Dataset And Checkpoint Assets

The generated DuplexOmni dataset and trained checkpoints are external assets.
They are not stored in this source repository. When public Hugging Face
repositories are available, place them under the local layout below or pass
explicit paths to the relevant scripts.

```text
DuplexOmni dataset repo: <duplexomni-dataset-repo>
DuplexOmni thinker checkpoint: <duplexomni-thinker-model-repo>
DuplexOmni talker checkpoint: <duplexomni-talker-model-repo>
```

Expected local layout:

```text
data/seed/*.cleaned.jsonl
data/content/inbound.jsonl
data/content/outbound.jsonl
outputs/pipeline_text/*.jsonl
outputs/parquet/final_training_parquet/
models/base-qwen3-omni/
models/duplexomni-thinker/
models/duplexomni-talker/
models/qwen3-tts/
models/mimi/
```

## Upstream Model Assets

These are public upstream Hugging Face assets used by the pipeline. They are not
generated DuplexOmni assets and are not copied into this repository.

| Role | Hugging Face source | Expected local path |
| --- | --- | --- |
| Base omni model for training | <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct> | `models/base-qwen3-omni/` |
| Optional thinking model reference | <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking> | user supplied |
| Optional audio captioner reference | <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner> | user supplied |
| Qwen3-TTS custom voice | <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice> | `models/qwen3-tts-custom/` |
| Qwen3-TTS base | <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base> | `models/qwen3-tts-base/` |
| Qwen3 forced aligner | <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B> | `models/qwen3-forced-aligner/` |
| Mimi audio codec | <https://huggingface.co/kyutai/mimi> | `models/mimi/` |

When DuplexOmni model repositories are available, place the downloaded
thinker/talker checkpoints under
`models/duplexomni-thinker/` and `models/duplexomni-talker/`.

## Public Content Sources

`data_pipeline/01_content/clean_all_datasets.py` expects these public Hugging
Face datasets under `$DATASETS_ROOT`:

| Key | Hugging Face source | Expected local subdirectory |
| --- | --- | --- |
| `wildchat` | <https://huggingface.co/datasets/allenai/WildChat-4.8M> | `allenai/WildChat-4.8M` |
| `mt_sft` | <https://huggingface.co/datasets/thomas-yanxin/MT-SFT-ShareGPT> | `thomas-yanxin/MT-SFT-ShareGPT` |
| `belle` | <https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M> | `BelleGroup/multiturn_chat_0.8M` |
| `coig` | <https://huggingface.co/datasets/BAAI/COIG> | `BAAI/COIG` |
| `coig_cqia` | <https://huggingface.co/datasets/m-a-p/COIG-CQIA> | `m-a-p/COIG-CQIA` |
| `oasst2` | <https://huggingface.co/datasets/OpenAssistant/oasst2> | `OpenAssistant/oasst2` |
| `no_robots` | <https://huggingface.co/datasets/HuggingFaceH4/no_robots> | `HuggingFaceH4/no_robots` |

`clean_ultrachat.py` is a separate helper for UltraChat-style input:

| Helper | Hugging Face source | Typical file |
| --- | --- | --- |
| UltraChat | <https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k> | `data/train_sft.jsonl` |

## Noise Sources

The noise stage can use MUSAN, FSD50K, or both.

- MUSAN official source: <https://www.openslr.org/17/>
- FSD50K official release: <https://zenodo.org/records/4060432>
- FSD50K companion site: <https://fsannotator.upf.edu/fsd/release/FSD50K/>

Expected local layout:

```text
$NOISE_ROOT/
  MUSAN/raw/musan/noise/free-sound/*.wav
  MUSAN/raw/musan/noise/sound-bible/*.wav
  FSD50K/raw/FSD50K/FSD50K.dev_audio/*.wav
```

`MUSAN` and `FSD50K` are not copied into the repository. They are external
datasets and must be downloaded by the user according to their licenses.

## Reference Voice

The bundled reference voice sample is:

```text
assets/reference_audio/google_Leda.wav
```

The reference text is:

```text
The weather is nice, and I speak calmly and clearly. 今天天气很好，我平静清楚地说话。
```

## 中文

目的：列出 DuplexOmni 仓库所需的数据集、模型 checkpoint、噪声语料和参考资产。

代码作用：本文只描述外部资源。除小型参考音色
`assets/reference_audio/google_Leda.wav` 外，这些资产不进入源码树。

使用方法：把资产下载到本地目录，再通过命令行参数或环境变量传入。不要把下载的数据集、checkpoint、生成 parquet、日志或媒体提交到仓库。

## DuplexOmni 数据集和 checkpoint 资产

生成后的 DuplexOmni 数据集和训练后的 checkpoint 是外部资产，不存放在源码仓库中。公开 Hugging Face 仓库可用后，可按下面的本地布局放置，或通过脚本参数显式传入路径。

```text
DuplexOmni dataset repo: <duplexomni-dataset-repo>
DuplexOmni thinker checkpoint: <duplexomni-thinker-model-repo>
DuplexOmni talker checkpoint: <duplexomni-talker-model-repo>
```

期望本地布局：

```text
data/seed/*.cleaned.jsonl
data/content/inbound.jsonl
data/content/outbound.jsonl
outputs/pipeline_text/*.jsonl
outputs/parquet/final_training_parquet/
models/base-qwen3-omni/
models/duplexomni-thinker/
models/duplexomni-talker/
models/qwen3-tts/
models/mimi/
```

## 上游模型资产

这些是 pipeline 使用的公开上游 Hugging Face 资产。它们不是生成后的 DuplexOmni 资产，也不复制进本仓库。

| 作用 | Hugging Face 来源 | 期望本地路径 |
| --- | --- | --- |
| 训练基础 omni 模型 | <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct> | `models/base-qwen3-omni/` |
| 可选 thinking 模型参考 | <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking> | 用户自行指定 |
| 可选音频 captioner 参考 | <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner> | 用户自行指定 |
| Qwen3-TTS custom voice | <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice> | `models/qwen3-tts-custom/` |
| Qwen3-TTS base | <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base> | `models/qwen3-tts-base/` |
| Qwen3 forced aligner | <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B> | `models/qwen3-forced-aligner/` |
| Mimi audio codec | <https://huggingface.co/kyutai/mimi> | `models/mimi/` |

DuplexOmni 模型仓库可用后，将下载的 thinker/talker checkpoint 分别放到
`models/duplexomni-thinker/` 和 `models/duplexomni-talker/`。

## 公开 content 数据源

`data_pipeline/01_content/clean_all_datasets.py` 期望 `$DATASETS_ROOT` 下有这些公开 Hugging Face 数据集：

| Key | Hugging Face 来源 | 期望本地子目录 |
| --- | --- | --- |
| `wildchat` | <https://huggingface.co/datasets/allenai/WildChat-4.8M> | `allenai/WildChat-4.8M` |
| `mt_sft` | <https://huggingface.co/datasets/thomas-yanxin/MT-SFT-ShareGPT> | `thomas-yanxin/MT-SFT-ShareGPT` |
| `belle` | <https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M> | `BelleGroup/multiturn_chat_0.8M` |
| `coig` | <https://huggingface.co/datasets/BAAI/COIG> | `BAAI/COIG` |
| `coig_cqia` | <https://huggingface.co/datasets/m-a-p/COIG-CQIA> | `m-a-p/COIG-CQIA` |
| `oasst2` | <https://huggingface.co/datasets/OpenAssistant/oasst2> | `OpenAssistant/oasst2` |
| `no_robots` | <https://huggingface.co/datasets/HuggingFaceH4/no_robots> | `HuggingFaceH4/no_robots` |

`clean_ultrachat.py` 是 UltraChat 风格输入的单独 helper：

| Helper | Hugging Face 来源 | 常用文件 |
| --- | --- | --- |
| UltraChat | <https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k> | `data/train_sft.jsonl` |

## 噪声数据源

噪声阶段可以使用 MUSAN、FSD50K，或二者同时使用。

- MUSAN 官方来源：<https://www.openslr.org/17/>
- FSD50K 官方 release：<https://zenodo.org/records/4060432>
- FSD50K companion site：<https://fsannotator.upf.edu/fsd/release/FSD50K/>

期望本地布局：

```text
$NOISE_ROOT/
  MUSAN/raw/musan/noise/free-sound/*.wav
  MUSAN/raw/musan/noise/sound-bible/*.wav
  FSD50K/raw/FSD50K/FSD50K.dev_audio/*.wav
```

`MUSAN` 和 `FSD50K` 不复制进仓库。它们是外部数据集，用户需要按各自许可证自行下载。

## 参考音色

随仓库提供的小型参考音色为：

```text
assets/reference_audio/google_Leda.wav
```

参考文本为：

```text
The weather is nice, and I speak calmly and clearly. 今天天气很好，我平静清楚地说话。
```
