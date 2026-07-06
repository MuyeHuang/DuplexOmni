# Third-Party Notices

This repository combines DuplexOmni source code with several upstream open-source
frameworks needed for training and inference. The top-level `LICENSE` applies to
DuplexOmni-owned source files. Upstream directories keep their original license
files and copyright notices.

## Bundled Upstream Source

| Path | Upstream role | License location |
| --- | --- | --- |
| `inference_framework/vllm_qwen3_omni/` | Modified vLLM runtime fork used by realtime serving. | `inference_framework/vllm_qwen3_omni/LICENSE` |
| `training_framework/qwen3_omni_training/` | Modified ms-swift/Qwen3-Omni training framework. | `training_framework/qwen3_omni_training/LICENSE` |
| `training_framework/Megatron-LM-core_v0.15.0/` | Bundled Megatron-LM core dependency used by the training framework. | `training_framework/Megatron-LM-core_v0.15.0/LICENSE` |

When redistributing this repository, retain the license files and attribution
notices inside those upstream directories. Modified upstream files should retain
their original headers where present and should be treated as derivative works of
their respective upstream projects.

## External Assets Not Included

Generated DuplexOmni datasets, model checkpoints, training parquet files,
downloaded public datasets, generated audio/video, logs, and internal service
endpoints are intentionally not included in this source repository.

Until public artifact repositories are ready, documentation uses placeholders
such as:

```text
<duplexomni-dataset-repo>
<duplexomni-thinker-model-repo>
<duplexomni-talker-model-repo>
<tts-model-repo-if-needed>
```

Users must download external assets separately and comply with the license terms
of each dataset, checkpoint, and model provider.

## Bundled Reference Audio

The only bundled audio asset is:

```text
assets/reference_audio/google_Leda.wav
```

It is included as a small reference voice prompt for public examples. Do not add
generated media or training data to the source repository.
