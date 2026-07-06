# 00 Seed Configs / Seed 配置

## English

Purpose: define controllable scenario/style/interaction constraints for writer
generation. This stage produces cleaned seed guidance JSONL files used by
`02_writer_director`.

Code role: the JSON configs define slots and guidance; the Python scripts sample
slot combinations and clean them with an OpenAI-compatible model.

Included seed families:

- `seed_big`
- `seed_bc_big`
- `seed_xp_big`
- `seed_sparse`
- `seed_emotion`
- `seed_video`

Each non-video family contains:

- `config.json`: slot definitions and sampling space.
- `guidance.json`: natural-language guidance for the generated slots.
- `generate_slot_permutations.py`: generate seed combinations.
- `clean_samples_with_vllm.py`: clean/normalize generated samples through an
  OpenAI-compatible endpoint.

`seed_video` contains only `config.json` and `guidance.json` because the video
branch uses separate video content generation.

Generated `samples*.jsonl` files are not committed. Keep cleaned seed files as
external Hugging Face dataset assets or local data files.

Usage: run one seed family at a time, then collect the cleaned JSONL files into
the Hugging Face dataset asset directory or a local `data/seed/` directory.

Example:

```bash
cd data_pipeline/00_seed_configs/seed_big
python generate_slot_permutations.py \
  --sample-limit 10000 \
  --output-jsonl samples.jsonl

python clean_samples_with_vllm.py \
  --input-jsonl samples.jsonl \
  --output-jsonl samples.cleaned.jsonl \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --model gpt-4.1
```

## 中文

目的：定义 writer 生成时使用的可控场景、风格和交互约束。本阶段产出清洗后的 seed guidance JSONL，供 `02_writer_director` 使用。

代码作用：JSON 配置定义 slot 和 guidance；Python 脚本负责采样 slot 组合，并通过 OpenAI-compatible 模型清洗。

使用方法：逐个 seed family 运行生成与清洗，将 cleaned JSONL 放入 Hugging Face dataset 或本地 `data/seed/` 目录。

包含的 seed family：

- `seed_big`
- `seed_bc_big`
- `seed_xp_big`
- `seed_sparse`
- `seed_emotion`
- `seed_video`

非视频 seed family 通常包含：

- `config.json`：slot 定义与采样空间。
- `guidance.json`：slot 对应的自然语言约束。
- `generate_slot_permutations.py`：生成 seed 组合。
- `clean_samples_with_vllm.py`：通过 OpenAI-compatible endpoint 清洗/规范化生成样本。

`seed_video` 只包含 `config.json` 和 `guidance.json`，因为视频分支有单独的视频内容生成步骤。

生成后的 `samples*.jsonl` 不进入代码仓库。清洗后的 seed 文件应作为外部 Hugging Face dataset 资产或本地数据文件保存。

最小运行示例见上方英文命令；在实际运行中将 endpoint、model 和输出路径替换为你的本地配置。
