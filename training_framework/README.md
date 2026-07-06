# Training Framework / 训练框架

## English

Purpose: package the Qwen3-Omni E2E training framework used by DuplexOmni and
document the public training launch convention.

Code role: this directory contains the ms-swift/Megatron-based trainer fork,
Qwen3-Omni E2E changes, recipe files, data schema readers, and the bundled
Megatron-LM core dependency.

Usage: build the final parquet with `data_pipeline`, download checkpoints from
the public model repositories when they are available, then train the Thinker
first and train the Talker from the saved Thinker checkpoint. Joint training is
not required for the standard training path.

## Directory Layout

```text
training_framework/
  qwen3_omni_training/
  Megatron-LM-core_v0.15.0/
```

Required environment variables:

```bash
cd /path/to/open_source
export SWIFT_SRC=$(pwd)/training_framework/qwen3_omni_training
export MEGATRON_LM_PATH=$(pwd)/training_framework/Megatron-LM-core_v0.15.0
export PYTHONPATH=$SWIFT_SRC:$PYTHONPATH
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MAX_PIXELS=176400
```

For SLURM, map scheduler variables to the same public variables:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
```

The large-scale example assumes 8 nodes x 8 GPUs. Smaller single-node smoke
runs are possible only after reducing tensor/expert/data parallel settings and
batch sizes to match local GPU memory.

## Thinker Training Template

```bash
megatron sft \
    --model /path/to/base-qwen3-omni-checkpoint \
    --model_type qwen3_omni \
    --load_safetensors true \
    --save_safetensors true \
    --dataset /path/to/training_v7_codec_noself \
    --tensor_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --sequence_parallel true \
    --micro_batch_size 2 \
    --global_batch_size 256 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 5e-7 \
    --max_epochs 1 \
    --save /path/to/duplexomni-thinker-output \
    --save_strategy steps \
    --save_interval 1000 \
    --max_length 32386 \
    --num_workers 32 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 32 \
    --model_author swift \
    --model_name duplexomni \
    --optimizer_cpu_offload false \
    --dataloader_pin_memory true \
    --dataloader_prefetch_factor 2 \
    --pipeline_model_parallel_size 1 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1
```

## Talker Training Template

```bash
megatron sft \
    --model /path/to/duplexomni-thinker-checkpoint \
    --model_type qwen3_omni_e2e \
    --load_safetensors true \
    --save_safetensors true \
    --dataset /path/to/training_v7_codec_noself \
    --tensor_model_parallel_size 2 \
    --expert_tensor_parallel_size 4 \
    --expert_model_parallel_size 8 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 256 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 5e-6 \
    --max_epochs 1 \
    --save /path/to/duplexomni-talker-output \
    --save_strategy steps \
    --save_interval 1000 \
    --max_length 21000 \
    --num_workers 32 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 32 \
    --model_author swift \
    --model_name duplexomni \
    --optimizer_cpu_offload false \
    --dataloader_pin_memory true \
    --dataloader_prefetch_factor 2 \
    --pipeline_model_parallel_size 1 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 4 \
    --megatron_extra_kwargs '{"qwen3_omni_e2e_thinker_loss_weight": 0.0, "qwen3_omni_e2e_ar_loss_weight": 1.0, "qwen3_omni_e2e_mlp_loss_weight": 1.0}' \
    --freeze_parameters language_model visual \
    --trainable_parameters talker
```

Expected training input is the final parquet produced by:

```text
data_pipeline/04_tts_parquet
  -> data_pipeline/05_mimi_codec
  -> data_pipeline/06_noise
```

## 中文

目的：打包 DuplexOmni 使用的 Qwen3-Omni E2E 训练框架，并说明公开训练启动约定。

代码作用：本目录包含基于 ms-swift/Megatron 的 trainer fork、Qwen3-Omni E2E 改动、训练 recipe、数据 schema reader，以及本仓库内的 Megatron-LM core 依赖。

使用方法：先用 `data_pipeline` 构建最终 parquet，在模型仓库可用后下载 checkpoint，然后先训练 Thinker，再从保存的 Thinker checkpoint 训练 Talker。标准训练流程不要求联合训练。

目录结构：

```text
training_framework/
  qwen3_omni_training/
  Megatron-LM-core_v0.15.0/
```

公开启动模板只依赖通用分布式变量：

```bash
cd /path/to/open_source
export SWIFT_SRC=$(pwd)/training_framework/qwen3_omni_training
export MEGATRON_LM_PATH=$(pwd)/training_framework/Megatron-LM-core_v0.15.0
export PYTHONPATH=$SWIFT_SRC:$PYTHONPATH
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MAX_PIXELS=176400
```

SLURM 环境可映射到同一组变量：

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
```

上方英文部分给出 Thinker 和 Talker 两段完整模板。大规模示例默认 8 节点 x 8 GPU。单机或小卡只适合 smoke，需要按本地显存缩小 tensor/expert/data parallel 和 batch size。

训练输入是数据管线产出的最终 parquet：

```text
data_pipeline/04_tts_parquet
  -> data_pipeline/05_mimi_codec
  -> data_pipeline/06_noise
```
