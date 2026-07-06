# -*- coding: utf-8 -*-
"""项目路径与常量配置。"""
import os
from pathlib import Path

# 项目根目录（所有输出保存在此）
PROJECT_ROOT = Path(__file__).resolve().parent

# Mimi 编码器（本地目录）
MIMI_PATH = Path(os.environ.get("MIMI_PATH", "models/mimi"))

# 原始训练数据（约 20GB+，不整文件读入）
TRAIN_JSONL = Path(os.environ.get("TRAIN_JSONL", "data/train.jsonl"))

# 微调后的 Thinker+Talker  checkpoint（用于提取 Thinker 特征与 Talker 权重）
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "models/checkpoint"))

# 采样 500 行保存路径
SAMPLED_500_JSONL = PROJECT_ROOT / "sampled_500.jsonl"

# Thinker 特征缓存目录（Layer0 / accept_layer 特征、mask 等）
THINKER_FEATURES_DIR = PROJECT_ROOT / "thinker_features"

# Mimi codes 缓存目录（与 thinker 样本一一对应）
MIMI_CODES_DIR = PROJECT_ROOT / "mimi_codes"

# 训练曲线与日志
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_OUT_DIR = PROJECT_ROOT / "checkpoints"

# Mimi 帧率 12.5 Hz => 80ms 一帧；目标时长取 480ms（6 帧）
FRAME_MS = 80
TARGET_AUDIO_MS = 480
NUM_MIMI_LAYERS_USE = 16  # 只使用前 16 层，与 Qwen3-Omni 一致
