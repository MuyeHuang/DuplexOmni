"""
Talker HTTP service: tensor-in (thinker top/bottom) -> codec + MTP + Code2Wav -> WAV.

- **Talker MoE（主解码）仅 vLLM**：`vllm_qwen3_omni` fork 内 `Qwen3OmniMoeTalkerForConditionalGeneration`，
  合并 `prompt_token_ids` + `prompt_embeds`（对齐 fast_agent talker 的 token/嵌入并行输入），KV / prefix cache / cudagraph。
- **MTP**：**默认** ``TALKER_MTP_SPLIT_ENGINE=1``：在 **子进程** 再启一套 vLLM（``mtp_engine=subproc_vllm``），终端会出现 **第二套** ``Capturing CUDA graphs``（MTP 引擎自己的 warmup）。**默认** ``TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES=0``：仅物理 **GPU 0**（可与 MoE rank0 共卡），``TALKER_GPU_MEM_UTIL`` / ``TALKER_MTP_GPU_MEM_UTIL`` 分账。若 ``TALKER_MTP_SPLIT_ENGINE=0`` 则 **shared_moetalker**（与 MoE 共 ``LLM``），启动 tqdm **只有** MoE 那条，**没有**单独 MTP 图捕获。排障日志默认开启：``TALKER_MTP_VERBOSE_LOG=1``、``TALKER_MTP_WORKER_VERBOSE=1``、``TALKER_MTP_PROFILE=1``（设为 ``0`` 关闭）。
- **禁止** `import vllm_omni`。需要 **CUDA**；无 GPU 则进程拒绝启动（不再提供 HF MoE 兜底）。

MTP、Code2Wav、FastAPI 在本模块；`LLM.apply_model` 的跨进程回调在 **`server_talker_rpc`**（避免 `__main__` pickle 问题）。
"""

from __future__ import annotations

import os
import sys

# vLLM V1 MultiprocExecutor 在 Linux 上默认用 fork 起 WorkerProc。父进程若在 fork 前已初始化
# CUDA，子进程再执行 torch.cuda.set_device 会触发 PyTorch 的
#「Cannot re-initialize CUDA in forked subprocess」。EngineCore 在拉起 worker 前的导入/
# 插件/探测路径可能已触碰 CUDA；主进程里 FastAPI lifespan 也会在 LLM() 前调用
# torch.cuda.is_available()。官方机制见 vllm/utils/system_utils.py:_maybe_force_spawn
# 与 vLLM 文档 multiprocessing 排障：在未显式设置时使用 spawn，避免 fork 继承 CUDA 上下文。
if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") is None:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# `LLM.apply_model` → `collective_rpc("apply_model", args=(func,))` 需把 **callable** 发到 EngineCore
# worker；v1 `serial_utils` 默认 msgpack 不能编码函数，见 vllm/v1/serial_utils.py:enc_hook。
# 官方开关：允许 pickle/cloudpickle（错误信息同上）。Talker 的 `apply_model` 回调在
# `server_talker_rpc` 模块中定义（避免挂在 `__main__` 上无法被 worker unpickle）。用户可显式设为 0。
if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") is None:
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

# 本服务当前阶段以定位耗时为优先：默认开启 host/worker 逐次 MTP 日志和内部 CUDA Event profile。
os.environ.setdefault("TALKER_MTP_VERBOSE_LOG", "1")
os.environ.setdefault("TALKER_MTP_WORKER_VERBOSE", "1")
os.environ.setdefault("TALKER_MTP_PROFILE", "1")

import asyncio
import copy
import io
import json
import logging
import multiprocessing
import time
import wave
from collections.abc import Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTalkerConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeCode2Wav

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VLLM_ROOT = os.environ.get(
    "VLLM_ROOT",
    os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "vllm_qwen3_omni")),
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if VLLM_ROOT not in sys.path:
    sys.path.insert(0, VLLM_ROOT)

# `VLLM_WORKER_MULTIPROC_METHOD=spawn` 子进程需能 `import server_talker_rpc`（apply_model 的 pickle 全局名）。
_pp = os.environ.get("PYTHONPATH", "")
_pp_parts = [p for p in _pp.split(os.pathsep) if p]
if ROOT_DIR not in _pp_parts:
    os.environ["PYTHONPATH"] = ROOT_DIR if not _pp else ROOT_DIR + os.pathsep + _pp

from server_talker_mtp_worker import mtp_worker_main  # noqa: E402
from server_talker_rpc import (  # noqa: E402
    vllm_apply_projection_rpc,
    vllm_mtp_run_rpc,
    vllm_read_last_sample_hidden_cpu,
)

from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator  # noqa: E402
from vllm.entrypoints.llm import LLM  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("server_talker")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 20000
    model: str = "models/qwen3-omni-talker"
    sample_rate: int = 24000
    # 至少 real_codec_steps_per_turn 次 layer0 + MTP，再留 1+ 步给 codec EOS；略放大避免偶发起不来 EOS 时直接断在 7 次上限。
    talker_decode_steps: int = 8
    real_codec_steps_per_turn: int = 6
    max_model_len: int = 32386
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 20
    chunk_size_code2wav: int = 300
    left_context_code2wav: int = 25
    device: str = "cuda:0"
    tensor_parallel_size: int = 1


def _server_config_from_env() -> ServerConfig:
    b = ServerConfig()
    return ServerConfig(
        host=os.environ.get("TALKER_HOST", b.host),
        port=int(os.environ.get("TALKER_PORT", str(b.port))),
        model=os.environ.get("TALKER_MODEL", b.model),
        sample_rate=b.sample_rate,
        talker_decode_steps=int(os.environ.get("TALKER_DECODE_STEPS", str(b.talker_decode_steps))),
        real_codec_steps_per_turn=int(os.environ.get("TALKER_REAL_CODEC_STEPS", str(b.real_codec_steps_per_turn))),
        max_model_len=b.max_model_len,
        temperature=float(os.environ.get("TALKER_TEMPERATURE", str(b.temperature))),
        top_p=float(os.environ.get("TALKER_TOP_P", str(b.top_p))),
        top_k=int(os.environ.get("TALKER_TOP_K", str(b.top_k))),
        chunk_size_code2wav=b.chunk_size_code2wav,
        left_context_code2wav=b.left_context_code2wav,
        device=os.environ.get("TALKER_DEVICE", b.device),
        tensor_parallel_size=int(os.environ.get("TALKER_TP", str(b.tensor_parallel_size))),
    )


def _torch_dtype_from_config(dtype_str: str | None) -> torch.dtype:
    if dtype_str is None:
        return torch.bfloat16
    d = str(dtype_str).lower()
    if d in ("bfloat16", "bf16"):
        return torch.bfloat16
    if d in ("float16", "fp16"):
        return torch.float16
    return torch.float32


def _mtp_inference_dtype(device: torch.device) -> torch.dtype:
    """Host 侧 codec 嵌入（``TalkerMtpEmbeddings``）推理 dtype。

    GPU 上固定 **bfloat16**，与 checkpoint **根** ``config.dtype``、vLLM ``dtype=auto`` 的 talker MoE 无关。
    CPU 上为 float32。
    """
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


_INT_TIMING_KEYS = frozenset(
    {
        "vllm_step_count",
        "mtp_call_count",
        "mtp_substep_count",
        "mtp_prefill_substep_count",
        "mtp_decode_substep_count",
        "mtp_cuda_sync",
        "mtp_profile_enabled",
    }
)


def _serialize_timings(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten for JSON header / logs: seconds rounded, integer counters kept int."""
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k in _INT_TIMING_KEYS:
            out[k] = int(v)
        elif isinstance(v, (int, float)):
            out[k] = round(float(v), 6)
        else:
            out[k] = v
    return out


def _mtp_cuda_sync_enabled() -> bool:
    return os.environ.get("TALKER_MTP_CUDA_SYNC", "").strip().lower() in ("1", "true", "yes")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _mtp_verbose_host() -> bool:
    """Host：``TALKER_MTP_VERBOSE_LOG=1`` 时每个 chunk 内逐次 MTP 打 INFO（默认开启）。"""
    return _env_truthy("TALKER_MTP_VERBOSE_LOG")


def _mtp_profile_enabled() -> bool:
    """Worker 内部 CUDA Event profile，默认开启；``TALKER_MTP_PROFILE=0`` 关闭。"""
    raw = os.environ.get("TALKER_MTP_PROFILE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _merge_numeric(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, (int, float)):
            dst[k] = float(dst.get(k, 0.0)) + float(v)


def _merge_count_map(dst: dict[str, Any], name: str, src: dict[str, Any]) -> None:
    values = src.get(name)
    if not isinstance(values, dict):
        return
    bucket = dst.setdefault(name, {})
    if not isinstance(bucket, dict):
        bucket = {}
        dst[name] = bucket
    for k, v in values.items():
        if isinstance(v, (int, float)):
            sk = str(k)
            bucket[sk] = int(bucket.get(sk, 0)) + int(v)


def _merge_mtp_profile(timings: dict[str, Any], profile: dict[str, Any]) -> None:
    if not profile:
        return
    acc = timings.setdefault(
        "mtp_profile",
        {
            "calls": 0,
            "num_events": 0,
            "ms": {},
            "counts": {},
        },
    )
    if not isinstance(acc, dict):
        return
    acc["calls"] = int(acc.get("calls", 0)) + int(profile.get("calls", 0) or 0)
    acc["num_events"] = int(acc.get("num_events", 0)) + int(profile.get("num_events", 0) or 0)
    ms = acc.setdefault("ms", {})
    counts = acc.setdefault("counts", {})
    if isinstance(ms, dict) and isinstance(profile.get("ms"), dict):
        _merge_numeric(ms, profile["ms"])
    if isinstance(counts, dict) and isinstance(profile.get("counts"), dict):
        for k, v in profile["counts"].items():
            if isinstance(v, (int, float)):
                counts[str(k)] = int(counts.get(str(k), 0)) + int(v)
    for name in (
        "segment_seq_len_counts",
        "attention_seq_len_counts",
        "num_code_groups_counts",
        "backend_missing_counts",
        "backend_attempt_counts",
        "backend_success_counts",
        "backend_error_counts",
    ):
        _merge_count_map(acc, name, profile)
    if "backend_order" not in acc and isinstance(profile.get("backend_order"), list):
        acc["backend_order"] = list(profile["backend_order"])
    if isinstance(profile.get("removed_attention_backends"), list):
        removed = acc.setdefault("removed_attention_backends", [])
        if isinstance(removed, list):
            for item in profile["removed_attention_backends"]:
                if item not in removed:
                    removed.append(item)


def _finalize_mtp_profile(timings: dict[str, Any]) -> None:
    profile = timings.get("mtp_profile")
    if not isinstance(profile, dict):
        return
    ms = profile.get("ms")
    counts = profile.get("counts")
    if not isinstance(ms, dict) or not isinstance(counts, dict):
        return
    calls = max(1, int(profile.get("calls", 0) or 0))
    residual_groups = max(1, int(counts.get("residual_groups", 0) or 0))
    mtp_blocks = max(1, int(counts.get("mtp_block_calls", 0) or 0))
    rounded_ms = {k: round(float(v), 4) for k, v in sorted(ms.items()) if isinstance(v, (int, float))}
    profile["ms"] = rounded_ms
    profile["avg_ms_per_call"] = {k: round(v / calls, 4) for k, v in rounded_ms.items()}
    profile["avg_ms_per_group"] = {k: round(v / residual_groups, 4) for k, v in rounded_ms.items()}
    profile["avg_ms_per_block"] = {k: round(v / mtp_blocks, 4) for k, v in rounded_ms.items()}
    profile["top_ms"] = [
        {"name": k, "ms": round(v, 4)}
        for k, v in sorted(rounded_ms.items(), key=lambda item: item[1], reverse=True)[:12]
    ]


def _mtp_split_engine_enabled() -> bool:
    """MTP 独立子进程 vLLM。**默认开启**（第二套 ``Capturing CUDA graphs``）；``TALKER_MTP_SPLIT_ENGINE=0`` 关闭。"""
    raw = os.environ.get("TALKER_MTP_SPLIT_ENGINE", "1").strip().lower()
    if raw in ("0", "false", "no", "off", "shared"):
        return False
    return True


def _cuda_visible_device_count(visible: str) -> int:
    return len([p for p in visible.split(",") if p.strip()])


def _spawn_mtp_vllm_subprocess(cfg: ServerConfig) -> tuple[multiprocessing.Process, Connection]:
    # 默认仅物理 GPU 0：4 卡 TP=4 时与 MoE rank0 同卡，显存由 TALKER_GPU_MEM_UTIL / TALKER_MTP_GPU_MEM_UTIL 分账
    vis = os.environ.get("TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES", "").strip() or "0"
    os.environ["TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES"] = vis
    tp = int(os.environ.get("TALKER_MTP_SUBPROC_TP", os.environ.get("TALKER_MTP_TP", "1")))
    nvis = _cuda_visible_device_count(vis)
    if tp != nvis:
        raise ValueError(
            f"TALKER_MTP_SUBPROC_TP（或 TALKER_MTP_TP）={tp} 必须与 "
            f"TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES={vis!r} 中 GPU 个数一致（当前 {nvis}）。"
        )
    os.environ.setdefault("TALKER_MODEL", cfg.model)
    os.environ.setdefault("TALKER_MAX_MODEL_LEN", str(cfg.max_model_len))

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    # Process() 无 env=；由 mtp_worker_main(conn, vis) 首行设置 CUDA_VISIBLE_DEVICES
    proc = ctx.Process(
        target=mtp_worker_main,
        args=(child_conn, vis),
        name="TalkerMtpVllm",
    )
    proc.start()
    child_conn.close()
    m1 = parent_conn.recv()
    if m1.get("stage") == "error":
        proc.join(timeout=10.0)
        raise RuntimeError(f"MTP 子进程启动失败: {m1.get('err')}\n{m1.get('tb', '')}")
    m2 = parent_conn.recv()
    if m2.get("stage") == "error":
        proc.join(timeout=10.0)
        raise RuntimeError(f"MTP 子进程加载模型失败: {m2.get('err')}\n{m2.get('tb', '')}")
    if m2.get("stage") != "ready":
        proc.join(timeout=10.0)
        raise RuntimeError(f"MTP 子进程握手异常: {m2!r}")
    LOGGER.info(
        "MTP 独立 vLLM 子进程已就绪 pid=%s tp=%s 子进程 CUDA_VISIBLE_DEVICES=%r（可与 MoE 共物理 GPU 0："
        "MoE TALKER_GPU_MEM_UTIL 默认 0.7，MTP 子进程 TALKER_MTP_GPU_MEM_UTIL 默认 0.1；若 OOM 再调低）。",
        m2.get("pid"),
        tp,
        vis,
    )
    return proc, parent_conn


# -----------------------------------------------------------------------------
# Safetensors / checkpoint helpers (no vllm-omni)
# -----------------------------------------------------------------------------


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _resolve_weight_files(model_path: str | Path) -> list[str]:
    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index_data = _load_json(index_path)
        shard_names = sorted(set(index_data["weight_map"].values()))
        return [str(root / name) for name in shard_names]
    single = root / "model.safetensors"
    if single.exists():
        return [str(single)]
    shards = sorted(root.rglob("*.safetensors"))
    if shards:
        return [str(p) for p in shards]
    raise FileNotFoundError(f"No safetensors under `{root}`.")


def iter_weight_tensors(
    model_path: str | Path,
    *,
    prefixes: Sequence[str] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    pt = tuple(prefixes or ())
    for wf in _resolve_weight_files(model_path):
        for name, tensor in safetensors_weights_iterator([wf], use_tqdm_on_load=False):
            if pt and not name.startswith(pt):
                continue
            yield name, tensor


def collect_prefixed_weights(
    model_path: str | Path,
    *,
    prefixes: Sequence[str],
    strip_prefix: str | None = None,
) -> list[tuple[str, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor]] = []
    for name, tensor in iter_weight_tensors(model_path, prefixes=prefixes):
        if strip_prefix and name.startswith(strip_prefix):
            name = name[len(strip_prefix) :]
        out.append((name, tensor))
    return out


# -----------------------------------------------------------------------------
# Host codec embeddings only (prefill / history). MTP transformer: vLLM worker.
# -----------------------------------------------------------------------------


@dataclass
class MtpOutput:
    full_codes: torch.Tensor
    summed_embedding: torch.Tensor


class TalkerMtpEmbeddings(nn.Module):
    """与旧 ``InlineMtpRunner`` 相同协议：layer0 + RVQ residual **嵌入** 用于 ``build_prefill`` / 历史拼接。

    **MTP transformer**（``Qwen3OmniMoeTalkerCodePredictor``）仅在 **vLLM worker** 内执行，见 ``vllm_mtp_run_rpc``。
    """

    def __init__(
        self,
        model_path: str,
        talker_config: Qwen3OmniMoeTalkerConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.talker_config = talker_config
        self.device = device
        self.dtype = dtype
        self.codec_bos_id = int(talker_config.codec_bos_id)
        self.codec_eos_id = int(talker_config.codec_eos_token_id)
        self.codec_pad_id = int(talker_config.codec_pad_id)
        self.hidden_size = int(talker_config.text_config.hidden_size)
        cp_cfg = talker_config.code_predictor_config
        self.layer0_embedding = nn.Embedding(
            int(talker_config.text_config.vocab_size),
            int(talker_config.text_config.hidden_size),
        )
        n_res = int(cp_cfg.num_code_groups) - 1
        self.residual_codec_embeddings = nn.ModuleList(
            [nn.Embedding(cp_cfg.vocab_size, cp_cfg.hidden_size) for _ in range(n_res)]
        )
        self._load_embedding_weights()
        self.to(device=device, dtype=dtype).eval()

    def _load_embedding_weights(self) -> None:
        for name, tensor in iter_weight_tensors(
            self.model_path,
            prefixes=("talker.model.codec_embedding.weight",),
        ):
            if name == "talker.model.codec_embedding.weight":
                self.layer0_embedding.weight.data.copy_(tensor)
                break
        else:
            raise RuntimeError("Missing talker.model.codec_embedding.weight")
        w = collect_prefixed_weights(
            self.model_path,
            prefixes=("talker.code_predictor.model.codec_embedding.",),
            strip_prefix="talker.code_predictor.",
        )
        by_key = {k: t for k, t in w}
        for i in range(len(self.residual_codec_embeddings)):
            key = f"model.codec_embedding.{i}.weight"
            found = by_key.get(key)
            if found is None:
                raise RuntimeError(f"Missing code predictor embedding weight `{key}` in checkpoint")
            self.residual_codec_embeddings[i].weight.data.copy_(found)

    @torch.inference_mode()
    def codec_bos(self) -> tuple[torch.Tensor, torch.Tensor]:
        tok = torch.tensor([self.codec_bos_id], dtype=torch.long, device=self.device)
        emb = self.layer0_embedding(tok)
        return tok.cpu(), emb.cpu()

    @torch.inference_mode()
    def codec_eos(self) -> tuple[torch.Tensor, torch.Tensor]:
        tok = torch.tensor([self.codec_eos_id], dtype=torch.long, device=self.device)
        emb = self.layer0_embedding(tok)
        return tok.cpu(), emb.cpu()

    @torch.inference_mode()
    def build_codec_history_inputs(self, full_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if full_codes.ndim == 2:
            full_codes = full_codes.unsqueeze(0)
        if full_codes.ndim != 3:
            raise RuntimeError(f"full_codes rank-3 expected, got {tuple(full_codes.shape)}")
        if full_codes.shape[0] != 1:
            raise RuntimeError("batch_size=1 only")
        layer0_ids = full_codes[0, 0, :].to(dtype=torch.long)
        codec_embeds = self._build_codec_sum_embeds_from_codes(full_codes).squeeze(0)
        bos_id, bos_embed = self.codec_bos()
        eos_id, eos_embed = self.codec_eos()
        token_ids = torch.cat((bos_id, layer0_ids.cpu().contiguous(), eos_id), dim=0)
        embeds = torch.cat(
            (
                bos_embed.to(codec_embeds.dtype),
                codec_embeds.to("cpu").contiguous(),
                eos_embed.to(codec_embeds.dtype),
            ),
            dim=0,
        )
        return token_ids.contiguous(), embeds.contiguous()

    @torch.inference_mode()
    def _build_codec_sum_embeds_from_codes(self, full_codes: torch.Tensor) -> torch.Tensor:
        expected_num_groups = len(self.residual_codec_embeddings) + 1
        if full_codes.shape[1] != expected_num_groups:
            raise RuntimeError(f"RVQ groups mismatch: expected {expected_num_groups} got {full_codes.shape[1]}")
        full_codes = full_codes.to(device=self.device, dtype=torch.long)
        summed = self.layer0_embedding(full_codes[:, 0, :])
        for group_idx, emb_layer in enumerate(self.residual_codec_embeddings, start=1):
            summed = summed + emb_layer(full_codes[:, group_idx, :])
        return summed.to(dtype=self.dtype)


# -----------------------------------------------------------------------------
# History + prefill (chunk-at-a-time; aligned with bridge.build_prefill)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# History + prefill (chunk-at-a-time; aligned with bridge.build_prefill)
# -----------------------------------------------------------------------------


@dataclass
class HistoryBlock:
    """Matches `fast_agent` `HistoryBlock` / `bridge.build_prefill` expectations."""

    codes: torch.Tensor
    conditioning_embeds: torch.Tensor
    conditioning_token_ids: torch.Tensor
    real_codec_steps: int


@dataclass
class SessionState:
    history: list[HistoryBlock] = field(default_factory=list)


def build_prefill_embeds(
    project_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mtp: TalkerMtpEmbeddings,
    session: SessionState,
    current_emb: torch.Tensor,
    current_top: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Token/embed scaffold aligned with `fast_agent/bridge.py` + `mtp_runner.py`.

    Conditioning rows use **codec_pad_id** on the token side-channel (not 0).
    History codec segments use **BOS + layer0 + EOS** from `build_codec_history_inputs`.
    """
    pad = int(mtp.codec_pad_id)
    bos_id, bos_embed = mtp.codec_bos()
    embed_parts: list[torch.Tensor] = []
    id_parts: list[torch.Tensor] = []
    for hb in session.history:
        cond_emb = hb.conditioning_embeds
        cond_ids = hb.conditioning_token_ids
        if cond_ids.shape[0] != cond_emb.shape[0]:
            raise RuntimeError(
                f"history conditioning ids/embeds length mismatch: ids={cond_ids.shape[0]} "
                f"emb={cond_emb.shape[0]}"
            )
        hist_ids, hist_emb = mtp.build_codec_history_inputs(hb.codes)
        embed_parts.append(cond_emb)
        id_parts.append(cond_ids)
        embed_parts.append(hist_emb.to(cond_emb.dtype))
        id_parts.append(hist_ids)
    cur_cond = project_fn(current_emb, current_top).detach().cpu()
    cur_ids = torch.full((cur_cond.shape[0],), pad, dtype=torch.long)
    embed_parts.append(cur_cond)
    id_parts.append(cur_ids)
    embed_parts.append(bos_embed.to(cur_cond.dtype))
    id_parts.append(bos_id)
    prompt_embeds = torch.cat(embed_parts, dim=0).contiguous()
    prompt_ids = torch.cat(id_parts, dim=0).contiguous()
    return prompt_ids, prompt_embeds, cur_cond, cur_ids


def append_codec_step(
    prompt_token_ids: torch.Tensor,
    prompt_embeds: torch.Tensor,
    layer0_token_id: int,
    summed_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_id = torch.tensor([layer0_token_id], dtype=torch.long)
    next_embed = summed_embedding.reshape(1, -1).to(prompt_embeds.dtype)
    return (
        torch.cat((prompt_token_ids, next_id), dim=0).contiguous(),
        torch.cat((prompt_embeds, next_embed), dim=0).contiguous(),
    )


def float_audio_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    audio = audio.detach().float().clamp(-1.0, 1.0)
    pcm = (audio * 32767.0).to(torch.int16).numpy().tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# vLLM talker (native fork): hf_overrides swaps in talker_config as top-level hf_config
# -----------------------------------------------------------------------------

_MROPE_CFG_KEYS = frozenset({"mrope_section", "mrope_interleaved", "interleaved"})


def _strip_mrope_markers_from_talker_text_config(text_config: PretrainedConfig) -> None:
    """令 vLLM 在构图阶段即认为 Talker 走标准 1D RoPE，与 `get_rope`/runner 一致。

    `uses_mrope(hf_config)`（见 `vllm/transformers_utils/config.py`）只看 **加载后、model.__init__ 之前**
    的 `text_config.rope_parameters`。若在 `qwen3_omni_moe_talker.py` 里才删 `mrope_section`，则
    `ModelConfig.uses_mrope` 仍为 True：`gpu_model_runner` 按 M-RoPE 造 `positions`（如 `[3, seq]`），
    而 `get_rope` 已因运行时清洗得到 **标准** `RotaryEmbedding`：`forward_static` 里对 positions
    `flatten()` 得到 `3*seq` 个索引，却与 `query` 的行数 `seq` 做 `view` → Dynamo fake tensor 报
    `shape '[3*s18, -1, 128]' is invalid for input of size 512*s59`。

    因此在 `hf_overrides` 返回的 **深拷贝** `text_config` 上同步去掉 M-RoPE 标记（与 Talker 真实
    scaffold 一致），**不是**关闭 compile / 不是 `enforce_eager` 兜底。
    """
    rp = getattr(text_config, "rope_parameters", None)
    if isinstance(rp, dict):
        cleaned = {k: v for k, v in rp.items() if k not in _MROPE_CFG_KEYS}
        if cleaned.get("rope_type") == "mrope":
            cleaned = {**cleaned, "rope_type": "default"}
        text_config.rope_parameters = cleaned if cleaned else None
    rs = getattr(text_config, "rope_scaling", None)
    if isinstance(rs, dict):
        cleaned_s = {k: v for k, v in rs.items() if k not in _MROPE_CFG_KEYS}
        text_config.rope_scaling = cleaned_s if cleaned_s else None


def _talker_hf_overrides(cfg: PretrainedConfig) -> PretrainedConfig:
    """Loaded from full checkpoint root config; return talker subtree for vLLM."""
    if not hasattr(cfg, "talker_config"):
        raise ValueError("Checkpoint config has no talker_config (expected Qwen3 Omni root).")
    tc = copy.deepcopy(cfg.talker_config)
    tc.architectures = ["Qwen3OmniMoeTalkerForConditionalGeneration"]
    _strip_mrope_markers_from_talker_text_config(tc.text_config)
    return tc


def _mtp_rank0_one_tensor(x: Any, name: str, ctx: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().contiguous()
    if isinstance(x, (list, tuple)):
        if not x:
            raise RuntimeError(f"{ctx}: {name} is empty sequence after IPC")
        el = x[0]
        if not isinstance(el, torch.Tensor):
            raise RuntimeError(
                f"{ctx}: {name}[0] expected torch.Tensor, got {type(el)} "
                f"(undecoded tuple IPC? ensure vllm_mtp_run_rpc returns a list of tensors first)"
            )
        return el.detach().cpu().contiguous()
    raise RuntimeError(f"{ctx}: {name} must be Tensor or list/tuple of Tensor, got {type(x)}")


def _apply_model_rank0_mtp(
    outs: list[Any],
    *,
    ctx: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], dict[str, Any]]:
    """解析 ``vllm_mtp_run_rpc`` 的 ``outs[0]``：两枚 CPU Tensor + 可选 GPU 毫秒/profile。"""
    if not outs:
        raise RuntimeError(f"{ctx}: apply_model returned empty list")
    raw = outs[0]
    if not isinstance(raw, (list, tuple)):
        raise RuntimeError(f"{ctx}: rank0 return must be list/tuple, got {type(raw)}")
    n = len(raw)
    if n not in (2, 3, 6, 7):
        raise RuntimeError(
            f"{ctx}: rank0 MTP return must have length 2/3 (tensors + optional profile) "
            f"or 6/7 (+ GPU ms + optional profile), got {n}"
        )
    full_codes = _mtp_rank0_one_tensor(raw[0], "full_codes", ctx)
    summed_embedding = _mtp_rank0_one_tensor(raw[1], "summed_embedding", ctx)
    gpu: dict[str, float] = {}
    profile: dict[str, Any] = {}
    if n in (6, 7):
        try:
            gpu = {
                "mtp_gpu_prep_ms": float(raw[2]),
                "mtp_gpu_forward_ms": float(raw[3]),
                "mtp_gpu_post_ms": float(raw[4]),
                "mtp_gpu_total_ms": float(raw[5]),
            }
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"{ctx}: bad GPU timing tail from worker: {raw[2:6]!r}") from e
    if n in (3, 7):
        maybe_profile = raw[2] if n == 3 else raw[6]
        if not isinstance(maybe_profile, dict):
            raise RuntimeError(f"{ctx}: profile tail must be dict, got {type(maybe_profile)}")
        profile = maybe_profile
    return full_codes, summed_embedding, gpu, profile


class TalkerEngine:
    """Talker MoE 与 MTP 均在 vLLM worker；host 仅 codec 嵌入（``TalkerMtpEmbeddings``）与 Code2Wav（HF）。"""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self._mtp_split = _mtp_split_engine_enabled()
        self._mtp_proc: multiprocessing.Process | None = None
        self._mtp_ipc: Connection | None = None
        if not self._mtp_split:
            LOGGER.warning(
                "TALKER_MTP_SPLIT_ENGINE=0（shared_moetalker）：启动阶段 **只有** MoE 的「Capturing CUDA graphs」，"
                "**不会出现** MTP 专用第二套图捕获；需要 MTP 子进程 tqdm 请 unset 本变量或设为 1（**默认 1**）。"
            )
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.root_config = AutoConfig.from_pretrained(cfg.model, trust_remote_code=True)
        self.dtype = _torch_dtype_from_config(getattr(self.root_config, "dtype", None))
        self.mtp_dtype = _mtp_inference_dtype(self.device)
        self._num_code_groups = int(self.root_config.talker_config.code_predictor_config.num_code_groups)

        prefix_cache = os.environ.get("TALKER_PREFIX_CACHE", "1") == "1"
        gpu_util = float(os.environ.get("TALKER_GPU_MEM_UTIL", "0.7"))
        enforce_eager = os.environ.get("TALKER_ENFORCE_EAGER", "0") == "1"

        LOGGER.info(
            "Loading vLLM talker tp=%s max_model_len=%s prefix_cache=%s enforce_eager=%s",
            cfg.tensor_parallel_size,
            cfg.max_model_len,
            prefix_cache,
            enforce_eager,
        )
        LOGGER.info(
            "TalkerEngine: root dtype=%s → Code2Wav/aux=%s; host codec 嵌入 dtype=%s; MTP 在 vLLM worker（num_code_groups=%s）",
            getattr(self.root_config, "dtype", None),
            self.dtype,
            self.mtp_dtype,
            self._num_code_groups,
        )
        LOGGER.info(
            "MTP RPC: TALKER_MTP_CUDAGRAPH_MODE=%r（match：组合 cudagraph 优先 mixed_mode/PIECEWISE + BatchDescriptor，"
            "与 Inductor 分区 CUDAGraphWrapper 对齐；**独立 MTP 引擎** 时另有子进程内捕获）。"
            " prefix cache 仍仅 generate() KV；MTP 子步 KV 见 code_predictor DynamicCache。",
            os.environ.get("TALKER_MTP_CUDAGRAPH_MODE", "match"),
        )
        if self._mtp_split:
            LOGGER.info(
                "将在 MoE LLM 就绪后 **fork MTP 子进程**（默认 CUDA_VISIBLE_DEVICES=0）；请继续往下看 **第二段** "
                "Loading safetensors / Capturing CUDA graphs（来自 talker_mtp_worker）。"
            )
        self.llm = LLM(
            model=cfg.model,
            trust_remote_code=True,
            hf_overrides=_talker_hf_overrides,
            enable_prompt_embeds=True,
            skip_tokenizer_init=True,
            max_model_len=cfg.max_model_len,
            tensor_parallel_size=cfg.tensor_parallel_size,
            dtype="auto",
            gpu_memory_utilization=gpu_util,
            enforce_eager=enforce_eager,
            enable_prefix_caching=prefix_cache,
        )
        if self._mtp_split:
            self._mtp_proc, self._mtp_ipc = _spawn_mtp_vllm_subprocess(cfg)
        self.mtp = TalkerMtpEmbeddings(
            cfg.model,
            self.root_config.talker_config,
            self.device,
            self.mtp_dtype,
        )
        self.code2wav = Qwen3OmniMoeCode2Wav._from_config(self.root_config.code2wav_config)
        self._load_code2wav_weights()
        self.code2wav.to(device=self.device, dtype=self.dtype).eval()
        if self._mtp_split:
            LOGGER.info(
                "TalkerEngine init done: mtp_engine=subproc_vllm（MTP 在子进程独立 LLM，日志中应有第二套 Capturing CUDA graphs；"
                "MoE 仍为当前进程 self.llm）。 model=%s talker_tp=%s mtp_subproc_cuda_visible=%r "
                "TALKER_MTP_SUBPROC_TP=%s max_model_len=%s TALKER_MTP_CUDAGRAPH_MODE=%r "
                "TALKER_MTP_VERBOSE_LOG=%r TALKER_MTP_WORKER_VERBOSE=%r TALKER_MTP_PROFILE=%r",
                cfg.model,
                cfg.tensor_parallel_size,
                os.environ.get("TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES", ""),
                os.environ.get("TALKER_MTP_SUBPROC_TP", os.environ.get("TALKER_MTP_TP", "1")),
                cfg.max_model_len,
                os.environ.get("TALKER_MTP_CUDAGRAPH_MODE", "match"),
                os.environ.get("TALKER_MTP_VERBOSE_LOG", ""),
                os.environ.get("TALKER_MTP_WORKER_VERBOSE", ""),
                os.environ.get("TALKER_MTP_PROFILE", ""),
            )
        else:
            LOGGER.info(
                "TalkerEngine init done: mtp_engine=shared_moetalker (MTP 经本 LLM.apply_model→vllm_mtp_run_rpc，"
                "与 MoE 共进程/共 compilation_config；启动时 Capturing CUDA graphs 仅覆盖 language_model 主路径)。"
                " model=%s talker_tp=%s max_model_len=%s TALKER_MTP_CUDAGRAPH_MODE=%r "
                "TALKER_MTP_VERBOSE_LOG=%r TALKER_MTP_WORKER_VERBOSE=%r TALKER_MTP_PROFILE=%r",
                cfg.model,
                cfg.tensor_parallel_size,
                cfg.max_model_len,
                os.environ.get("TALKER_MTP_CUDAGRAPH_MODE", "match"),
                os.environ.get("TALKER_MTP_VERBOSE_LOG", ""),
                os.environ.get("TALKER_MTP_WORKER_VERBOSE", ""),
                os.environ.get("TALKER_MTP_PROFILE", ""),
            )

    def shutdown(self) -> None:
        if self._mtp_ipc is not None:
            try:
                self._mtp_ipc.send("shutdown")
            except Exception:
                pass
            try:
                self._mtp_ipc.close()
            except Exception:
                pass
            self._mtp_ipc = None
        if self._mtp_proc is not None:
            self._mtp_proc.join(timeout=300.0)
            if self._mtp_proc.is_alive():
                LOGGER.warning("MTP 子进程未在超时内退出，发送 terminate")
                self._mtp_proc.terminate()
                self._mtp_proc.join(timeout=30.0)
            self._mtp_proc = None

    def _load_code2wav_weights(self) -> None:
        sd = self.code2wav.state_dict()
        loaded: dict[str, torch.Tensor] = {}
        for name, tensor in iter_weight_tensors(self.cfg.model, prefixes=("code2wav.",)):
            key = name[len("code2wav."):]
            if key in sd:
                loaded[key] = tensor
        self.code2wav.load_state_dict(loaded, strict=False)

    def _project(self, emb: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        outs = self.llm.apply_model(
            partial(
                vllm_apply_projection_rpc,
                emb=emb.contiguous(),
                top=top.contiguous(),
            )
        )
        return outs[0]

    def _mtp_vllm_run(
        self,
        last_talker_hidden: torch.Tensor,
        layer0_token_id: int,
        *,
        timing_acc: dict[str, Any] | None = None,
    ) -> MtpOutput:
        """单次 MTP：worker 内 ``code_predictor`` + ``apply_model``；无 HF fallback。"""
        last_cpu = last_talker_hidden.detach().cpu().contiguous()
        if last_cpu.ndim == 1:
            last_cpu = last_cpu.unsqueeze(0)
        vlog = _mtp_verbose_host()
        t_rpc0 = time.perf_counter()
        if self._mtp_split:
            ipc = self._mtp_ipc
            if ipc is None:
                raise RuntimeError("MTP split-engine 未初始化 IPC")
            ipc.send(
                {
                    "op": "mtp",
                    "last_h": last_cpu,
                    "layer0_token_id": int(layer0_token_id),
                    "temperature": float(self.cfg.temperature),
                    "top_k": int(self.cfg.top_k),
                    "top_p": float(self.cfg.top_p),
                }
            )
            resp = ipc.recv()
            if not isinstance(resp, dict) or not resp.get("ok"):
                tb = resp.get("tb", "") if isinstance(resp, dict) else ""
                err = resp.get("err", resp) if isinstance(resp, dict) else resp
                raise RuntimeError(f"MTP 子进程失败: {err}\n{tb}")
            outs = resp["outs"]
        else:
            outs = self.llm.apply_model(
                partial(
                    vllm_mtp_run_rpc,
                    last_talker_hidden=last_cpu,
                    layer0_token_id=int(layer0_token_id),
                    temperature=self.cfg.temperature,
                    top_k=self.cfg.top_k,
                    top_p=self.cfg.top_p,
                )
            )
        rpc_wall_s = time.perf_counter() - t_rpc0
        full_codes, summed, gpu_ms, profile = _apply_model_rank0_mtp(outs, ctx="vllm_mtp_run_rpc")
        if timing_acc is not None and gpu_ms:
            for k, v in gpu_ms.items():
                timing_acc[k] = float(timing_acc.get(k, 0.0)) + float(v)
        if timing_acc is not None and profile:
            _merge_mtp_profile(timing_acc, profile)
        if vlog:
            gpu_total_ms = float(gpu_ms.get("mtp_gpu_total_ms", 0.0)) if gpu_ms else 0.0
            rpc_overhead_ms = max(0.0, rpc_wall_s * 1000.0 - gpu_total_ms) if gpu_ms else None
            profile_top = {}
            if profile and isinstance(profile.get("top_ms"), list):
                profile_top = {
                    str(item.get("name")): float(item.get("ms", 0.0))
                    for item in profile["top_ms"][:8]
                    if isinstance(item, dict)
                }
            elif profile and isinstance(profile.get("ms"), dict):
                profile_top = {
                    str(k): round(float(v), 4)
                    for k, v in sorted(
                        profile["ms"].items(),
                        key=lambda item: float(item[1]) if isinstance(item[1], (int, float)) else 0.0,
                        reverse=True,
                    )[:8]
                    if isinstance(v, (int, float))
                }
            LOGGER.info(
                "MTP apply_model: layer0_token_id=%s last_h_shape=%s rpc_wall_ms=%.3f "
                "gpu_ms prep=%s forward=%s post=%s total=%s rpc_overhead_ms=%s "
                "full_codes_shape=%s summed_shape=%s profile_calls=%s profile_top_ms=%s",
                int(layer0_token_id),
                tuple(last_cpu.shape),
                rpc_wall_s * 1000.0,
                round(float(gpu_ms.get("mtp_gpu_prep_ms", 0.0)), 4) if gpu_ms else None,
                round(float(gpu_ms.get("mtp_gpu_forward_ms", 0.0)), 4) if gpu_ms else None,
                round(float(gpu_ms.get("mtp_gpu_post_ms", 0.0)), 4) if gpu_ms else None,
                round(gpu_total_ms, 4) if gpu_ms else None,
                round(float(rpc_overhead_ms), 4) if rpc_overhead_ms is not None else None,
                tuple(full_codes.shape),
                tuple(summed.shape),
                profile.get("calls") if profile else None,
                json.dumps(profile_top, ensure_ascii=False),
            )
        return MtpOutput(full_codes=full_codes, summed_embedding=summed)

    def _vllm_talker_step(
        self,
        prompt_token_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        cache_salt: str,
        *,
        timing_acc: dict[str, Any] | None = None,
    ) -> tuple[int, torch.Tensor]:
        sp = SamplingParams(
            max_tokens=1,
            detokenize=False,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
        )
        if prompt_token_ids.shape[0] != prompt_embeds.shape[0]:
            raise RuntimeError(
                f"talker prompt mismatch ids={prompt_token_ids.shape[0]} emb={prompt_embeds.shape[0]}"
            )
        prompt: dict[str, Any] = {
            "prompt_token_ids": prompt_token_ids.tolist(),
            "prompt_embeds": prompt_embeds.cpu().contiguous(),
            "cache_salt": cache_salt,
        }
        t_g0 = time.perf_counter()
        req = self.llm.generate([prompt], sp, use_tqdm=False)[0]
        if timing_acc is not None:
            timing_acc["vllm_generate_s"] = timing_acc.get("vllm_generate_s", 0.0) + (
                time.perf_counter() - t_g0
            )
        out0 = req.outputs[0]
        if not out0.token_ids:
            raise RuntimeError("vLLM talker returned no token")
        tid = int(out0.token_ids[0])
        t_h0 = time.perf_counter()
        rows = self.llm.apply_model(vllm_read_last_sample_hidden_cpu)
        if timing_acc is not None:
            timing_acc["vllm_apply_model_s"] = timing_acc.get("vllm_apply_model_s", 0.0) + (
                time.perf_counter() - t_h0
            )
        h = rows[0]
        if h is None:
            raise RuntimeError("vLLM talker did not capture hidden (compute_logits hook)")
        return tid, h[-1:].contiguous()

    @torch.inference_mode()
    def decode_wav(self, rvq: torch.Tensor) -> bytes | None:
        if rvq.numel() == 0:
            return None
        x = rvq.to(device=self.device, dtype=torch.long)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        wavs = self.code2wav.chunked_decode(
            x,
            chunk_size=self.cfg.chunk_size_code2wav,
            left_context_size=self.cfg.left_context_code2wav,
        )
        w = wavs.squeeze(0).detach().cpu() if wavs is not None else None
        if w is None:
            return None
        return float_audio_to_wav_bytes(w, self.cfg.sample_rate)

    def process_chunk(
        self,
        session: SessionState,
        text_embedding: torch.Tensor,
        top_hidden_state: torch.Tensor,
        *,
        cache_salt: str,
    ) -> tuple[bytes | None, dict[str, Any]]:
        t0 = time.perf_counter()
        timings: dict[str, Any] = {
            "prefill_s": 0.0,
            "vllm_generate_s": 0.0,
            "vllm_apply_model_s": 0.0,
            "mtp_s": 0.0,
            "code2wav_s": 0.0,
            "rvq_cat_s": 0.0,
            "mtp_cuda_sync": int(_mtp_cuda_sync_enabled()),
            "mtp_profile_enabled": int(_mtp_profile_enabled()),
            "mtp_model_prefill_s": 0.0,
            "mtp_model_decode_s": 0.0,
            "mtp_iter_tail_s": 0.0,
            "mtp_run_prep_s": 0.0,
            "mtp_run_summed_s": 0.0,
            "mtp_cpu_offload_s": 0.0,
            "mtp_gpu_prep_ms": 0.0,
            "mtp_gpu_forward_ms": 0.0,
            "mtp_gpu_post_ms": 0.0,
            "mtp_gpu_total_ms": 0.0,
            "mtp_avg_ms_gpu_prep": 0.0,
            "mtp_avg_ms_gpu_forward": 0.0,
            "mtp_avg_ms_gpu_post": 0.0,
            "mtp_avg_ms_gpu_total": 0.0,
            "mtp_rpc_overhead_s": 0.0,
        }
        if text_embedding.shape != top_hidden_state.shape:
            raise RuntimeError(
                f"shape mismatch: emb {text_embedding.shape} vs top {top_hidden_state.shape}"
            )
        t_pf0 = time.perf_counter()
        prompt_ids, prompt_embeds, cur_cond_cpu, cur_cond_ids = build_prefill_embeds(
            self._project,
            self.mtp,
            session,
            text_embedding,
            top_hidden_state,
        )
        timings["prefill_s"] = time.perf_counter() - t_pf0

        full_groups: list[torch.Tensor] = []
        layer0_ids: list[int] = []
        eos_hit = False
        last_tid: int | None = None
        vllm_loop_iters = 0
        mtp_calls = 0
        for _ in range(self.cfg.talker_decode_steps):
            vllm_loop_iters += 1
            tid, last_h = self._vllm_talker_step(
                prompt_ids,
                prompt_embeds,
                cache_salt,
                timing_acc=timings,
            )
            last_tid = tid
            if tid == self.mtp.codec_eos_id:
                eos_hit = True
                break
            if len(layer0_ids) >= self.cfg.real_codec_steps_per_turn:
                break
            t_m0 = time.perf_counter()
            mtp_out = self._mtp_vllm_run(last_h, tid, timing_acc=timings)
            timings["mtp_s"] += time.perf_counter() - t_m0
            mtp_calls += 1
            full_groups.append(mtp_out.full_codes)
            layer0_ids.append(tid)
            prompt_ids, prompt_embeds = append_codec_step(
                prompt_ids,
                prompt_embeds,
                tid,
                mtp_out.summed_embedding,
            )

        has_six = (
            len(layer0_ids) == self.cfg.real_codec_steps_per_turn
            and len(full_groups) == self.cfg.real_codec_steps_per_turn
        )
        valid = eos_hit and has_six
        t_engine = time.perf_counter() - t0
        timings["engine_wall_s"] = t_engine
        timings["vllm_step_count"] = float(vllm_loop_iters)
        timings["mtp_call_count"] = float(mtp_calls)
        timings["vllm_subtotal_s"] = timings["vllm_generate_s"] + timings["vllm_apply_model_s"]
        # MTP：优先使用 worker **CUDA Event** 回传的毫秒（``mtp_gpu_*_ms``）；否则按子步数对 **RPC 墙钟**
        # ``mtp_s`` 比例分摊（旧行为）。
        dec_per_run = self._num_code_groups - 1
        if mtp_calls > 0 and dec_per_run > 0:
            prefill_n = mtp_calls
            decode_n = mtp_calls * (dec_per_run - 1) if dec_per_run > 1 else 0
            substeps = mtp_calls * dec_per_run
            timings["mtp_substep_count"] = substeps
            timings["mtp_prefill_substep_count"] = prefill_n
            timings["mtp_decode_substep_count"] = decode_n
            mtp_wall = float(timings["mtp_s"])
            gpu_tot_ms = float(timings.get("mtp_gpu_total_ms", 0.0))
            if gpu_tot_ms > 0.0:
                timings["mtp_run_prep_s"] = float(timings["mtp_gpu_prep_ms"]) / 1000.0
                timings["mtp_model_decode_s"] = float(timings["mtp_gpu_forward_ms"]) / 1000.0
                timings["mtp_run_summed_s"] = float(timings["mtp_gpu_post_ms"]) / 1000.0
                timings["mtp_model_total_s"] = gpu_tot_ms / 1000.0
                timings["mtp_rpc_overhead_s"] = max(0.0, mtp_wall - timings["mtp_model_total_s"])
                timings["mtp_avg_ms_gpu_prep"] = float(timings["mtp_gpu_prep_ms"]) / mtp_calls
                timings["mtp_avg_ms_gpu_forward"] = float(timings["mtp_gpu_forward_ms"]) / mtp_calls
                timings["mtp_avg_ms_gpu_post"] = float(timings["mtp_gpu_post_ms"]) / mtp_calls
                timings["mtp_avg_ms_gpu_total"] = gpu_tot_ms / mtp_calls
                # 与旧字段对齐：prefill 桶放 GPU prep（H2D+embed），decode 桶放 ``cp.forward``
                timings["mtp_model_prefill_s"] = timings["mtp_run_prep_s"]
                timings["mtp_avg_ms_model_prefill"] = timings["mtp_avg_ms_gpu_prep"]
                timings["mtp_avg_ms_model_decode"] = timings["mtp_avg_ms_gpu_forward"]
                timings["mtp_avg_ms_model_substep"] = (gpu_tot_ms / substeps) if substeps else 0.0
            else:
                timings["mtp_model_total_s"] = mtp_wall
                timings["mtp_model_prefill_s"] = mtp_wall * (prefill_n / substeps)
                timings["mtp_model_decode_s"] = mtp_wall * (decode_n / substeps)
                timings["mtp_avg_ms_model_prefill"] = (
                    (timings["mtp_model_prefill_s"] / prefill_n) * 1000.0 if prefill_n else 0.0
                )
                timings["mtp_avg_ms_model_decode"] = (
                    (timings["mtp_model_decode_s"] / decode_n) * 1000.0 if decode_n else 0.0
                )
                timings["mtp_avg_ms_model_substep"] = (mtp_wall / substeps) * 1000.0
                timings["mtp_rpc_overhead_s"] = 0.0
            timings["mtp_avg_ms_iter_tail"] = 0.0
            if gpu_tot_ms > 0.0:
                accounted = (
                    float(timings["mtp_run_prep_s"])
                    + float(timings["mtp_model_decode_s"])
                    + float(timings["mtp_run_summed_s"])
                    + float(timings["mtp_cpu_offload_s"])
                    + float(timings["mtp_iter_tail_s"])
                )
                ref_wall = float(timings["mtp_model_total_s"])
            else:
                accounted = (
                    float(timings["mtp_model_prefill_s"])
                    + float(timings["mtp_model_decode_s"])
                    + float(timings["mtp_run_prep_s"])
                    + float(timings["mtp_run_summed_s"])
                    + float(timings["mtp_cpu_offload_s"])
                    + float(timings["mtp_iter_tail_s"])
                )
                ref_wall = mtp_wall
            timings["mtp_internal_accounted_s"] = accounted
            timings["mtp_internal_residual_s"] = max(0.0, ref_wall - accounted)
        else:
            timings["mtp_substep_count"] = 0
            timings["mtp_prefill_substep_count"] = 0
            timings["mtp_decode_substep_count"] = 0
        _finalize_mtp_profile(timings)
        meta: dict[str, Any] = {
            "valid_turn": valid,
            "eos_emitted": eos_hit,
            "layer0_steps": len(layer0_ids),
            "codec_eos_id": self.mtp.codec_eos_id,
            "last_layer0_token_id": last_tid,
            "elapsed_s": t_engine,
            "mtp_engine": "subproc_vllm" if self._mtp_split else "shared_moetalker",
            "mtp_rpc_path": (
                "Pipe->mtp_worker_main->LLM.apply_model(vllm_mtp_run_rpc)"
                if self._mtp_split
                else "LLM.apply_model(server_talker_rpc.vllm_mtp_run_rpc)"
            ),
        }
        if self._mtp_split:
            meta["mtp_subproc_cuda_visible"] = os.environ.get(
                "TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES", ""
            )
            meta["mtp_subproc_pid"] = self._mtp_proc.pid if self._mtp_proc is not None else None
        if not has_six:
            meta["timings"] = _serialize_timings(timings)
            return None, meta

        t_cat0 = time.perf_counter()
        pieces = [g if g.ndim == 2 else g.squeeze(0) for g in full_groups]
        rvq = torch.cat(pieces, dim=-1)
        timings["rvq_cat_s"] = time.perf_counter() - t_cat0
        t_c2w0 = time.perf_counter()
        wav = self.decode_wav(rvq)
        timings["code2wav_s"] = time.perf_counter() - t_c2w0
        meta["timings"] = _serialize_timings(timings)
        if wav is None:
            meta["wav_decode_failed"] = True
            return None, meta
        if valid:
            session.history.append(
                HistoryBlock(
                    codes=rvq.cpu().contiguous(),
                    conditioning_embeds=cur_cond_cpu,
                    conditioning_token_ids=cur_cond_ids.clone(),
                    real_codec_steps=len(layer0_ids),
                )
            )
        meta["rvq_shape"] = list(rvq.shape)
        meta["tensor_parallel_size"] = self.cfg.tensor_parallel_size
        return wav, meta


# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------

sessions: dict[str, SessionState] = {}
engine: TalkerEngine | None = None
session_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    cfg = _server_config_from_env()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Talker MoE 仅支持 vLLM + CUDA。请在有 GPU 的环境启动，或改走其他服务；"
            "本进程不再加载 HuggingFace Talker 主干。"
        )
    LOGGER.info(
        "Loading vLLM talker model=%s tp=%s device=%s",
        cfg.model,
        cfg.tensor_parallel_size,
        cfg.device,
    )
    engine = TalkerEngine(cfg)
    yield
    if engine is not None:
        engine.shutdown()
    engine = None
    sessions.clear()


app = FastAPI(title="server_talker", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.delete("/v1/talker/session/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    async with session_lock:
        sessions.pop(session_id, None)
    return {"session_id": session_id, "cleared": "true"}


async def _process_turn_payload_bytes(
    *,
    session_id: str,
    raw: bytes,
    source_name: str,
    internal_mode: bool,
) -> Response:
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    t_handler0 = time.perf_counter()
    http_read_s = 0.0
    try:
        t_ld0 = time.perf_counter()
        bio = io.BytesIO(raw)
        payload = torch.load(bio, map_location="cpu", weights_only=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"torch.load failed: {e}") from e

    turns = payload.get("turns") or []
    if not turns:
        raise HTTPException(status_code=400, detail="missing turns in pt")
    t0 = turns[0]
    top = t0.get("top_hidden_state")
    emb = t0.get("text_embedding")
    if not torch.is_tensor(top) or not torch.is_tensor(emb):
        raise HTTPException(status_code=400, detail="top_hidden_state / text_embedding must be tensors")
    payload_decode_validate_s = time.perf_counter() - t_ld0

    t_lock0 = time.perf_counter()
    async with session_lock:
        lock_wait_s = time.perf_counter() - t_lock0
        t_eng0 = time.perf_counter()
        st = sessions.setdefault(session_id, SessionState())
        wav_bytes, meta = engine.process_chunk(st, emb, top, cache_salt=session_id)
        process_chunk_in_lock_s = time.perf_counter() - t_eng0
    handler_wall_s = time.perf_counter() - t_handler0

    timings = meta.setdefault("timings", {})
    timings["http_read_s"] = round(http_read_s, 6)
    timings["payload_decode_validate_s"] = round(payload_decode_validate_s, 6)
    timings["session_lock_wait_s"] = round(lock_wait_s, 6)
    timings["process_chunk_in_lock_s"] = round(process_chunk_in_lock_s, 6)
    timings["handler_wall_s"] = round(handler_wall_s, 6)
    mtp_profile = timings.get("mtp_profile")
    mtp_profile_top = mtp_profile.get("top_ms") if isinstance(mtp_profile, dict) else None

    LOGGER.info(
        "talker_chunk file=%s session_id=%s valid_turn=%s mtp_engine=%s mtp_subproc_gpus=%s mtp_calls=%s "
        "handler_wall_s=%.4f mtp_profile_top_ms=%s timings=%s",
        source_name,
        session_id,
        meta.get("valid_turn"),
        meta.get("mtp_engine"),
        meta.get("mtp_subproc_cuda_visible") or "-",
        timings.get("mtp_call_count"),
        handler_wall_s,
        json.dumps(mtp_profile_top, ensure_ascii=False),
        json.dumps(timings, ensure_ascii=False),
    )

    headers = {"X-Talker-Meta": json.dumps(meta, ensure_ascii=True, separators=(",", ":"))}
    if wav_bytes is None:
        if not internal_mode:
            return Response(
                content=json.dumps({"ok": False, "meta": meta}, ensure_ascii=False).encode("utf-8"),
                media_type="application/json",
                status_code=200,
                headers=headers,
            )
        return Response(status_code=204, headers=headers)
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@app.post("/v1/talker/chunk")
async def talker_chunk(
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> Response:
    raw = await file.read()
    source_name = getattr(file, "filename", None) or "upload.pt"
    return await _process_turn_payload_bytes(
        session_id=session_id,
        raw=raw,
        source_name=source_name,
        internal_mode=False,
    )


@app.post("/internal/talker/turn/{session_id}")
async def internal_talker_turn(session_id: str, request: Request) -> Response:
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="empty request body")
    return await _process_turn_payload_bytes(
        session_id=session_id,
        raw=raw,
        source_name="internal.turn",
        internal_mode=True,
    )


def main() -> None:
    import argparse

    b = ServerConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("TALKER_HOST", b.host))
    p.add_argument("--port", type=int, default=int(os.environ.get("TALKER_PORT", str(b.port))))
    p.add_argument("--model", default=os.environ.get("TALKER_MODEL", b.model))
    p.add_argument("--device", default=os.environ.get("TALKER_DEVICE", b.device))
    p.add_argument("--tp", type=int, default=int(os.environ.get("TALKER_TP", str(b.tensor_parallel_size))), help="TALKER_TP tensor parallel size")
    args = p.parse_args()
    os.environ["TALKER_HOST"] = args.host
    os.environ["TALKER_PORT"] = str(args.port)
    os.environ["TALKER_MODEL"] = args.model
    os.environ["TALKER_DEVICE"] = args.device
    os.environ["TALKER_TP"] = str(args.tp)
    uvicorn.run(
        "server_talker:app",
        host=args.host,
        port=args.port,
        factory=False,
        log_level="info",
    )


sys.modules.setdefault("server_talker", sys.modules[__name__])

if __name__ == "__main__":
    main()
