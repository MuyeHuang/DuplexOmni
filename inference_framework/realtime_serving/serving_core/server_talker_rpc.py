"""
Callbacks passed to ``LLM.apply_model`` / ``collective_rpc``.

These **must** live in a real importable module. If defined in ``server_talker.py`` and you run
``python server_talker.py``, their ``__qualname__`` resolves to ``__main__.<fn>``. vLLM workers
pickle the callable and unpickle in a process whose ``__main__`` is **not** ``server_talker``,
so unpickling fails with: "not the same object as __main__._vllm_...".

See ``talker_修改日志.md`` 第 12 次修改.

**MTP 性能（CUDA 图 / 计时）**

- **Prefix cache**：``enable_prefix_caching`` 只作用于 **``generate()``** 的 **KV 块复用**；
  **``vllm_mtp_run_rpc``** 走 **``apply_model``**，**不经过** block manager，**没有** talker 级 prefix cache。
- **CUDA 图**：默认 **``match``** 时组合模式优先 **``mixed_mode()``**（**``FULL_AND_PIECEWISE`` → PIECEWISE**），与 **Inductor 分区** 上 **``CUDAGraphWrapper(PIECEWISE)``** 对齐；并设置 **``BatchDescriptor``**（外层 2 token + ``code_predictor`` 内按段更新）。**``full`` / ``piecewise`` / ``none``** 仍可强制。MoE 启动 tqdm 仍只覆盖 **language_model**；MTP 子图在 **首次** 命中形状时 **惰性捕获**（见 ``CUDAGraphWrapper``）。
- **细粒度 GPU 时间**：在 worker 上用 **``torch.cuda.Event``** 分段（prep / ``cp.forward`` / post），经返回值尾部 **4 个 float（毫秒）** 回传。
- **日志**：默认开启 **``TALKER_MTP_VERBOSE_LOG=1``** / **``TALKER_MTP_WORKER_VERBOSE=1``**。
  Full-call CUDA graph 路径要求 **``TALKER_MTP_PROFILE=0``**（capture/replay 期间禁止 profile 插桩）。
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from vllm.config import CUDAGraphMode, get_current_vllm_config_or_none
from vllm.forward_context import BatchDescriptor, create_forward_context, override_forward_context
from vllm.logger import init_logger

_vllm_rpc_log = init_logger(__name__)


class _MtpCudaGraphRunner:
    """Capture/replay one full ``cp.forward`` call with fixed-shape CUDA graph."""

    def __init__(
        self,
        model: nn.Module,
        cp: nn.Module,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        warmup_iters: int = 3,
    ) -> None:
        self.model = model
        self.cp = cp
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.warmup_iters = int(max(1, warmup_iters))
        self.dev = next(model.parameters()).device
        self.dt = next(model.parameters()).dtype
        if self.dev.type != "cuda":
            raise RuntimeError("MTP full-call CUDA graph requires CUDA device.")
        hs = int(model.config.text_config.hidden_size)
        self.static_last_h = torch.empty((1, 1, hs), device=self.dev, dtype=self.dt)
        self.static_layer0_code = torch.empty((1, 1), device=self.dev, dtype=torch.long)
        self.graph = torch.cuda.CUDAGraph()
        self.pos_all: torch.Tensor | None = None
        self.current_input: torch.Tensor | None = None
        self._capture()

    def _capture(self) -> None:
        if _env_enabled("TALKER_MTP_PROFILE", default=True):
            raise RuntimeError(
                "MTP full-call CUDA graph requires TALKER_MTP_PROFILE=0 "
                "(profiling instrumentation is not allowed during capture/replay)."
            )
        dummy_h = torch.zeros_like(self.static_last_h)
        dummy_token = 0
        for _ in range(self.warmup_iters):
            self.static_last_h.copy_(dummy_h)
            self.static_layer0_code.fill_(dummy_token)
            layer0_embed = self.model.embed_input_ids(self.static_layer0_code)
            self.cp.forward(self.static_layer0_code, layer0_embed, self.static_last_h)
        torch.cuda.synchronize(device=self.dev)
        try:
            with torch.cuda.graph(self.graph):
                layer0_embed = self.model.embed_input_ids(self.static_layer0_code)
                self.pos_all, self.current_input = self.cp.forward(
                    self.static_layer0_code,
                    layer0_embed,
                    self.static_last_h,
                )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "MTP full-call CUDA graph capture failed for cp.forward. "
                f"This path is fail-fast by design (no fallback). cause={type(exc).__name__}: {exc}"
            ) from exc
        if self.pos_all is None or self.current_input is None:
            raise RuntimeError("MTP full-call CUDA graph capture produced empty outputs.")

    def replay(self, *, last_h: torch.Tensor, layer0_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.static_last_h.copy_(last_h)
        self.static_layer0_code.fill_(int(layer0_token_id))
        try:
            self.graph.replay()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "MTP full-call CUDA graph replay failed. This path does not allow fallback."
            ) from exc
        assert self.pos_all is not None
        assert self.current_input is not None
        return self.pos_all, self.current_input


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _profile_top_ms(profile: dict[str, Any], limit: int = 8) -> dict[str, float]:
    ms = profile.get("ms")
    if not isinstance(ms, dict):
        return {}
    items = sorted(
        ((str(k), float(v)) for k, v in ms.items() if isinstance(v, (int, float))),
        key=lambda item: item[1],
        reverse=True,
    )
    return {k: round(v, 4) for k, v in items[:limit]}


def _mtp_resolve_cudagraph_runtime_mode(raw: CUDAGraphMode) -> CUDAGraphMode:
    """Map ``CompilationConfig.cudagraph_mode`` to a value legal for ``ForwardContext``.

    **Inductor 分区** 上的静态图包装器固定为 **``PIECEWISE``**（见
    ``maybe_use_cudagraph_partition_wrapper``）。若此处用 **``decode_mode()`` → FULL**，
    则 ``CUDAGraphWrapper`` 会因 ``runtime_mode`` 不匹配而 **始终走 eager**，**不会**
    捕获/回放 MTP 子图。

    对 **组合枚举**（如 ``FULL_AND_PIECEWISE``）：优先 **``mixed_mode()``**（通常为
    **PIECEWISE**）。**``FULL_DECODE_ONLY``** 的 ``mixed_mode()`` 为 **NONE** 时退回到
    **``decode_mode()``**（FULL）。
    """
    if raw.valid_runtime_modes():
        return raw
    if raw.separate_routine():
        m = raw.mixed_mode()
        if m.valid_runtime_modes() and m != CUDAGraphMode.NONE:
            return m
        return raw.decode_mode()
    return CUDAGraphMode.NONE


def vllm_project_worker(model: nn.Module, emb: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    e = emb.to(device=dev, dtype=dt)
    t = top.to(device=dev, dtype=dt)
    with torch.inference_mode():
        return (model.text_projection(e) + model.hidden_projection(t)).cpu().contiguous()


def vllm_apply_projection_rpc(model: nn.Module, emb: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
    return vllm_project_worker(model, emb, top)


def vllm_read_last_sample_hidden_cpu(model: nn.Module):
    return getattr(model, "_talker_last_sample_hidden_cpu", None)


def vllm_mtp_run_rpc(
    model: nn.Module,
    last_talker_hidden: torch.Tensor,
    layer0_token_id: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list:
    """Run MTP on vLLM worker ``model.code_predictor`` (single layer0 position); return CPU tensors.

    Matches worker ``Qwen3OmniMoeTalkerCodePredictor.forward`` + summed embedding construction
    from ``InlineMtpRunner._build_summed_embedding``. MTP runs entirely on GPU until this returns.

    Args:
        last_talker_hidden: CPU tensor, shape ``[1, H]`` (last decode row from ``compute_logits``).
        layer0_token_id: Layer-0 codec id for this step.
        temperature / top_k / top_p: Shared with talker main decode; applied onto
            ``code_predictor`` before ``torch.ops.vllm.qwen3_omni_code_predictor_sample``.
            When ``temperature <= 0`` we collapse MTP to greedy via ``top_k=1, top_p=1``.

    Returns:
        **list**：前两项为 **CPU Tensor** ``full_codes``、``summed_embedding``（**必须**为 list 容器以便
        EngineCore 解码，见第 25 次日志）。在 CUDA 上另有 **4 个 Python float（毫秒）**：
        ``prep_ms``（H2D + embed）、``forward_ms``（``cp.forward``）、``post_ms``（summed 拼接）、
        ``total_ms``（以上 GPU 区间总和，``elapsed_time`` 链式相加）。
    """
    cp = model.code_predictor
    mtp_temperature = float(temperature)
    mtp_top_k = int(top_k)
    mtp_top_p = float(top_p)
    processors = []
    if mtp_temperature <= 0.0:
        mtp_top_k = 1
        mtp_top_p = 1.0
    elif mtp_temperature != 1.0:
        processors.append(TemperatureLogitsWarper(temperature=mtp_temperature))
    if 0.0 < mtp_top_p < 1.0:
        processors.append(TopPLogitsWarper(top_p=mtp_top_p))
    if mtp_top_k > 0:
        processors.append(TopKLogitsWarper(top_k=mtp_top_k))
    cp.logits_processors = LogitsProcessorList(processors)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    use_cuda_timers = dev.type == "cuda"
    if not use_cuda_timers:
        raise RuntimeError("MTP full-call CUDA graph path requires CUDA worker; no fallback enabled.")
    ev: list[torch.cuda.Event] = []
    if use_cuda_timers:
        for _ in range(4):
            ev.append(torch.cuda.Event(enable_timing=True))

    # ``apply_model`` 不在正常 ``set_forward_context`` 路径内；fork 里 MTP 采样 op 从
    # ``get_forward_context().no_compile_layers[layer_name]`` 取 ``code_predictor`` 做 TopK/TopP。
    # Worker 上 **没有** ``set_current_vllm_config()``（仅引擎前向会设），故 **不能** 依赖
    # ``get_current_vllm_config()``；``Qwen3OmniMoeTalkerForConditionalGeneration`` 在构造时已挂
    # ``model.vllm_config``，用其与 ``create_forward_context`` 得到与捕获图一致的 ``static_forward_context``。
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        vllm_config = getattr(model, "vllm_config", None)
    if vllm_config is None:
        raise RuntimeError(
            "vllm_mtp_run_rpc: cannot resolve VllmConfig for ForwardContext "
            "(get_current_vllm_config_or_none() is unset and model has no vllm_config)."
        )
    cg_env = os.environ.get("TALKER_MTP_CUDAGRAPH_MODE", "match").strip().lower()
    if cg_env in ("none", "0", "false", "off"):
        cg_raw = CUDAGraphMode.NONE
    elif cg_env in ("full",):
        cg_raw = CUDAGraphMode.FULL
    elif cg_env in ("piecewise", "piece"):
        cg_raw = CUDAGraphMode.PIECEWISE
    else:
        cg_raw = vllm_config.compilation_config.cudagraph_mode
    cg_mode = _mtp_resolve_cudagraph_runtime_mode(cg_raw)
    # ``CUDAGraphWrapper`` 需要非空 ``batch_descriptor``；MTP 首段为 last_h∥layer0 → 2 token。
    mtp_outer_bd = BatchDescriptor(
        num_tokens=2,
        num_reqs=None,
        uniform=True,
        has_lora=False,
        num_active_loras=0,
    )
    _vllm_rpc_log.info_once(
        "vllm_mtp_run_rpc: MTP ForwardContext cudagraph_runtime_mode=%s (from config %s, "
        "TALKER_MTP_CUDAGRAPH_MODE=%r); batch_descriptor=%s. "
        "Compound modes map to mixed_mode() (PIECEWISE) when non-NONE so Inductor partition "
        "CUDAGraphWrappers can replay; per-segment descriptor is set inside code_predictor.forward.",
        getattr(cg_mode, "name", str(cg_mode)),
        getattr(cg_raw, "name", str(cg_raw)),
        cg_env,
        mtp_outer_bd,
        scope="local",
    )
    forward_ctx = create_forward_context(
        None,
        vllm_config,
        cudagraph_runtime_mode=cg_mode,
        batch_descriptor=mtp_outer_bd,
    )
    with override_forward_context(forward_ctx):
        with torch.inference_mode():
            if use_cuda_timers:
                ev[0].record()
            last_h = last_talker_hidden.to(device=dev, dtype=dt)
            if last_h.ndim == 1:
                last_h = last_h.unsqueeze(0)
            if last_h.ndim == 2:
                last_h = last_h.unsqueeze(1)
            runner = getattr(cp, "_talker_mtp_graph_runner", None)
            runner_key = (
                float(mtp_temperature),
                int(mtp_top_k),
                float(mtp_top_p),
                str(dev),
                str(dt),
                int(model.config.text_config.hidden_size),
            )
            if runner is None or getattr(cp, "_talker_mtp_graph_runner_key", None) != runner_key:
                runner = _MtpCudaGraphRunner(
                    model,
                    cp,
                    temperature=mtp_temperature,
                    top_k=mtp_top_k,
                    top_p=mtp_top_p,
                )
                cp._talker_mtp_graph_runner = runner
                cp._talker_mtp_graph_runner_key = runner_key
            if use_cuda_timers:
                ev[1].record()
            pos_all, current_input = runner.replay(last_h=last_h, layer0_token_id=layer0_token_id)
            if use_cuda_timers:
                ev[2].record()
            hs = int(model.config.text_config.hidden_size)
            layer0_for_sum = model.embed_input_ids(pos_all[:, 0, :])
            middle_hiddens = current_input[:, 2:-1, :]
            last_layer_code = pos_all[:, -1, :]
            last_layer_embed = cp.model.codec_embedding[-1](last_layer_code)
            if middle_hiddens.numel() == 0:
                summed = layer0_for_sum + last_layer_embed
            else:
                summed = layer0_for_sum + middle_hiddens.sum(dim=1, keepdim=False) + last_layer_embed
            summed = summed.reshape(-1, hs)
            if use_cuda_timers:
                ev[3].record()

    profile_dict: dict[str, Any] = {}
    out: list = [
        pos_all.squeeze(0).detach().cpu().contiguous(),
        summed.squeeze(0).detach().cpu().contiguous(),
    ]
    if use_cuda_timers:
        torch.cuda.synchronize(device=dev)
        prep_ms = float(ev[0].elapsed_time(ev[1]))
        forward_ms = float(ev[1].elapsed_time(ev[2]))
        post_ms = float(ev[2].elapsed_time(ev[3]))
        total_ms = float(ev[0].elapsed_time(ev[3]))
        out.extend([prep_ms, forward_ms, post_ms, total_ms])
    if profile_dict:
        out.append(profile_dict)
    if _env_enabled("TALKER_MTP_WORKER_VERBOSE", default=True):
        fc, sm = out[0], out[1]
        if len(out) >= 6:
            _vllm_rpc_log.info(
                "vllm_mtp_run_rpc done: layer0_token_id=%s full_codes_shape=%s summed_shape=%s "
                "gpu_ms prep=%.4f forward=%.4f post=%.4f total=%.4f profile_calls=%s profile_top_ms=%s",
                int(layer0_token_id),
                tuple(fc.shape),
                tuple(sm.shape),
                float(out[2]),
                float(out[3]),
                float(out[4]),
                float(out[5]),
                profile_dict.get("calls") if profile_dict else None,
                _profile_top_ms(profile_dict),
            )
        else:
            _vllm_rpc_log.info(
                "vllm_mtp_run_rpc done: layer0_token_id=%s full_codes_shape=%s summed_shape=%s "
                "(no gpu_ms) profile_calls=%s profile_top_ms=%s",
                int(layer0_token_id),
                tuple(fc.shape),
                tuple(sm.shape),
                profile_dict.get("calls") if profile_dict else None,
                _profile_top_ms(profile_dict),
            )
    return out
