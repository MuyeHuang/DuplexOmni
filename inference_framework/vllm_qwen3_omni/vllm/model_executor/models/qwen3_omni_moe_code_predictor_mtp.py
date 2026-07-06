"""Qwen3-Omni Code Predictor with MTP (Multi-Token Prediction) support.

This module implements the code predictor component for Qwen3-Omni talker models.

The code predictor generates residual RVQ (Residual Vector Quantization) codes
autoregressively, predicting layers 1 to N based on layer-0 codes from the talker.
"""

import os
from collections import namedtuple
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Cache, PretrainedConfig
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CUDAGraphMode, CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


_MTP_PROFILE_OWNER: Any | None = None


def _profile_current() -> Any | None:
    return _MTP_PROFILE_OWNER


def _profile_set_current(owner: Any | None) -> Any | None:
    global _MTP_PROFILE_OWNER
    prev = _MTP_PROFILE_OWNER
    _MTP_PROFILE_OWNER = owner
    return prev


def _profile_span_start() -> torch.cuda.Event | None:
    owner = _profile_current()
    if owner is None:
        return None
    return owner._mtp_profile_span_start()


def _profile_span_end(name: str, start: torch.cuda.Event | None) -> None:
    owner = _profile_current()
    if owner is not None:
        owner._mtp_profile_span_end(name, start)


def _profile_count(name: str, inc: int = 1) -> None:
    owner = _profile_current()
    if owner is not None:
        owner._mtp_profile_count(name, inc)


def _profile_count_map(name: str, key: Any, inc: int = 1) -> None:
    owner = _profile_current()
    if owner is not None:
        owner._mtp_profile_count_map(name, key, inc)


def _profile_backend_order(names: list[str]) -> None:
    owner = _profile_current()
    if owner is not None:
        owner._mtp_profile_backend_order(names)


def _profile_removed_backend(reason: str) -> None:
    owner = _profile_current()
    if owner is not None:
        owner._mtp_profile_removed_backend(reason)

# ============================================================================
# Code Predictor Attention Layer
# ============================================================================


class Qwen3OmniCodePredictorAttention(nn.Module):
    """Multi-head self-attention for code predictor with vLLM optimization."""

    def __init__(
        self,
        config,
        layer_idx: int,
        vllm_config: VllmConfig = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.layer_idx = int(layer_idx)

        self.num_heads = config.code_predictor_config.num_attention_heads
        self.num_key_value_heads = config.code_predictor_config.num_key_value_heads
        self.head_dim = getattr(
            config.code_predictor_config,
            "head_dim",
            config.code_predictor_config.hidden_size // config.code_predictor_config.num_attention_heads,
        )
        self.hidden_size = config.code_predictor_config.hidden_size

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=True,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=True,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.code_predictor_config.max_position_embeddings,
            rope_parameters=None,
            dual_chunk_attention_config=None,
        )

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        # Query/Key normalization
        self.q_norm = RMSNorm(self.head_dim, eps=config.code_predictor_config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.code_predictor_config.rms_norm_eps)
        self.is_causal = True
        self.config = config

        self.attention_backends = ["flash_attention_2", "xformers", "eager", "sdpa"]
        self.removed_attention_backends: list[str] = []
        cudagraph_mode = get_current_vllm_config().compilation_config.cudagraph_mode
        if "flash_attention_2" in ALL_ATTENTION_FUNCTIONS and cudagraph_mode.has_full_cudagraphs():
            logger.warning(
                f"CUDAGraphMode.{cudagraph_mode.name} is currently not supported "
                f"with flash attention for Qwen3-Omni talker MTP."
                f"removing flash attention from attention_backends"
            )
            self.attention_backends.remove("flash_attention_2")
            self.removed_attention_backends.append(
                f"flash_attention_2: cudagraph_mode={cudagraph_mode.name} has_full_cudagraphs"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        _profile_count("attention_calls")
        _profile_count_map("attention_seq_len_counts", seq_len)
        _profile_backend_order(self.attention_backends)
        for reason in self.removed_attention_backends:
            _profile_removed_backend(reason)

        t = _profile_span_start()
        qkv, _ = self.qkv_proj(hidden_states)
        _profile_span_end("attn_qkv_proj", t)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        t = _profile_span_start()
        # Reshape for attention
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply normalization
        q = self.q_norm(q).contiguous()
        k = self.k_norm(k).contiguous()
        q = q.reshape(-1, self.q_size)
        k = k.reshape(-1, self.kv_size)

        # Apply RoPE
        q, k = self.rotary_emb(position_ids, q, k)

        # Reshape for attention
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        _profile_span_end("attn_qk_norm_rope", t)

        v_heads = v.transpose(1, 2).contiguous()
        q_heads = q.transpose(1, 2).contiguous()
        k_heads = k.transpose(1, 2).contiguous()

        if past_key_values is not None:
            t = _profile_span_start()
            sin, cos = self.rotary_emb.get_cos_sin(seq_len)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k_heads, v_heads = past_key_values.update(k_heads, v_heads, self.layer_idx, cache_kwargs)
            _profile_span_end("attn_kv_cache_update", t)

        # Try attention backends in order of preference, with runtime error handling
        # This handles cases where the backend is registered but not actually available
        attn_output = None
        last_error = None

        for backend_name in self.attention_backends:
            if backend_name not in ALL_ATTENTION_FUNCTIONS:
                _profile_count_map("backend_missing_counts", backend_name)
                continue

            t_backend = None
            try:
                _profile_count_map("backend_attempt_counts", backend_name)
                attention_interface = ALL_ATTENTION_FUNCTIONS[backend_name]
                if position_ids is None:
                    pos_attn = None
                elif position_ids.numel() == bsz * seq_len:
                    pos_attn = position_ids.view(bsz, seq_len)
                else:
                    pos_attn = position_ids[:seq_len].unsqueeze(0)
                t_backend = _profile_span_start()
                attn_output, _ = attention_interface(
                    self,
                    q_heads,
                    k_heads,
                    v_heads,
                    None,
                    dropout=0.0 if not self.training else getattr(self, "attention_dropout", 0.0),
                    scaling=self.head_dim**-0.5,
                    sliding_window=None,
                    use_cache=use_cache,
                    position_ids=pos_attn,
                    output_hidden_states=True,
                    output_attentions=False,
                )
                _profile_span_end(f"attn_backend_{backend_name}", t_backend)
                _profile_count_map("backend_success_counts", backend_name)
                break
            except (ValueError, ImportError, RuntimeError, AttributeError) as e:
                _profile_span_end(f"attn_backend_{backend_name}", t_backend)
                _profile_count_map("backend_error_counts", backend_name)
                # Store error and try next backend
                last_error = e
                continue

        if attn_output is None:
            raise RuntimeError(
                f"All attention backends failed. Last error: {last_error}. "
                "Please install flash-attn, or ensure PyTorch's scaled_dot_product_attention is available."
            )
        attn_output = attn_output.reshape(*(hidden_states.shape[:-1]), -1).contiguous()

        t = _profile_span_start()
        attn_output, _ = self.o_proj(attn_output)
        _profile_span_end("attn_o_proj", t)
        return attn_output


# ============================================================================
# Code Predictor MLP Layer
# ============================================================================


class Qwen3OmniCodePredictorMLP(nn.Module):
    """Feed-forward network for code predictor with fused gate/up projection."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.code_predictor_config.hidden_size
        intermediate_size = config.code_predictor_config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,
        )

        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        t = _profile_span_start()
        gate_up, _ = self.gate_up_proj(hidden_states)
        _profile_span_end("mlp_gate_up_proj", t)
        gate, up = gate_up.chunk(2, dim=-1)
        t = _profile_span_start()
        down, _ = self.down_proj(F.silu(gate) * up)
        _profile_span_end("mlp_act_down_proj", t)
        return down


# ============================================================================
# MTP Layer (Multi-Token Prediction Layer)
# ============================================================================


class Qwen3OmniCodePredictorMTPLayer(nn.Module):
    """MTP layer for speculative decoding - predicts next residual code layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.self_attn = Qwen3OmniCodePredictorAttention(
            config,
            layer_idx,
            vllm_config=type(
                "VllmConfig",
                (),
                {"cache_config": cache_config, "quant_config": quant_config, "model_config": model_config},
            )(),
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3OmniCodePredictorMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(
            config.code_predictor_config.hidden_size, eps=config.code_predictor_config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.code_predictor_config.hidden_size, eps=config.code_predictor_config.rms_norm_eps
        )

    def mtp_block(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        _profile_count("mtp_block_calls")
        # Self-attention with residual
        residual = hidden_states
        t = _profile_span_start()
        hidden_states = self.input_layernorm(hidden_states)
        _profile_span_end("block_input_layernorm", t)
        t = _profile_span_start()
        hidden_states = self.self_attn(hidden_states, past_key_values, cache_position, use_cache, position_ids)
        _profile_span_end("block_self_attn_total", t)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        t = _profile_span_start()
        hidden_states = self.post_attention_layernorm(hidden_states)
        _profile_span_end("block_post_attention_layernorm", t)
        t = _profile_span_start()
        hidden_states = self.mlp(hidden_states)
        _profile_span_end("block_mlp_total", t)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3OmniCodePredictorBaseModel(nn.Module):
    """
    Base model for code predictor - matches HF Qwen3OmniMoeTalkerCodePredictorModel structure.

    This is a simple transformer that processes inputs_embeds and outputs hidden states.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.code_predictor_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.num_code_groups = config.num_code_groups

        # Codec embeddings (for layers 1-num_code_groups-1)
        self.codec_embedding = nn.ModuleList(
            [
                VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                )
                for _ in range(config.num_code_groups - 1)
            ]
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3OmniCodePredictorMTPLayer(
                    vllm_config.model_config.hf_config,
                    f"{prefix}.layers.{idx}",
                    model_config=vllm_config.model_config,
                    layer_idx=idx,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                )
                for idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Forward pass matching HF structure.

        Args:
            inputs_embeds: [batch, seq_len, hidden_size]
            position_ids: Optional position IDs tensor
            past_key_values: Optional cached key-value pairs
            use_cache: Whether to use cache
            cache_position: Optional cache position tensor
            **kwargs: Additional keyword arguments

        Returns:
            Named tuple with .last_hidden_state and .past_key_values attributes
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer.mtp_block(hidden_states, past_key_values, cache_position, use_cache, position_ids)

        t = _profile_span_start()
        hidden_states = self.norm(hidden_states)
        _profile_span_end("model_final_norm", t)

        Output = namedtuple("Output", ["last_hidden_state", "past_key_values"])
        return Output(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def get_input_embeddings(self):
        """Return codec embeddings for HF compatibility."""
        return self.codec_embedding


def code_predictor_sample(
    logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    t = _profile_span_start()
    logits = self.logits_processors(None, logits[:, -1])
    _profile_span_end("sample_logits_processors", t)
    t = _profile_span_start()
    probs = F.softmax(logits, dim=-1)
    _profile_span_end("sample_softmax", t)
    t = _profile_span_start()
    code = torch.multinomial(probs.squeeze(1), num_samples=1)  # [batch, 1]
    _profile_span_end("sample_multinomial", t)
    return code


def code_predictor_sample_fake(
    logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty((logits.shape[0], 1), dtype=torch.int64, device=logits.device)


direct_register_custom_op(
    op_name="qwen3_omni_code_predictor_sample",
    op_func=code_predictor_sample,
    fake_impl=code_predictor_sample_fake,
)


@support_torch_compile
class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
    """
    Code predictor wrapper matching HF structure.

    Structure:
    - self.model: Qwen3OmniCodePredictorBaseModel (transformer)
    - self.lm_head: ModuleList of output heads
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        talker_code_predictor_config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.config = talker_code_predictor_config
        self.vocab_size = self.config.code_predictor_config.vocab_size
        self.num_code_groups = self.config.code_predictor_config.num_code_groups

        # Base transformer model (matches HF structure)
        self.model = Qwen3OmniCodePredictorBaseModel(vllm_config=vllm_config, prefix=prefix)

        # Output heads for each residual layer (1-num_layers-1)
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(
                    self.config.code_predictor_config.hidden_size,
                    self.config.code_predictor_config.vocab_size,
                    bias=False,
                )
                for _ in range(self.num_code_groups - 1)
            ]
        )
        self.logits_processors = LogitsProcessorList(
            [
                TopKLogitsWarper(top_k=50),
                TopPLogitsWarper(top_p=0.8),
            ]
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

    def _mtp_profile_env_enabled(self) -> bool:
        raw = os.environ.get("TALKER_MTP_PROFILE", "1").strip().lower()
        return raw not in ("0", "false", "no", "off")

    def _mtp_profile_reset(self, enabled: bool | None = None) -> bool:
        active = self._mtp_profile_env_enabled() if enabled is None else bool(enabled)
        active = bool(active and torch.cuda.is_available())
        self._mtp_profile_active = active
        self._mtp_profile_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._mtp_profile_counts: dict[str, int] = {}
        self._mtp_profile_count_maps: dict[str, dict[str, int]] = {}
        self._mtp_profile_backend_order_value: list[str] = []
        self._mtp_profile_removed_backend_values: list[str] = []
        return active

    def _mtp_profile_span_start(self) -> torch.cuda.Event | None:
        if not bool(getattr(self, "_mtp_profile_active", False)):
            return None
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def _mtp_profile_span_end(self, name: str, start: torch.cuda.Event | None) -> None:
        if start is None or not bool(getattr(self, "_mtp_profile_active", False)):
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._mtp_profile_events.append((str(name), start, end))

    def _mtp_profile_count(self, name: str, inc: int = 1) -> None:
        if not bool(getattr(self, "_mtp_profile_active", False)):
            return
        counts = self._mtp_profile_counts
        counts[str(name)] = int(counts.get(str(name), 0)) + int(inc)

    def _mtp_profile_count_map(self, name: str, key: Any, inc: int = 1) -> None:
        if not bool(getattr(self, "_mtp_profile_active", False)):
            return
        maps = self._mtp_profile_count_maps
        bucket = maps.setdefault(str(name), {})
        skey = str(key)
        bucket[skey] = int(bucket.get(skey, 0)) + int(inc)

    def _mtp_profile_backend_order(self, names: list[str]) -> None:
        if not bool(getattr(self, "_mtp_profile_active", False)):
            return
        if not self._mtp_profile_backend_order_value:
            self._mtp_profile_backend_order_value = [str(x) for x in names]

    def _mtp_profile_removed_backend(self, reason: str) -> None:
        if not bool(getattr(self, "_mtp_profile_active", False)):
            return
        reason = str(reason)
        if reason not in self._mtp_profile_removed_backend_values:
            self._mtp_profile_removed_backend_values.append(reason)

    def _mtp_profile_finalize(self) -> dict[str, Any]:
        if not bool(getattr(self, "_mtp_profile_active", False)):
            return {}
        events = list(getattr(self, "_mtp_profile_events", []))
        if events:
            torch.cuda.synchronize()
        ms: dict[str, float] = {}
        for name, start, end in events:
            try:
                ms[name] = ms.get(name, 0.0) + float(start.elapsed_time(end))
            except RuntimeError:
                continue
        counts = {k: int(v) for k, v in getattr(self, "_mtp_profile_counts", {}).items()}
        profile: dict[str, Any] = {
            "calls": int(counts.get("code_predictor_calls", 0)),
            "num_events": int(len(events)),
            "ms": {k: round(float(v), 4) for k, v in sorted(ms.items())},
            "counts": counts,
        }
        for name, values in getattr(self, "_mtp_profile_count_maps", {}).items():
            profile[name] = {k: int(v) for k, v in sorted(values.items())}
        if self._mtp_profile_backend_order_value:
            profile["backend_order"] = list(self._mtp_profile_backend_order_value)
        if self._mtp_profile_removed_backend_values:
            profile["removed_attention_backends"] = list(self._mtp_profile_removed_backend_values)
        calls = max(1, int(profile["calls"]))
        residual_groups = max(1, int(counts.get("residual_groups", 0)))
        mtp_blocks = max(1, int(counts.get("mtp_block_calls", 0)))
        profile["avg_ms_per_call"] = {
            k: round(float(v) / calls, 4) for k, v in profile["ms"].items()
        }
        profile["avg_ms_per_group"] = {
            k: round(float(v) / residual_groups, 4) for k, v in profile["ms"].items()
        }
        profile["avg_ms_per_block"] = {
            k: round(float(v) / mtp_blocks, 4) for k, v in profile["ms"].items()
        }
        profile["top_ms"] = [
            {"name": k, "ms": round(float(v), 4)}
            for k, v in sorted(profile["ms"].items(), key=lambda item: float(item[1]), reverse=True)[:12]
        ]
        self._mtp_profile_active = False
        return profile

    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for code predictor.

        Args:
            layer0_code:
                Code index for code-group (layer) 0.
                Shape: [batch_size, 1], dtype typically int64.

            last_talker_hidden:

                Shape: [batch_size, hidden_size].

        Returns:
            pos_all_layers:
                Predicted codes for all code groups, including `layer0_code`.
                Shape: [batch_size, num_code_groups, 1].

            current_input:
                The final input embedding sequence after appending embeddings of all
                predicted codes (one token per predicted layer).
                Shape: [batch_size, num_code_groups + 2, hidden_size].
        """
        profile_prev = _profile_set_current(
            self if bool(getattr(self, "_mtp_profile_active", False)) else None
        )
        _profile_count("code_predictor_calls")
        _profile_count_map("num_code_groups_counts", self.num_code_groups)
        t = _profile_span_start()
        pos_codes = [layer0_code]
        try:
            current_input = torch.cat([last_talker_hidden, layer0_embed], dim=1)
        except Exception as e:
            print(f"Error in current_input: {e}")
            print(f"last_talker_hidden shape: {last_talker_hidden.shape}")
            print(f"prev_embed shape: {layer0_embed.shape}")
            _profile_set_current(profile_prev)
            raise e
        _profile_span_end("cp_initial_cat", t)
        batch_size = current_input.shape[0]
        device = current_input.device

        try:
            past_key_values: Cache | None = None
            past_len = 0
            new_embed: torch.Tensor | None = None

            logger.info_once(
                "MTP code_predictor: incremental KV cache; per-segment BatchDescriptor under "
                "override_forward_context for PIECEWISE CUDAGraph replay (seq_len 2 then 1)",
                scope="local",
            )

            for layer_idx in range(self.num_code_groups - 1):
                _profile_count("residual_groups")
                t = _profile_span_start()
                if layer_idx == 0:
                    segment = current_input
                    seq_len = segment.shape[1]
                    cache_position = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long)
                    position_ids = torch.arange(seq_len, device=device, dtype=torch.int64).repeat(batch_size)
                else:
                    assert new_embed is not None
                    segment = new_embed
                    seq_len = 1
                    cache_position = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long)
                    # CUDA graph capture-safe scalar materialization (avoid host->device tensor creation).
                    position_ids = torch.full((batch_size,), past_len, device=device, dtype=torch.int64)
                _profile_count_map("segment_seq_len_counts", seq_len)
                _profile_span_end("group_segment_prep", t)

                def _mtp_base_forward():
                    return self.model(
                        inputs_embeds=segment,
                        attention_mask=None,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_position,
                    )

                t = _profile_span_start()
                if (
                    is_forward_context_available()
                    and get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.NONE
                ):
                    parent = get_forward_context()
                    bd = BatchDescriptor(
                        num_tokens=int(seq_len),
                        num_reqs=None,
                        uniform=True,
                        has_lora=False,
                        num_active_loras=0,
                    )
                    with override_forward_context(replace(parent, batch_descriptor=bd)):
                        outputs = _mtp_base_forward()
                else:
                    outputs = _mtp_base_forward()
                _profile_span_end("group_model_total", t)
                hidden_state = outputs.last_hidden_state
                past_key_values = outputs.past_key_values
                past_len += seq_len

                t = _profile_span_start()
                logits = self.lm_head[layer_idx](hidden_state[:, -1:, :])
                _profile_span_end("group_lm_head", t)
                code = torch.ops.vllm.qwen3_omni_code_predictor_sample(logits, self.layer_name)
                pos_codes.append(code)
                t = _profile_span_start()
                new_embed = self.model.codec_embedding[layer_idx](code)
                current_input = torch.cat([current_input, new_embed], dim=1)
                _profile_span_end("group_embed_cat", t)

            t = _profile_span_start()
            pos_all_layers = torch.stack(pos_codes, dim=1)
            _profile_span_end("cp_stack_codes", t)
            return pos_all_layers, current_input
        finally:
            _profile_set_current(profile_prev)

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with mapping for fused QKV and gate_up projections.

        Maps original HF weights (q_proj, k_proj, v_proj, gate_proj, up_proj)
        to fused vLLM weights (qkv_proj, gate_up_proj).
        """
        # Mapping for fused projections
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Skip rotary embeddings
            if "rotary_emb.inv_freq" in name:
                continue

            # Handle stacked/fused parameters
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip if parameter doesn't exist (e.g., bias)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Non-stacked parameters - use default loading
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight)
                else:
                    param.data.copy_(loaded_weight)

            loaded_params.add(name)

        return loaded_params
