# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.spec_utils import build_module
from megatron.training import get_args

from swift.llm import ModelType, deep_getattr
from swift.utils import get_logger, is_last_rank
from ..constant import MegatronModelType
from ..gpt_model import GPTModel
from ..model_provider import _get_transformer_layer_spec
from ..register import MegatronModelMeta, register_megatron_model
from .qwen3_vl import Qwen3OmniBridge, Qwen3Omni_Vit, Qwen3VLGPTModel

logger = get_logger()


def _mark_replicated_params(module: nn.Module) -> None:
    for param in module.parameters(recurse=True):
        setattr(param, 'average_gradients_across_tp_domain', True)


def _move_module_to_runtime_device(module: nn.Module) -> nn.Module:
    args = get_args()
    if torch.cuda.is_available():
        module.to(device=torch.cuda.current_device(), dtype=args.torch_dtype)
    else:
        module.to(dtype=args.torch_dtype)
    return module


@contextmanager
def _patch_qwen3_omni_submodel_args(hf_text_config):
    args = get_args()
    attention_bias = bool(getattr(hf_text_config, 'attention_bias', False))
    num_attention_heads = int(hf_text_config.num_attention_heads)
    num_key_value_heads = int(hf_text_config.num_key_value_heads)
    patch_values = {
        'hidden_size': int(hf_text_config.hidden_size),
        'num_attention_heads': num_attention_heads,
        'num_query_groups': num_key_value_heads,
        'group_query_attention': num_key_value_heads != num_attention_heads,
        'num_experts': getattr(hf_text_config, 'num_experts', None),
        'qk_layernorm': True,
        'multi_latent_attention': False,
        'use_shared_expert_gate': True,
        'moe_shared_expert_intermediate_size': getattr(hf_text_config, 'shared_expert_intermediate_size', None),
        'moe_router_enable_expert_bias': False,
        'add_bias_linear': attention_bias,
        'add_qkv_bias': attention_bias,
    }
    sentinel = object()
    old_values = {k: getattr(args, k, sentinel) for k in patch_values}
    try:
        for key, value in patch_values.items():
            setattr(args, key, value)
        yield
    finally:
        for key, value in old_values.items():
            if value is sentinel:
                delattr(args, key)
            else:
                setattr(args, key, value)


def _build_qwen3_omni_submodel_config(base_config, hf_text_config):
    config = deepcopy(base_config)
    config.num_layers = int(hf_text_config.num_hidden_layers)
    config.hidden_size = int(hf_text_config.hidden_size)
    config.ffn_hidden_size = int(hf_text_config.intermediate_size)
    config.num_attention_heads = int(hf_text_config.num_attention_heads)
    config.num_query_groups = int(hf_text_config.num_key_value_heads)
    config.kv_channels = int(getattr(hf_text_config, 'head_dim', hf_text_config.hidden_size // hf_text_config.num_attention_heads))
    config.num_moe_experts = getattr(hf_text_config, 'num_experts', None)
    config.num_experts = config.num_moe_experts
    config.moe_router_topk = int(getattr(hf_text_config, 'num_experts_per_tok', config.moe_router_topk))
    config.num_experts_per_tok = config.moe_router_topk
    config.moe_ffn_hidden_size = int(getattr(hf_text_config, 'moe_intermediate_size', config.ffn_hidden_size))
    config.moe_shared_expert_intermediate_size = getattr(hf_text_config, 'shared_expert_intermediate_size', None)
    config.shared_expert_intermediate_size = config.moe_shared_expert_intermediate_size
    config.decoder_sparse_step = int(getattr(hf_text_config, 'decoder_sparse_step', 1))
    config.mlp_only_layers = list(getattr(hf_text_config, 'mlp_only_layers', []))
    config.moe_router_enable_expert_bias = False
    config.layernorm_epsilon = float(getattr(hf_text_config, 'rms_norm_eps', config.layernorm_epsilon))
    config.attention_dropout = float(getattr(hf_text_config, 'attention_dropout', config.attention_dropout))
    config.hidden_dropout = 0.0
    config.gated_linear_unit = True
    return config


def _enable_shared_expert_gate_in_layer_spec(transformer_layer_spec) -> None:
    layer_specs = getattr(transformer_layer_spec, 'layer_specs', None)
    if layer_specs is None:
        layer_specs = [transformer_layer_spec]
    for layer_spec in layer_specs:
        submodules = getattr(layer_spec, 'submodules', None)
        mlp = getattr(submodules, 'mlp', None)
        mlp_submodules = getattr(mlp, 'submodules', None)
        shared_experts = getattr(mlp_submodules, 'shared_experts', None)
        if shared_experts is None:
            continue
        params = dict(getattr(shared_experts, 'params', {}) or {})
        params['gate'] = True
        shared_experts.params = params


class _NativeTalkerOutput:

    def __init__(self, *, logits: torch.Tensor, hidden_states, generation_steps: Optional[int] = None):
        self.logits = logits
        self.hidden_states = hidden_states
        self.generation_steps = generation_steps


class _Qwen3OmniResizeMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.linear_fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
        self.act_fn = nn.SiLU()
        _move_module_to_runtime_device(self)
        _mark_replicated_params(self)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class _Qwen3OmniNativeTalkerBackbone(nn.Module):

    def __init__(self, gpt_model: GPTModel, codec_vocab_size: int, hidden_size: int):
        super().__init__()
        self.gpt = gpt_model
        self.codec_embedding = nn.Embedding(codec_vocab_size, hidden_size)
        _move_module_to_runtime_device(self.codec_embedding)
        _mark_replicated_params(self.codec_embedding)
        layer_norm_spec = getattr(getattr(self.gpt, 'decoder', None), 'submodules', None)
        layer_norm_spec = getattr(layer_norm_spec, 'layer_norm', None)
        self._norm = None if layer_norm_spec is None else build_module(
            layer_norm_spec,
            config=self.gpt.config,
            hidden_size=self.gpt.config.hidden_size,
            eps=self.gpt.config.layernorm_epsilon,
        )

    @property
    def config(self):
        return self.gpt.config

    @property
    def layers(self):
        return self.gpt.decoder.layers

    @property
    def norm(self):
        return self._norm

    def forward(self, inputs_embeds: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device, dtype=torch.long)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, batch_size, -1)
        elif position_ids.ndim != 3:
            raise RuntimeError(
                f'native talker expects position_ids rank-2 [batch, seq] or rank-3 [3, batch, seq], '
                f'got shape={tuple(position_ids.shape)}')
        decoder_input = inputs_embeds.transpose(0, 1).contiguous()
        args = get_args()
        if args.sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            if seq_len % tp_size != 0:
                raise RuntimeError(
                    f'native talker sequence length must be divisible by tensor parallel size when sequence_parallel '
                    f'is enabled: seq_len={seq_len}, tp_size={tp_size}')
            shard_len = seq_len // tp_size
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            shard_start = tp_rank * shard_len
            shard_end = shard_start + shard_len
            decoder_input = decoder_input[shard_start:shard_end].contiguous()
            # Keep full position ids for mRoPE. Under sequence parallel the hidden states are
            # reassembled inside attention before RoPE is applied, so slicing position_ids here
            # makes the rotary frequencies shorter than the query/key sequence.
        local_seq_len = decoder_input.shape[0]
        dummy_input_ids = torch.zeros((batch_size, local_seq_len), dtype=torch.long, device=inputs_embeds.device)
        if not self.gpt.pre_process:
            self.gpt.decoder.set_input_tensor(decoder_input)
        try:
            hidden_states = self.gpt(
                input_ids=dummy_input_ids,
                position_ids=position_ids,
                attention_mask=None,
                decoder_input=decoder_input,
                labels=None)
        finally:
            if not self.gpt.pre_process:
                self.gpt.decoder.input_tensor = None
        if self._norm is not None:
            hidden_states = self._norm(hidden_states)
        if get_args().sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        if hidden_states.ndim != 3:
            raise RuntimeError(f'native talker backbone expects rank-3 hidden states, got {tuple(hidden_states.shape)}')
        if hidden_states.shape[0] == seq_len and hidden_states.shape[1] == batch_size:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        elif hidden_states.shape[0] != batch_size:
            raise RuntimeError(
                f'native talker backbone returned unexpected hidden shape {tuple(hidden_states.shape)} for '
                f'batch={batch_size}, seq={seq_len}')
        return hidden_states

    def get_input_embeddings(self):
        return self.codec_embedding


class _Qwen3OmniExternalCodePredictor(nn.Module):

    def __init__(self, predictor_config):
        super().__init__()
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorModel

        self.config = predictor_config
        self.model = Qwen3OmniMoeTalkerCodePredictorModel(predictor_config)
        self.lm_head = nn.ModuleList(
            [nn.Linear(predictor_config.hidden_size, predictor_config.vocab_size, bias=False)
             for _ in range(predictor_config.num_code_groups - 1)])
        _move_module_to_runtime_device(self)
        _mark_replicated_params(self)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                cache_position=None,
                generation_steps=None,
                **kwargs):
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = int(inputs_embeds.shape[1] - 2)
        else:
            if generation_steps is None:
                raise RuntimeError('native external code_predictor requires `generation_steps`.')
            generation_steps = int(generation_steps)
            if generation_steps <= 0:
                raise RuntimeError(
                    'native external code_predictor generation stage expects `generation_steps >= 1` when '
                    '`inputs_embeds` is not prefilled.')
            inputs_embeds = self.get_input_embeddings()[generation_steps - 1](input_ids)
        if generation_steps < 0 or generation_steps >= len(self.lm_head):
            raise RuntimeError(
                f'native external code_predictor generation_steps out of range: step={generation_steps}, '
                f'num_heads={len(self.lm_head)}')
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head[generation_steps](hidden_states)
        return _NativeTalkerOutput(
            logits=logits, hidden_states=(hidden_states, ), generation_steps=generation_steps + 1)


class _Qwen3OmniNativeTalker(nn.Module):

    def __init__(self, talker_config, model_backbone: _Qwen3OmniNativeTalkerBackbone,
                 code_predictor: _Qwen3OmniExternalCodePredictor):
        super().__init__()
        text_hidden = int(talker_config.text_config.hidden_size)
        thinker_hidden = int(talker_config.thinker_hidden_size)
        self.config = talker_config
        self.model = model_backbone
        self.text_projection = _Qwen3OmniResizeMLP(thinker_hidden, thinker_hidden, text_hidden)
        self.hidden_projection = _Qwen3OmniResizeMLP(thinker_hidden, thinker_hidden, text_hidden)
        self.codec_head = nn.Linear(text_hidden, int(talker_config.text_config.vocab_size), bias=False)
        _move_module_to_runtime_device(self.codec_head)
        _mark_replicated_params(self.codec_head)
        self.code_predictor = code_predictor

    @property
    def dtype(self):
        return self.model.codec_embedding.weight.dtype

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_hidden_states=True,
                return_dict=True,
                **kwargs):
        if inputs_embeds is None:
            raise RuntimeError('native talker requires `inputs_embeds`.')
        hidden_states = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        logits = self.codec_head(hidden_states)
        return _NativeTalkerOutput(logits=logits, hidden_states=(hidden_states, ))


class Qwen3OmniE2E_Vit(Qwen3Omni_Vit):
    module_mapping = {'thinker': 'thinker', 'code2wav': 'code2wav'}
    # Do not freeze talker for E2E training.
    _generator = ['code2wav']


class Qwen3OmniE2EModel(Qwen3VLGPTModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.talker = None
        if not self.post_process:
            return
        if self.visual is None:
            raise RuntimeError('qwen3_omni_e2e requires visual config to build native talker.')
        talker_config = getattr(self.visual.model_config, 'talker_config', None)
        if talker_config is None:
            raise RuntimeError('qwen3_omni_e2e visual model config is missing `talker_config`.')
        rope_scaling = getattr(talker_config.text_config, 'rope_scaling', None)
        rope_scaling_factor = 1.0
        if isinstance(rope_scaling, dict):
            rope_scaling_factor = float(rope_scaling.get('factor', 1.0))
        with _patch_qwen3_omni_submodel_args(talker_config.text_config):
            talker_model_config = _build_qwen3_omni_submodel_config(self.config, talker_config.text_config)
            talker_layer_spec = _get_transformer_layer_spec(
                get_args().transformer_impl == 'transformer_engine', talker_model_config)
            if (getattr(get_args(), 'use_shared_expert_gate', False) and talker_model_config.num_experts
                    and talker_model_config.moe_shared_expert_intermediate_size):
                _enable_shared_expert_gate_in_layer_spec(talker_layer_spec)
            talker_backbone = _Qwen3OmniNativeTalkerBackbone(
                GPTModel(
                    config=talker_model_config,
                    transformer_layer_spec=talker_layer_spec,
                    vocab_size=int(talker_config.text_config.vocab_size),
                    max_sequence_length=int(talker_config.text_config.max_position_embeddings),
                    pre_process=False,
                    post_process=False,
                    fp16_lm_cross_entropy=False,
                    parallel_output=True,
                    share_embeddings_and_output_weights=False,
                    position_embedding_type='mrope',
                    rotary_percent=1.0,
                    rotary_base=int(getattr(talker_config.text_config, 'rope_theta', 10000.0)),
                    hf_rope_scaling=rope_scaling,
                    rope_scaling=rope_scaling is not None,
                    rope_scaling_factor=rope_scaling_factor,
                    seq_len_interpolation_factor=None,
                ),
                codec_vocab_size=int(talker_config.text_config.vocab_size),
                hidden_size=int(talker_config.text_config.hidden_size),
            )
        code_predictor = _Qwen3OmniExternalCodePredictor(talker_config.code_predictor_config)
        self.talker = _Qwen3OmniNativeTalker(talker_config, talker_backbone, code_predictor)

    @staticmethod
    def _parse_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in {'1', 'true', 'yes', 'y', 'on'}

    @staticmethod
    def _dist_rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _debug_step_enabled(self) -> bool:
        args = get_args()
        return getattr(args, 'qwen3_omni_e2e_debug_trace', True) and getattr(
            args, 'curr_iteration', 0) < getattr(args, 'qwen3_omni_e2e_debug_trace_steps', 2)

    def _debug_rank_enabled(self) -> bool:
        args = get_args()
        return getattr(args, 'qwen3_omni_e2e_debug_all_ranks', False) or self._dist_rank() == 0

    @staticmethod
    def _contains_anomaly(obj) -> bool:
        if torch.is_tensor(obj):
            if not torch.is_floating_point(obj):
                return False
            return torch.isnan(obj).any().item() or torch.isinf(obj).any().item()
        if isinstance(obj, (tuple, list)):
            return any(Qwen3OmniE2EModel._contains_anomaly(x) for x in obj)
        return False

    def _debug_log(self, msg: str, *, force: bool = False) -> None:
        if not force and not (self._debug_step_enabled() and self._debug_rank_enabled()):
            return
        rank = self._dist_rank()
        step = getattr(get_args(), 'curr_iteration', -1)
        logger.warning(f'[qwen3_omni_e2e][rank={rank}][iter={step}] {msg}')

    def _tensor_stats(self, name: str, tensor: Optional[torch.Tensor], *, force: bool = False) -> None:
        if tensor is None:
            self._debug_log(f'{name}: None', force=force)
            return
        if not torch.is_tensor(tensor):
            self._debug_log(f'{name}: non-tensor type={type(tensor).__name__}', force=True)
            return
        anomaly = self._contains_anomaly(tensor)
        if not (force or anomaly or (self._debug_step_enabled() and self._debug_rank_enabled())):
            return
        shape = tuple(tensor.shape)
        msg = f'{name}: shape={shape}, dtype={tensor.dtype}, device={tensor.device}'
        if tensor.numel() == 0:
            self._debug_log(msg + ', empty', force=True)
            return
        if torch.is_floating_point(tensor):
            det = tensor.detach().float()
            nan_count = int(torch.isnan(det).sum().item())
            inf_count = int(torch.isinf(det).sum().item())
            finite = det[torch.isfinite(det)]
            if finite.numel() > 0:
                msg += (
                    f', nan={nan_count}, inf={inf_count}, min={finite.min().item():.6e}, max={finite.max().item():.6e},'
                    f' mean={finite.mean().item():.6e}, std={finite.std(unbiased=False).item():.6e},'
                    f' absmax={finite.abs().max().item():.6e}')
            else:
                msg += f', nan={nan_count}, inf={inf_count}, finite=0'
        else:
            det = tensor.detach()
            msg += f', min={det.min().item()}, max={det.max().item()}'
        self._debug_log(msg, force=force or anomaly)

    def _id_stats(self, name: str, tensor: Optional[torch.Tensor], special_ids: Dict[str, int], *, force: bool = False) -> None:
        if tensor is None:
            self._debug_log(f'{name}: None', force=force)
            return
        if tensor.ndim > 2 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        self._tensor_stats(name, tensor, force=force)
        if tensor.ndim == 1:
            inspect = tensor.unsqueeze(0)
        else:
            inspect = tensor
        summary = []
        for k, v in special_ids.items():
            summary.append(f'{k}={int((inspect == v).sum().item())}')
        valid = inspect[inspect != -100]
        if valid.numel() > 0:
            summary.append(f'valid={valid.numel()}')
            summary.append(f'valid_min={valid.min().item()}')
            summary.append(f'valid_max={valid.max().item()}')
        self._debug_log(f'{name} special-counts: ' + ', '.join(summary), force=force)

    @staticmethod
    def _extract_tensor(obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (tuple, list)):
            for item in obj:
                if torch.is_tensor(item):
                    return item
        return None

    def _register_module_trace(self, stack: ExitStack, module, name: str) -> None:
        def _hook(_module, inputs, output):
            input_tensor = None
            if isinstance(inputs, tuple) and inputs:
                input_tensor = self._extract_tensor(inputs[0])
            else:
                input_tensor = self._extract_tensor(inputs)
            if input_tensor is not None:
                self._tensor_stats(f'{name}.input', input_tensor)
            if isinstance(output, tuple):
                for idx, item in enumerate(output[:2]):
                    if torch.is_tensor(item):
                        self._tensor_stats(f'{name}.output{idx}', item)
            elif torch.is_tensor(output):
                self._tensor_stats(f'{name}.output', output)

        handle = module.register_forward_hook(_hook)
        stack.callback(handle.remove)

    def _install_debug_hooks(self, stack: ExitStack) -> None:
        if not self._debug_step_enabled():
            return
        args = get_args()
        thinker_last_layers = getattr(args, 'qwen3_omni_e2e_debug_thinker_last_layers', 6)
        thinker_layers = list(getattr(self.language_model.decoder, 'layers', []))
        if thinker_layers:
            start = max(0, len(thinker_layers) - thinker_last_layers)
            for idx in range(start, len(thinker_layers)):
                self._register_module_trace(stack, thinker_layers[idx], f'thinker.layer{idx}')
        talker = self.talker
        self._debug_log(
            'talker config: '
            f'num_layers={len(talker.model.layers)}, num_experts={getattr(talker.model.config, "num_experts", "na")}, '
            f'experts_per_tok={getattr(talker.model.config, "num_experts_per_tok", "na")}, '
            f'decoder_sparse_step={getattr(talker.model.config, "decoder_sparse_step", "na")}',
            force=True)
        for idx, layer in enumerate(talker.model.layers):
            self._register_module_trace(stack, layer, f'talker.layer{idx}')
            self._register_module_trace(stack, layer.self_attention, f'talker.layer{idx}.self_attention')
            self._register_module_trace(stack, layer.mlp, f'talker.layer{idx}.mlp')

    def _maybe_force_eager_talker_attention(self) -> None:
        args = get_args()
        if not getattr(args, 'qwen3_omni_e2e_force_eager_talker_attn', False):
            return
        talker = self.talker
        self._debug_log(
            'qwen3_omni_e2e_force_eager_talker_attn is ignored for native talker backbone; '
            'Megatron attention implementation is selected by runtime config.',
            force=True)

    def _validate_replicated_aux_grad_sync(self) -> None:
        if getattr(self, '_validated_replicated_aux_grad_sync', False):
            return
        if self.talker is None:
            raise RuntimeError('qwen3_omni_e2e requires native `self.talker` for replicated aux grad validation.')
        talker = self.talker
        module_specs = (
            ('talker.model.codec_embedding', getattr(getattr(talker, 'model', None), 'codec_embedding', None)),
            ('talker.text_projection', getattr(talker, 'text_projection', None)),
            ('talker.hidden_projection', getattr(talker, 'hidden_projection', None)),
            ('talker.codec_head', getattr(talker, 'codec_head', None)),
            ('talker.code_predictor', getattr(talker, 'code_predictor', None)),
        )
        for module_name, module in module_specs:
            if module is None:
                raise RuntimeError(f'qwen3_omni_e2e missing required module `{module_name}` for grad sync validation.')
            missing = []
            for param_name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if not getattr(param, 'average_gradients_across_tp_domain', False):
                    missing.append(param_name)
                    if len(missing) >= 24:
                        break
            if missing:
                raise RuntimeError(
                    f'qwen3_omni_e2e requires replicated TP grad sync for `{module_name}`, '
                    f'but {len(missing)} parameters are missing `average_gradients_across_tp_domain`: {missing}')
        self._validated_replicated_aux_grad_sync = True

    @staticmethod
    def _to_batch_seq_hidden(tensor: torch.Tensor, *, batch_size: int, hidden_size: int, name: str) -> torch.Tensor:
        if tensor.ndim != 3:
            raise RuntimeError(f'{name} must be rank-3, got shape={tuple(tensor.shape)}')
        if tensor.shape[0] == batch_size and tensor.shape[-1] == hidden_size:
            return tensor.contiguous()
        if tensor.shape[1] == batch_size and tensor.shape[-1] == hidden_size:
            return tensor.transpose(0, 1).contiguous()
        if tensor.shape[0] == batch_size and tensor.shape[1] == hidden_size:
            return tensor.transpose(1, 2).contiguous()
        if tensor.shape[1] == batch_size and tensor.shape[0] == hidden_size:
            return tensor.permute(1, 2, 0).contiguous()
        raise RuntimeError(
            f'Cannot normalize `{name}` to [batch, seq, hidden]: '
            f'shape={tuple(tensor.shape)}, batch_size={batch_size}, hidden_size={hidden_size}')

    @staticmethod
    def _slice_sp_tensor(tensor: Optional[torch.Tensor], target_seq_len: int) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.ndim < 2:
            return tensor
        seq_len = tensor.shape[1]
        if seq_len == target_seq_len:
            return tensor
        args = get_args()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        if args.sequence_parallel and tp_size > 1 and seq_len % tp_size == 0:
            shard_len = seq_len // tp_size
            start = tp_rank * shard_len
            end = start + shard_len
            sliced = tensor[:, start:end]
            if sliced.shape[1] == target_seq_len:
                return sliced
        # Fallback: keep consistent length to avoid shape mismatch.
        return tensor[:, :target_seq_len]

    real_codec_tokens_per_turn = 6

    @staticmethod
    def _masked_ce_loss(logits: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if labels is None:
            return logits.new_zeros(())
        flat_labels = labels.reshape(-1)
        valid_mask = flat_labels != -100
        if not valid_mask.any():
            return logits.new_zeros(())
        flat_logits = logits.reshape(-1, logits.shape[-1]).float()
        return F.cross_entropy(flat_logits[valid_mask], flat_labels[valid_mask])

    @staticmethod
    def _causal_ce_loss(logits: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if labels is None:
            return logits.new_zeros(())
        if logits.ndim != 3 or labels.ndim != 2:
            raise RuntimeError(
                f'causal_ce expects logits [batch, seq, vocab] and labels [batch, seq], '
                f'got logits={tuple(logits.shape)}, labels={tuple(labels.shape)}')
        if logits.shape[1] != labels.shape[1]:
            raise RuntimeError(
                f'causal_ce requires aligned seq length before shift, got logits={tuple(logits.shape)}, '
                f'labels={tuple(labels.shape)}')
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        flat_labels = shift_labels.reshape(-1)
        valid_mask = flat_labels != -100
        if not valid_mask.any():
            return logits.new_zeros(())
        flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1]).float()
        return F.cross_entropy(flat_logits[valid_mask], flat_labels[valid_mask])

    @staticmethod
    def _mean_thinker_loss(output_tensor: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if not isinstance(output_tensor, torch.Tensor):
            return torch.tensor(0.0, device='cuda')
        if labels is None:
            return output_tensor.float().mean()
        mask = labels != -100
        if output_tensor.shape == labels.shape:
            if mask.any():
                return (output_tensor.float() * mask).sum() / mask.sum()
            return output_tensor.new_zeros((), dtype=torch.float32)
        return output_tensor.float().mean()

    def _gather_sp_hidden(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None or tensor.ndim != 3:
            return tensor
        args = get_args()
        if args.sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1:
            return gather_from_sequence_parallel_region(tensor)
        return tensor

    def _pp_broadcast_tensor(self, tensor: Optional[torch.Tensor], *, name: str) -> Optional[torch.Tensor]:
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        if pp_size <= 1:
            return tensor
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        if tensor is not None:
            device = tensor.device
        else:
            device = torch.device('cuda', torch.cuda.current_device())
        owner_count = torch.tensor([1 if tensor is not None else 0], dtype=torch.int64, device=device)
        dist.all_reduce(owner_count, group=pp_group)
        if owner_count.item() != 1:
            raise RuntimeError(f'Expected exactly one PP owner for `{name}`, got {owner_count.item()}.')
        owner_rank = torch.tensor(
            [parallel_state.get_pipeline_model_parallel_rank() if tensor is not None else 0],
            dtype=torch.int64,
            device=device)
        dist.all_reduce(owner_rank, group=pp_group)
        src_rank = dist.get_global_rank(pp_group, int(owner_rank.item()))
        meta_data = torch.zeros(10, dtype=torch.int64, device=device)
        dtype_mapping = {
            torch.float64: 0,
            torch.float32: 1,
            torch.float16: 2,
            torch.bfloat16: 3,
            torch.uint8: 4,
            torch.int64: 5,
            torch.int32: 6,
            torch.bool: 7,
        }
        dtype_mapping_r = {v: k for k, v in dtype_mapping.items()}
        if tensor is None:
            dist.broadcast(meta_data, src=src_rank, group=pp_group)
            ndim = int(meta_data[0].item())
            if ndim <= 0:
                raise RuntimeError(f'Invalid PP metadata for `{name}`: {meta_data.tolist()}')
            shape = meta_data[1:1 + ndim].tolist()
            dtype = dtype_mapping_r[int(meta_data[-1].item())]
            tensor = torch.empty(shape, device=device, dtype=dtype)
            dist.broadcast(tensor, src=src_rank, group=pp_group)
        else:
            meta_data[0] = tensor.ndim
            meta_data[1:1 + tensor.ndim] = torch.tensor(tensor.shape, dtype=torch.int64, device=device)
            meta_data[-1] = dtype_mapping[tensor.dtype]
            dist.broadcast(meta_data, src=src_rank, group=pp_group)
            dist.broadcast(tensor, src=src_rank, group=pp_group)
        self._tensor_stats(f'pp.{name}', tensor, force=True)
        return tensor

    def _build_talker_attention_mask(self, seq_len: int, *, device) -> torch.Tensor:
        return torch.ones((1, seq_len), dtype=torch.long, device=device)

    def _infer_sample_media_counts(self, input_ids: torch.Tensor, *, use_audio_in_video: bool) -> Dict[str, int]:
        talker_cfg = self.talker.config
        ids = input_ids[0] if input_ids.ndim == 2 else input_ids.reshape(-1)
        audio_count = int((ids == talker_cfg.audio_start_token_id).sum().item())
        image_count = 0
        video_count = 0
        vision_starts = torch.nonzero(ids == talker_cfg.vision_start_token_id, as_tuple=False).reshape(-1)
        for start_idx in vision_starts.tolist():
            if start_idx + 1 >= ids.numel():
                continue
            next_token = int(ids[start_idx + 1].item())
            if next_token == talker_cfg.image_token_id:
                image_count += 1
            elif use_audio_in_video and next_token == talker_cfg.audio_start_token_id:
                video_count += 1
            elif next_token == talker_cfg.video_token_id:
                video_count += 1
        return {'audio': audio_count, 'image': image_count, 'video': video_count}

    @staticmethod
    def _slice_sample_meta(meta, start: int, count: int):
        if meta is None or count <= 0:
            return None
        if torch.is_tensor(meta):
            return meta[start:start + count]
        return meta[start:start + count]

    @staticmethod
    def _materialize_meta(meta, *, device: torch.device):
        if meta is None:
            return None
        if torch.is_tensor(meta):
            return meta.to(device=device)
        values = [v if torch.is_tensor(v) else torch.as_tensor(v) for v in meta]
        return torch.stack(values, dim=0).to(device=device)

    @staticmethod
    def _meta_length(meta) -> int:
        if meta is None:
            return 0
        if torch.is_tensor(meta):
            return int(meta.shape[0])
        return len(meta)

    @staticmethod
    def _normalize_rvq_codes(rvq_codes: torch.Tensor) -> torch.Tensor:
        if rvq_codes.ndim == 4 and rvq_codes.shape[1] == 1:
            rvq_codes = rvq_codes[:, 0]
        if rvq_codes.ndim != 3:
            raise RuntimeError(f'rvq_codes must be rank-3 [batch, num_groups, time], got shape={tuple(rvq_codes.shape)}')
        if rvq_codes.shape[1] == 16:
            return rvq_codes.contiguous()
        if rvq_codes.shape[2] == 16:
            return rvq_codes.transpose(1, 2).contiguous()
        raise RuntimeError(f'Cannot normalize rvq_codes to [batch, 16, time], got shape={tuple(rvq_codes.shape)}')

    @staticmethod
    def _normalize_rvq_length(rvq_length: torch.Tensor) -> torch.Tensor:
        if rvq_length.ndim == 2 and rvq_length.shape[1] == 1:
            rvq_length = rvq_length[:, 0]
        if rvq_length.ndim != 1:
            raise RuntimeError(f'rvq_length must be rank-1 [batch], got shape={tuple(rvq_length.shape)}')
        return rvq_length.contiguous()

    @staticmethod
    def _normalize_codec_turn_boundaries(codec_turn_boundaries: torch.Tensor) -> torch.Tensor:
        if codec_turn_boundaries.ndim == 3 and codec_turn_boundaries.shape[1] == 1:
            codec_turn_boundaries = codec_turn_boundaries[:, 0]
        if codec_turn_boundaries.ndim != 2:
            raise RuntimeError(
                'codec_turn_boundaries must be rank-2 [batch, max_turns + 1], '
                f'got shape={tuple(codec_turn_boundaries.shape)}')
        return codec_turn_boundaries.long().contiguous()

    @staticmethod
    def _normalize_codec_turn_count(codec_turn_count: torch.Tensor) -> torch.Tensor:
        if codec_turn_count.ndim == 2 and codec_turn_count.shape[1] == 1:
            codec_turn_count = codec_turn_count[:, 0]
        if codec_turn_count.ndim != 1:
            raise RuntimeError(
                f'codec_turn_count must be rank-1 [batch], got shape={tuple(codec_turn_count.shape)}')
        return codec_turn_count.long().contiguous()

    @staticmethod
    def _normalize_codec_turn_indices(codec_assistant_turn_indices: torch.Tensor) -> torch.Tensor:
        if codec_assistant_turn_indices.ndim == 3 and codec_assistant_turn_indices.shape[1] == 1:
            codec_assistant_turn_indices = codec_assistant_turn_indices[:, 0]
        if codec_assistant_turn_indices.ndim != 2:
            raise RuntimeError(
                'codec_assistant_turn_indices must be rank-2 [batch, max_turns], '
                f'got shape={tuple(codec_assistant_turn_indices.shape)}')
        return codec_assistant_turn_indices.long().contiguous()

    @staticmethod
    def _normalize_assistant_total_count(assistant_total_count: torch.Tensor) -> torch.Tensor:
        if assistant_total_count.ndim == 2 and assistant_total_count.shape[1] == 1:
            assistant_total_count = assistant_total_count[:, 0]
        if assistant_total_count.ndim != 1:
            raise RuntimeError(
                f'assistant_total_count must be rank-1 [batch], got shape={tuple(assistant_total_count.shape)}')
        return assistant_total_count.long().contiguous()

    def _build_codec_sum_embeds(self, rvq_codes: torch.Tensor) -> torch.Tensor:
        if rvq_codes.ndim != 3:
            raise RuntimeError(f'rvq_codes must be rank-3 [batch, num_groups, time], got shape={tuple(rvq_codes.shape)}')
        talker = self.talker
        predictor_embeds = talker.code_predictor.get_input_embeddings()
        expected_num_groups = len(predictor_embeds) + 1
        if rvq_codes.shape[1] != expected_num_groups:
            raise RuntimeError(
                f'RVQ num_groups mismatch for codec embeddings: expected={expected_num_groups}, '
                f'got={rvq_codes.shape[1]}')
        layer0_codes = rvq_codes[:, 0, :]
        all_layer_embeds_sum = talker.get_input_embeddings()(layer0_codes)
        for group_idx, predictor_embed in enumerate(predictor_embeds, start=1):
            all_layer_embeds_sum = all_layer_embeds_sum + predictor_embed(rvq_codes[:, group_idx, :])
        return all_layer_embeds_sum.to(talker.dtype)

    def _build_target_codec_inputs(self, rvq_codes: torch.Tensor) -> Dict[str, torch.Tensor]:
        talker = self.talker
        if rvq_codes.ndim != 3:
            raise RuntimeError(f'Target rvq_codes must be rank-3 [batch, num_groups, time], got {tuple(rvq_codes.shape)}')
        num_codec_tokens = int(rvq_codes.shape[-1])
        if num_codec_tokens != self.real_codec_tokens_per_turn:
            raise RuntimeError(
                f'Each talker turn must contain exactly {self.real_codec_tokens_per_turn} real codec tokens, '
                f'got {num_codec_tokens}')
        batch_size = rvq_codes.shape[0]
        device = rvq_codes.device
        codec_sum_embeds = self._build_codec_sum_embeds(rvq_codes)
        bos_ids = torch.full((batch_size, 1), talker.config.codec_bos_id, dtype=torch.long, device=device)
        eos_ids = torch.full((batch_size, 1), talker.config.codec_eos_token_id, dtype=torch.long, device=device)
        bos_embed = talker.get_input_embeddings()(bos_ids).to(talker.dtype)
        eos_embed = talker.get_input_embeddings()(eos_ids).to(talker.dtype)
        layer0_codes = rvq_codes[:, 0, :]
        inputs_embeds = torch.cat((bos_embed, codec_sum_embeds, eos_embed), dim=1)
        bos_labels = torch.full((batch_size, 1), -100, dtype=torch.long, device=device)
        # BOS is a manually prepended decoder input; it must not become a supervised target.
        labels = torch.cat((bos_labels, layer0_codes, eos_ids), dim=1)
        expected_ar_steps = self.real_codec_tokens_per_turn + 2
        if inputs_embeds.shape[1] != expected_ar_steps or labels.shape[1] != expected_ar_steps:
            raise RuntimeError(
                f'Unexpected talker AR span length: inputs={inputs_embeds.shape[1]}, labels={labels.shape[1]}, '
                f'expected={expected_ar_steps}')
        return {
            'inputs_embeds': inputs_embeds,
            'labels': labels,
            'num_codec_tokens': num_codec_tokens,
        }

    def _collect_assistant_segments(self, input_ids: torch.Tensor) -> List[Dict[str, int]]:
        cfg = self.visual.model_config
        im_start_positions = torch.nonzero(input_ids[0] == cfg.im_start_token_id, as_tuple=False).reshape(-1)
        im_start_indexes = torch.cat((im_start_positions, input_ids.new_tensor([input_ids.shape[1]])), dim=0)
        assistant_segments = []
        assistant_idx = 0
        for i in range(len(im_start_indexes) - 1):
            start = int(im_start_indexes[i].item())
            end = int(im_start_indexes[i + 1].item())
            if start + 1 >= input_ids.shape[1]:
                continue
            role_token = int(input_ids[0, start + 1].item())
            if role_token == cfg.assistant_token_id:
                assistant_segments.append({'assistant_idx': assistant_idx, 'start': start, 'end': end})
                assistant_idx += 1
        return assistant_segments

    def _build_talker_assistant_scaffold(self, thinker_embed: torch.Tensor, thinker_top_hidden: torch.Tensor, start: int,
                                         end: int) -> torch.Tensor:
        talker = self.talker
        assistant_token_count = end - start
        if assistant_token_count < 2:
            raise RuntimeError(
                f'Assistant segment must contain at least two tokens for assist[:-1] conditioning: '
                f'start={start}, end={end}, len={assistant_token_count}')
        visible_end = end - 1
        assistant_embed_hidden = talker.text_projection(thinker_embed[:, start:visible_end]).to(talker.dtype)
        assistant_top_hidden = talker.hidden_projection(thinker_top_hidden[:, start:visible_end]).to(talker.dtype)
        assistant_hidden = assistant_embed_hidden + assistant_top_hidden
        if assistant_hidden.shape[1] != assistant_token_count - 1:
            raise RuntimeError(
                f'Unexpected assistant conditioning length: expected={assistant_token_count - 1}, '
                f'got={assistant_hidden.shape[1]}')
        # Omit the final assistant token so talker never trains on top-hidden states that are
        # unavailable at the streaming decode boundary.
        return assistant_hidden

    def _get_talker_max_seq_len(self) -> int:
        talker = self.talker
        text_cfg = getattr(talker.config, 'text_config', None)
        for cfg_obj in (text_cfg, talker.config, getattr(talker, 'model', None), getattr(getattr(talker, 'model', None), 'config', None)):
            if cfg_obj is None:
                continue
            value = getattr(cfg_obj, 'max_position_embeddings', None)
            if value:
                return int(value)
        args = get_args()
        value = getattr(args, 'max_length', None)
        return int(value) if value else 0

    @staticmethod
    def _get_talker_seq_multiple() -> int:
        args = get_args()
        if getattr(args, 'sequence_parallel', False):
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            if tp_size > 1:
                return int(tp_size)
        return 1

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        if multiple <= 1:
            return value
        return ((value + multiple - 1) // multiple) * multiple

    def _pad_talker_sequence_for_parallel(self, inputs_embeds: torch.Tensor,
                                          labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_multiple = self._get_talker_seq_multiple()
        padded_seq_len = self._round_up_to_multiple(int(inputs_embeds.shape[1]), seq_multiple)
        pad_len = padded_seq_len - int(inputs_embeds.shape[1])
        if pad_len <= 0:
            return inputs_embeds, labels
        pad_ids = torch.full(
            (inputs_embeds.shape[0], pad_len),
            self.talker.config.codec_pad_id,
            dtype=torch.long,
            device=inputs_embeds.device)
        pad_embeds = self.talker.get_input_embeddings()(pad_ids).to(self.talker.dtype)
        pad_labels = torch.full((labels.shape[0], pad_len), -100, dtype=torch.long, device=labels.device)
        return torch.cat((inputs_embeds, pad_embeds), dim=1), torch.cat((labels, pad_labels), dim=1)

    def _build_talker_interleaved_batch(self, thinker_embed: torch.Tensor, thinker_top_hidden: torch.Tensor,
                                        input_ids: torch.Tensor, rvq_codes: torch.Tensor,
                                        codec_turn_boundaries: torch.Tensor,
                                        codec_assistant_turn_indices: torch.Tensor,
                                        assistant_total_count: Optional[int] = None) -> Dict[str, object]:
        talker = self.talker
        assistant_segments = self._collect_assistant_segments(input_ids)
        if not assistant_segments:
            raise RuntimeError('No assistant segments found while building interleaved talker batch.')
        assistant_segments_by_idx = {seg['assistant_idx']: seg for seg in assistant_segments}
        codec_turn_boundaries = codec_turn_boundaries.long()
        codec_assistant_turn_indices = codec_assistant_turn_indices.long()
        assistant_shift = 0
        if assistant_total_count is not None and assistant_total_count > len(assistant_segments):
            assistant_shift = assistant_total_count - len(assistant_segments)
            self._debug_log(
                f'talker.assistant_suffix_align: assistant_total_count={assistant_total_count}, '
                f'present_assistants={len(assistant_segments)}, shift={assistant_shift}',
                force=True)
        turn_specs = []
        total_seq_len = 0
        for turn_idx in range(codec_assistant_turn_indices.shape[0]):
            assistant_idx = int(codec_assistant_turn_indices[turn_idx].item())
            if assistant_idx < 0:
                raise RuntimeError(f'codec turn contains negative assistant index: turn_idx={turn_idx}, value={assistant_idx}')
            local_assistant_idx = assistant_idx - assistant_shift
            if local_assistant_idx < 0:
                self._debug_log(
                    f'skip truncated prefix assistant turn: assistant_idx={assistant_idx}, shift={assistant_shift}',
                    force=True)
                continue
            segment = assistant_segments_by_idx.get(local_assistant_idx)
            if segment is None:
                raise RuntimeError(
                    f'codec turn references missing assistant segment: assistant_idx={assistant_idx}, '
                    f'local_assistant_idx={local_assistant_idx}, available={sorted(assistant_segments_by_idx.keys())}')
            code_start = int(codec_turn_boundaries[turn_idx].item())
            code_end = int(codec_turn_boundaries[turn_idx + 1].item())
            if code_end <= code_start:
                raise RuntimeError(
                    f'codec turn must contain a positive number of tokens: '
                    f'assistant_idx={assistant_idx}, start={code_start}, end={code_end}')
            turn_codes = rvq_codes[:, :, code_start:code_end]
            scaffold_embeds = self._build_talker_assistant_scaffold(
                thinker_embed,
                thinker_top_hidden,
                segment['start'],
                segment['end'])
            target_codec = self._build_target_codec_inputs(turn_codes)
            scaffold_len = int(scaffold_embeds.shape[1])
            labels_prefix = torch.full((1, scaffold_len), -100, dtype=torch.long, device=rvq_codes.device)
            turn_labels = torch.cat((labels_prefix, target_codec['labels']), dim=1)
            turn_input_len = scaffold_len + int(target_codec['inputs_embeds'].shape[1])
            if turn_labels.shape[1] != turn_input_len:
                raise RuntimeError(
                    f'talker turn input/label length mismatch: inputs={turn_input_len}, labels={turn_labels.shape[1]}')
            turn_specs.append({
                'assistant_idx': assistant_idx,
                'local_assistant_idx': local_assistant_idx,
                'scaffold_embeds': scaffold_embeds,
                'target_inputs': target_codec['inputs_embeds'],
                'labels': turn_labels,
                'scaffold_len': int(scaffold_embeds.shape[1]),
                'num_codec_tokens': int(target_codec['num_codec_tokens']),
                'codes': turn_codes,
                'input_len': turn_input_len,
            })
            total_seq_len += turn_input_len
        if not turn_specs:
            raise RuntimeError('No valid talker turns remained after assistant/codec alignment.')
        max_seq_len = self._get_talker_max_seq_len()
        seq_multiple = self._get_talker_seq_multiple()
        dropped_turns = 0
        while turn_specs and max_seq_len > 0 and self._round_up_to_multiple(total_seq_len, seq_multiple) > max_seq_len:
            removed = turn_specs.pop(0)
            total_seq_len -= removed['input_len']
            dropped_turns += 1
        if dropped_turns > 0:
            self._debug_log(
                f'talker.overlength: dropped_earliest_turns={dropped_turns}, kept_turns={len(turn_specs)}, '
                f'final_seq_len={total_seq_len}, max_seq_len={max_seq_len}',
                force=True)
        if not turn_specs:
            raise RuntimeError('All talker turns were dropped by max sequence length filtering.')
        full_inputs_embeds_list = []
        labels_list = []
        turn_infos = []
        seq_cursor = 0
        for turn in turn_specs:
            full_inputs_embeds_list.extend([turn['scaffold_embeds'], turn['target_inputs']])
            labels_list.append(turn['labels'])
            turn_infos.append({
                'assistant_idx': turn['assistant_idx'],
                'local_assistant_idx': turn['local_assistant_idx'],
                'hidden_start': seq_cursor + turn['scaffold_len'],
                'codec_len': turn['num_codec_tokens'],
                'codes': turn['codes'],
            })
            seq_cursor += turn['input_len']
        full_inputs_embeds = torch.cat(full_inputs_embeds_list, dim=1)
        talker_labels = torch.cat(labels_list, dim=1)
        if full_inputs_embeds.shape[1] != talker_labels.shape[1]:
            raise RuntimeError(
                f'talker interleaved input/label length mismatch: inputs={full_inputs_embeds.shape[1]}, '
                f'labels={talker_labels.shape[1]}')
        full_inputs_embeds, talker_labels = self._pad_talker_sequence_for_parallel(full_inputs_embeds, talker_labels)
        self._debug_log(
            f'talker.interleaved: turns={len(turn_infos)}, full_seq={full_inputs_embeds.shape[1]}, '
            f'valid_labels={int((talker_labels != -100).sum().item())}',
            force=True)
        self._tensor_stats('talker.interleaved.full_inputs_embeds', full_inputs_embeds, force=True)
        self._id_stats('talker.interleaved.labels', talker_labels, {
            'ignore': -100,
            'codec_bos': talker.config.codec_bos_id,
            'codec_eos': talker.config.codec_eos_token_id,
        }, force=True)
        return {
            'full_inputs_embeds': full_inputs_embeds,
            'talker_labels': talker_labels,
            'turn_infos': turn_infos,
        }

    def _extract_talker_hidden(self, talker_outputs) -> torch.Tensor:
        hidden_states = talker_outputs.hidden_states
        if isinstance(hidden_states, tuple):
            hidden_list = hidden_states[0]
            if isinstance(hidden_list, (tuple, list)):
                return hidden_list[-1]
        if isinstance(hidden_states, (tuple, list)):
            return hidden_states[-1]
        return talker_outputs.logits.new_zeros(())

    def _compute_code_predictor_loss(self, codec_hidden: torch.Tensor, rvq_codes: torch.Tensor) -> torch.Tensor:
        code_predictor = self.talker.code_predictor
        predictor_embeds = code_predictor.get_input_embeddings()
        batch_size, num_codec_tokens, hidden_dim = codec_hidden.shape
        hidden_flat = codec_hidden.reshape(-1, 1, hidden_dim)
        layer0_embed = self.talker.get_input_embeddings()(rvq_codes[:, 0, :]).reshape(-1, 1, hidden_dim)
        mtp_total_loss = codec_hidden.new_zeros((), dtype=torch.float32)
        num_mtp_layers = rvq_codes.shape[1] - 1
        for mtp_layer_idx in range(num_mtp_layers):
            embed_list = [hidden_flat, layer0_embed]
            for prev_layer in range(mtp_layer_idx):
                prev_codes = rvq_codes[:, prev_layer + 1, :]
                prev_embed = predictor_embeds[prev_layer](prev_codes).reshape(-1, 1, hidden_dim)
                embed_list.append(prev_embed)
            mtp_inputs = torch.cat(embed_list, dim=1).to(self.talker.dtype)
            target_labels = rvq_codes[:, mtp_layer_idx + 1, :].reshape(-1)
            mtp_outputs = code_predictor(
                input_ids=None, inputs_embeds=mtp_inputs, generation_steps=mtp_layer_idx, use_cache=False)
            mtp_logits = mtp_outputs.logits[:, -1, :]
            mtp_total_loss = mtp_total_loss + F.cross_entropy(mtp_logits.float(), target_labels)
            self._tensor_stats(f'mtp.layer{mtp_layer_idx}.logits', mtp_logits, force=True)
        return mtp_total_loss / max(num_mtp_layers, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params=None,
        packed_seq_params=None,
        rvq_codes: torch.Tensor = None,
        rvq_length: torch.Tensor = None,
        hist_rvq_codes: torch.Tensor = None,
        hist_rvq_length: torch.Tensor = None,
        codec_turn_boundaries: torch.Tensor = None,
        codec_turn_count: torch.Tensor = None,
        codec_assistant_turn_indices: torch.Tensor = None,
        assistant_total_count: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        args = get_args()
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            raise RuntimeError('qwen3_omni_e2e currently supports pipeline_model_parallel_size=1 only.')
        if getattr(args, 'context_parallel_size', 1) > 1:
            raise RuntimeError('qwen3_omni_e2e currently supports context_parallel_size=1 only.')
        if self.visual is None:
            raise RuntimeError('qwen3_omni_e2e requires visual/talker modules to be instantiated on the running rank.')
        self._validate_replicated_aux_grad_sync()
        self._maybe_force_eager_talker_attention()
        if decoder_input is not None:
            thinker_decoder_input = decoder_input
        elif self.pre_process:
            kwargs.update({'input_ids': input_ids, 'packed_seq_params': packed_seq_params})
            with self._patch_word_embeddings(kwargs):
                thinker_decoder_input = self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            thinker_decoder_input = None
            kwargs = {}
        thinker_decoder = getattr(self.language_model, 'decoder', None)
        if thinker_decoder is None:
            raise RuntimeError('Failed to find thinker decoder module for top-hidden capture.')
        thinker_top_hidden_local = None

        def _capture_top_hidden(_module, _inputs, output):
            nonlocal thinker_top_hidden_local
            thinker_top_hidden_local = self._extract_tensor(output)

        with ExitStack() as debug_stack:
            top_hidden_handle = thinker_decoder.register_forward_hook(_capture_top_hidden)
            debug_stack.callback(top_hidden_handle.remove)
            self._install_debug_hooks(debug_stack)
            self._tensor_stats('thinker.decoder_input_pre_lm', thinker_decoder_input, force=True)
            thinker_output = self.language_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=thinker_decoder_input,
                labels=labels,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )
        if not self.post_process:
            return thinker_output
        if thinker_top_hidden_local is None:
            raise RuntimeError('Failed to capture thinker top hidden states for talker fusion.')
        self._debug_log(
            'thinker top hidden captured from decoder output: '
            f'shape={tuple(thinker_top_hidden_local.shape)}, '
            f'dtype={thinker_top_hidden_local.dtype}, '
            f'requires_grad={thinker_top_hidden_local.requires_grad}',
            force=True)

        thinker_loss = self._mean_thinker_loss(thinker_output, labels)
        if rvq_codes is None or rvq_length is None:
            raise RuntimeError('rvq_codes and rvq_length are required for qwen3_omni_e2e training.')
        if codec_turn_boundaries is None or codec_turn_count is None or codec_assistant_turn_indices is None:
            raise RuntimeError(
                'codec_turn_boundaries, codec_turn_count and codec_assistant_turn_indices are required for '
                'assistant-only interleaved qwen3_omni_e2e training.')
        if assistant_total_count is None:
            raise RuntimeError('assistant_total_count is required for truncation-safe interleaved qwen3_omni_e2e training.')
        self._tensor_stats('rvq.codes_raw', rvq_codes, force=True)
        self._tensor_stats('rvq.length_raw', rvq_length, force=True)
        self._tensor_stats('rvq.turn_boundaries_raw', codec_turn_boundaries, force=True)
        self._tensor_stats('rvq.turn_count_raw', codec_turn_count, force=True)
        self._tensor_stats('rvq.turn_indices_raw', codec_assistant_turn_indices, force=True)
        self._tensor_stats('rvq.assistant_total_count_raw', assistant_total_count, force=True)
        rvq_codes = self._normalize_rvq_codes(rvq_codes)
        rvq_length = self._normalize_rvq_length(rvq_length)
        codec_turn_boundaries = self._normalize_codec_turn_boundaries(codec_turn_boundaries)
        codec_turn_count = self._normalize_codec_turn_count(codec_turn_count)
        codec_assistant_turn_indices = self._normalize_codec_turn_indices(codec_assistant_turn_indices)
        assistant_total_count = self._normalize_assistant_total_count(assistant_total_count)
        if rvq_codes.shape[0] != input_ids.shape[0] or rvq_length.shape[0] != input_ids.shape[0]:
            raise RuntimeError(
                f'RVQ batch mismatch: rvq_codes={tuple(rvq_codes.shape)}, rvq_length={tuple(rvq_length.shape)}, '
                f'input_ids_batch={input_ids.shape[0]}')
        if codec_turn_boundaries.shape[0] != input_ids.shape[0] or codec_turn_count.shape[0] != input_ids.shape[0]:
            raise RuntimeError(
                'Codec turn metadata batch mismatch: '
                f'boundaries={tuple(codec_turn_boundaries.shape)}, turn_count={tuple(codec_turn_count.shape)}, '
                f'input_ids_batch={input_ids.shape[0]}')
        if codec_assistant_turn_indices.shape[0] != input_ids.shape[0]:
            raise RuntimeError(
                'Codec assistant turn indices batch mismatch: '
                f'indices={tuple(codec_assistant_turn_indices.shape)}, input_ids_batch={input_ids.shape[0]}')
        if assistant_total_count.shape[0] != input_ids.shape[0]:
            raise RuntimeError(
                'assistant_total_count batch mismatch: '
                f'assistant_total_count={tuple(assistant_total_count.shape)}, input_ids_batch={input_ids.shape[0]}')
        thinker_embed = self._gather_sp_hidden(thinker_decoder_input) if thinker_decoder_input is not None else None
        if thinker_embed is None:
            raise RuntimeError('Failed to materialize thinker embedding states for assistant-only talker training.')
        thinker_embed = self._to_batch_seq_hidden(
            thinker_embed,
            batch_size=input_ids.shape[0],
            hidden_size=self.talker.config.thinker_hidden_size,
            name='thinker_embed')
        thinker_top_hidden = self._gather_sp_hidden(thinker_top_hidden_local)
        thinker_top_hidden = self._to_batch_seq_hidden(
            thinker_top_hidden,
            batch_size=input_ids.shape[0],
            hidden_size=self.talker.config.thinker_hidden_size,
            name='thinker_top_hidden')
        if thinker_top_hidden.shape[1] != thinker_embed.shape[1]:
            raise RuntimeError(
                f'thinker hidden sequence mismatch: embed={tuple(thinker_embed.shape)}, '
                f'top_hidden={tuple(thinker_top_hidden.shape)}')
        self._tensor_stats('thinker.embed_full', thinker_embed, force=True)
        self._tensor_stats('thinker.top_hidden_full', thinker_top_hidden, force=True)
        self._tensor_stats('rvq.codes', rvq_codes, force=True)
        self._tensor_stats('rvq.length', rvq_length, force=True)
        self._tensor_stats('rvq.turn_boundaries', codec_turn_boundaries, force=True)
        self._tensor_stats('rvq.turn_count', codec_turn_count, force=True)
        self._tensor_stats('rvq.turn_indices', codec_assistant_turn_indices, force=True)
        self._tensor_stats('rvq.assistant_total_count', assistant_total_count, force=True)

        ar_loss_sum = thinker_loss.new_zeros(())
        mlp_loss_sum = thinker_loss.new_zeros(())
        ar_tokens = input_ids.new_zeros((), dtype=torch.int)
        mlp_tokens = input_ids.new_zeros((), dtype=torch.int)
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            current_len = int(rvq_length[b].item())
            turn_count = int(codec_turn_count[b].item())
            if current_len <= 0 or turn_count <= 0:
                raise RuntimeError(f'Invalid talker sample metadata: sample={b}, current_len={current_len}, turn_count={turn_count}')
            sample_codes = rvq_codes[b:b + 1, :, :current_len]
            sample_boundaries = codec_turn_boundaries[b, :turn_count + 1].clone()
            sample_turn_indices = codec_assistant_turn_indices[b, :turn_count].clone()
            if sample_boundaries.shape[0] != turn_count + 1:
                raise RuntimeError(
                    f'Invalid codec boundary shape for sample={b}: shape={tuple(sample_boundaries.shape)}, '
                    f'turn_count={turn_count}')
            if (sample_boundaries < 0).any() or (sample_turn_indices < 0).any():
                raise RuntimeError(
                    f'Negative codec metadata in sample={b}: boundaries={sample_boundaries.tolist()}, '
                    f'indices={sample_turn_indices.tolist()}')
            if int(sample_boundaries[0].item()) != 0 or not torch.all(sample_boundaries[1:] > sample_boundaries[:-1]):
                raise RuntimeError(f'Invalid codec boundaries in sample={b}: {sample_boundaries.tolist()}')
            last_boundary = int(sample_boundaries[-1].item())
            if last_boundary != current_len:
                raise RuntimeError(
                    f'Codec boundaries must cover the full RVQ length in sample={b}: '
                    f'last_boundary={last_boundary}, current_len={current_len}')
            sample_input_ids = input_ids[b:b + 1]
            sample_embed = thinker_embed[b:b + 1]
            sample_top_hidden = thinker_top_hidden[b:b + 1]
            sample_assistant_total_count = int(assistant_total_count[b].item())
            self._tensor_stats(f'rvq.sample{b}.codes', sample_codes, force=True)
            self._tensor_stats(f'rvq.sample{b}.turn_boundaries', sample_boundaries, force=True)
            self._tensor_stats(f'rvq.sample{b}.turn_indices', sample_turn_indices, force=True)
            interleaved = self._build_talker_interleaved_batch(
                sample_embed,
                sample_top_hidden,
                sample_input_ids,
                sample_codes,
                sample_boundaries,
                sample_turn_indices,
                sample_assistant_total_count)
            talker_attention_mask = self._build_talker_attention_mask(
                interleaved['full_inputs_embeds'].shape[1], device=interleaved['full_inputs_embeds'].device)
            self._tensor_stats(f'talker.sample{b}.attention_mask', talker_attention_mask, force=True)
            talker_outputs = self.talker(
                inputs_embeds=interleaved['full_inputs_embeds'],
                attention_mask=talker_attention_mask,
                labels=None,
                output_hidden_states=True,
                return_dict=True,
            )
            self._tensor_stats(f'talker.sample{b}.logits', talker_outputs.logits, force=True)
            ar_loss_sample = self._causal_ce_loss(talker_outputs.logits, interleaved['talker_labels'])
            ar_tokens_sample = (interleaved['talker_labels'][:, 1:] != -100).sum().to(torch.int)
            ar_loss_sum = ar_loss_sum + ar_loss_sample.float() * ar_tokens_sample.to(ar_loss_sample.dtype)
            ar_tokens = ar_tokens + ar_tokens_sample
            talker_hidden_state = self._extract_talker_hidden(talker_outputs)
            self._tensor_stats(f'talker.sample{b}.last_hidden_state', talker_hidden_state, force=True)
            codec_hidden_segments = []
            mtp_code_segments = []
            for turn_info in interleaved['turn_infos']:
                hidden_start = int(turn_info['hidden_start'])
                codec_len = int(turn_info['codec_len'])
                codec_hidden = talker_hidden_state[:, hidden_start:hidden_start + codec_len, :]
                if codec_hidden.shape[1] != codec_len:
                    raise RuntimeError(
                        f'Codec hidden slice length mismatch in sample={b}: start={hidden_start}, '
                        f'expected={codec_len}, got={codec_hidden.shape[1]}')
                codec_hidden_segments.append(codec_hidden)
                mtp_code_segments.append(turn_info['codes'])
            if not codec_hidden_segments:
                raise RuntimeError(f'No codec hidden segments collected for MTP in sample={b}.')
            codec_hidden = torch.cat(codec_hidden_segments, dim=1)
            mtp_codes = torch.cat(mtp_code_segments, dim=2)
            if codec_hidden.shape[1] != mtp_codes.shape[2]:
                raise RuntimeError(
                    f'MTP alignment mismatch in sample={b}: codec_hidden={codec_hidden.shape[1]}, '
                    f'mtp_codes={mtp_codes.shape[2]}')
            self._tensor_stats(f'talker.sample{b}.codec_hidden_concat', codec_hidden, force=True)
            self._tensor_stats(f'talker.sample{b}.mtp_codes_concat', mtp_codes, force=True)
            mtp_loss_sample = self._compute_code_predictor_loss(codec_hidden, mtp_codes)
            mtp_tokens_sample = input_ids.new_tensor(int(mtp_codes.shape[2] * max(mtp_codes.shape[1] - 1, 0)), dtype=torch.int)
            mlp_loss_sum = mlp_loss_sum + mtp_loss_sample.float() * mtp_tokens_sample.to(mtp_loss_sample.dtype)
            mlp_tokens = mlp_tokens + mtp_tokens_sample
        ar_loss = ar_loss_sum / ar_tokens.to(ar_loss_sum.dtype).clamp_min(1)
        mlp_loss = mlp_loss_sum / mlp_tokens.to(mlp_loss_sum.dtype).clamp_min(1)
        self._tensor_stats('loss.thinker', thinker_loss, force=True)
        self._tensor_stats('loss.ar', ar_loss, force=True)
        self._tensor_stats('loss.mlp', mlp_loss, force=True)
        thinker_tokens = (labels != -100).sum().to(torch.int) if labels is not None else input_ids.new_zeros((), dtype=torch.int)
        thinker_loss_sum = thinker_loss * thinker_tokens.to(thinker_loss.dtype)
        return {
            'thinker_loss': thinker_loss,
            'ar_loss': ar_loss,
            'mlp_loss': mlp_loss,
            'thinker_loss_sum': thinker_loss_sum,
            'ar_loss_sum': ar_loss_sum,
            'mlp_loss_sum': mlp_loss_sum,
            'thinker_tokens': thinker_tokens,
            'ar_tokens': ar_tokens,
            'mlp_tokens': mlp_tokens,
        }


class Qwen3OmniE2EBridge(Qwen3OmniBridge):
    e2e_visual_module_specs = (
        ('thinker', 'thinker'),
        ('talker', 'talker'),
        ('code2wav', 'code2wav'),
    )
    required_prefixes: List[str] = [
        'thinker.model.layers.',
        'thinker.model.embed_tokens.',
        'thinker.model.norm.',
        'thinker.lm_head.',
        'talker.model.layers.',
        'talker.model.codec_embedding.',
        'talker.model.norm.',
        'talker.hidden_projection.',
        'talker.text_projection.',
        'talker.codec_head.',
        'talker.code_predictor.model.codec_embedding.',
        'talker.code_predictor.model.layers.',
        'talker.code_predictor.model.norm.',
        'talker.code_predictor.lm_head.',
        'thinker.audio_tower.',
        'thinker.visual.',
        'code2wav.',
    ]
    expected_layer_counts = {
        'thinker.model.layers.': 48,
        'talker.model.layers.': 20,
        'talker.code_predictor.model.layers.': 5,
    }

    @classmethod
    def _collect_missing_prefixes(cls, keys: Iterable[str]) -> List[str]:
        key_list = list(keys)
        return [prefix for prefix in cls.required_prefixes if not any(k.startswith(prefix) for k in key_list)]

    def _validate_prefixes(self, keys: Iterable[str], stage: str) -> None:
        missing = self._collect_missing_prefixes(keys)
        if missing:
            raise RuntimeError(f'[{stage}] missing required prefixes: {missing}')

    @classmethod
    def _infer_layer_count(cls, keys: Iterable[str], prefix: str) -> int:
        indices = set()
        for key in keys:
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            layer_str = suffix.split('.', 1)[0]
            if layer_str.isdigit():
                indices.add(int(layer_str))
        return 0 if not indices else max(indices) + 1

    def _validate_layer_counts(self, keys: Iterable[str], stage: str) -> None:
        key_list = list(keys)
        for prefix, expected in self.expected_layer_counts.items():
            actual = self._infer_layer_count(key_list, prefix)
            if actual != expected:
                raise RuntimeError(f'[{stage}] layer count mismatch for `{prefix}`: expected={expected}, actual={actual}')

    def _convert_native_talker_state(self, mg_talker, hf_state_dict, hf_prefix: str, to_mcore: bool):
        if to_mcore:
            if mg_talker is None:
                raise RuntimeError('qwen3_omni_e2e model is missing native `talker` required for load_weights.')
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        talker_text_config = mg_talker.config.text_config if mg_talker is not None else self.hf_model.talker.config.text_config
        saved_hf_layers = self.hf_layers
        self.hf_layers = deep_getattr(self.hf_model, 'talker.model.layers')
        try:
            with _patch_qwen3_omni_submodel_args(talker_text_config):
                self._set_state_dict(mg_talker.model, 'codec_embedding.weight', hf_state_dict, 'model.codec_embedding.weight', to_mcore)
                num_layers = self.expected_layer_counts['talker.model.layers.']
                for layer_idx in range(num_layers):
                    mg_layer = None if mg_talker is None else mg_talker.model.layers[layer_idx]
                    layer_state = self._set_layer_state(mg_layer, hf_state_dict, 'model.layers.', layer_idx, to_mcore)
                    if not to_mcore:
                        hf_state_dict.update(layer_state)
                self._set_state_dict(mg_talker.model, 'norm.weight', hf_state_dict, 'model.norm.weight', to_mcore)
        finally:
            self.hf_layers = saved_hf_layers
        hf_state_dict.update(self._set_module(None if mg_talker is None else mg_talker.text_projection, hf_state_dict,
                                              'text_projection.', to_mcore))
        hf_state_dict.update(self._set_module(None if mg_talker is None else mg_talker.hidden_projection, hf_state_dict,
                                              'hidden_projection.', to_mcore))
        hf_state_dict.update(self._set_module(None if mg_talker is None else mg_talker.codec_head, hf_state_dict,
                                              'codec_head.', to_mcore))
        hf_state_dict.update(self._set_module(None if mg_talker is None else mg_talker.code_predictor, hf_state_dict,
                                              'code_predictor.', to_mcore))
        if to_mcore:
            return {}
        return self._add_prefix(hf_state_dict, hf_prefix)

    def _convert_pre_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        lm_model = getattr(mg_model, 'language_model') if self.args.is_multimodal else mg_model
        self._set_state_dict(lm_model, 'embedding.word_embeddings.weight', hf_state_dict, self.hf_embed_key, to_mcore)
        if self.args.is_multimodal:
            visual = getattr(mg_model, 'visual', None)
            for hf_prefix_name, visual_attr in self.e2e_visual_module_specs:
                if hf_prefix_name == 'talker':
                    talker_state = self._convert_native_talker_state(getattr(mg_model, 'talker', None), hf_state_dict,
                                                                     f'{hf_prefix}{hf_prefix_name}.', to_mcore)
                    if not to_mcore:
                        hf_state_dict.update(self._remove_prefix(talker_state, f'{hf_prefix}'))
                    continue
                if visual is None:
                    mg_module = None
                else:
                    if not hasattr(visual, visual_attr):
                        raise RuntimeError(
                            f'qwen3_omni_e2e visual module is missing `{visual_attr}` required for `{hf_prefix_name}`')
                    mg_module = getattr(visual, visual_attr)
                hf_state_dict.update(self._set_module(mg_module, hf_state_dict, f'{hf_prefix}{hf_prefix_name}.', to_mcore))
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _get_reference_index_path(self) -> str:
        candidate_paths = []
        cached = getattr(self, '_reference_index_path', None)
        if cached:
            candidate_paths.append(cached)
        candidate_paths.append(os.path.join(self.args.model_dir, 'model.safetensors.index.json'))
        for index_path in candidate_paths:
            if index_path and os.path.isfile(index_path):
                self._reference_index_path = index_path
                return index_path
        raise RuntimeError(
            'qwen3_omni_e2e export validation requires an existing `model.safetensors.index.json`; '
            f'tried: {candidate_paths}')

    def _load_reference_weight_keys(self) -> Set[str]:
        index_path = self._get_reference_index_path()
        with open(index_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        weight_map = payload.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f'invalid or empty safetensors index at `{index_path}`')
        return set(weight_map.keys())

    def _validate_exact_export_coverage(self, exported_keys: Iterable[str], stage: str) -> None:
        expected_keys = self._load_reference_weight_keys()
        exported_key_set = set(exported_keys)
        missing = sorted(expected_keys - exported_key_set)
        unexpected = sorted(exported_key_set - expected_keys)
        if missing or unexpected:
            details = []
            if missing:
                preview = ', '.join(missing[:24])
                remainder = len(missing) - min(len(missing), 24)
                if remainder > 0:
                    preview = f'{preview}, ... (+{remainder} more)'
                details.append(f'missing {len(missing)} keys: {preview}')
            if unexpected:
                preview = ', '.join(unexpected[:24])
                remainder = len(unexpected) - min(len(unexpected), 24)
                if remainder > 0:
                    preview = f'{preview}, ... (+{remainder} more)'
                details.append(f'unexpected {len(unexpected)} keys: {preview}')
            raise RuntimeError(
                f'[{stage}] exported safetensors keys mismatch vs `{self._get_reference_index_path()}`; '
                + '; '.join(details))

    def load_weights(self, mg_model, hf_model_dir: str, is_peft_format: bool = False, adapter_name: str = 'default'):
        self._reference_index_path = os.path.join(hf_model_dir, 'model.safetensors.index.json')
        super().load_weights(mg_model, hf_model_dir, is_peft_format=is_peft_format, adapter_name=adapter_name)
        if os.environ.get('QWEN3_OMNI_E2E_BRIDGE_ROUNDTRIP', '0') == '1':
            # Optional expensive check: export once and verify key/prefix coverage.
            exported_keys: Set[str] = set()
            for k, _ in self.export_weights([mg_model], target_device='cpu', only_last_rank=True, tqdm_desc='RoundTrip: '):
                exported_keys.add(k)
            self._validate_prefixes(exported_keys, 'round_trip_export')
            self._validate_layer_counts(exported_keys, 'round_trip_export')
            if not is_peft_format:
                self._validate_exact_export_coverage(exported_keys, 'round_trip_export')

    def export_weights(self,
                       mg_models,
                       target_device=None,
                       only_last_rank: bool = False,
                       is_peft_format: bool = False,
                       tqdm_desc: str = 'Exporting: '):
        exported_keys: Set[str] = set()
        for k, v in super().export_weights(
                mg_models,
                target_device=target_device,
                only_last_rank=only_last_rank,
                is_peft_format=is_peft_format,
                tqdm_desc=tqdm_desc):
            exported_keys.add(k)
            yield k, v
        validate_on_this_rank = (not only_last_rank) or is_last_rank()
        if validate_on_this_rank:
            self._validate_prefixes(exported_keys, 'export')
            self._validate_layer_counts(exported_keys, 'export')
            if not is_peft_format:
                self._validate_exact_export_coverage(exported_keys, 'export')


def _qwen3_omni_e2e_extra_args_provider(parser):
    parser.add_argument('--qwen3-omni-e2e-thinker-loss-weight', type=float, default=1.0)
    parser.add_argument('--qwen3-omni-e2e-ar-loss-weight', type=float, default=1.0)
    parser.add_argument('--qwen3-omni-e2e-mlp-loss-weight', type=float, default=1.0)
    parser.add_argument('--qwen3-omni-e2e-debug-trace', type=Qwen3OmniE2EModel._parse_bool, default=True)
    parser.add_argument('--qwen3-omni-e2e-debug-trace-steps', type=int, default=2)
    parser.add_argument('--qwen3-omni-e2e-debug-thinker-last-layers', type=int, default=6)
    parser.add_argument('--qwen3-omni-e2e-debug-all-ranks', type=Qwen3OmniE2EModel._parse_bool, default=False)
    parser.add_argument('--qwen3-omni-e2e-force-eager-talker-attn', type=Qwen3OmniE2EModel._parse_bool, default=False)
    return parser


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.qwen3_omni_e2e,
        [ModelType.qwen3_omni_e2e],
        model_cls=Qwen3OmniE2EModel,
        bridge_cls=Qwen3OmniE2EBridge,
        visual_cls=Qwen3OmniE2E_Vit,
        extra_args_provider=_qwen3_omni_e2e_extra_args_provider))
