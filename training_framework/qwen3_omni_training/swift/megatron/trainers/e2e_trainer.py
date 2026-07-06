# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial

import torch
from megatron.training import get_args, get_timers

from .trainer import MegatronTrainer


class Qwen3OmniE2ETrainer(MegatronTrainer):

    def loss_func_e2e(self, output_tensor: dict):
        args = get_args()
        thinker_loss = output_tensor['thinker_loss'].float()
        ar_loss = output_tensor['ar_loss'].float()
        mlp_loss = output_tensor['mlp_loss'].float()
        thinker_loss_sum = output_tensor['thinker_loss_sum'].float()
        ar_loss_sum = output_tensor['ar_loss_sum'].float()
        mlp_loss_sum = output_tensor['mlp_loss_sum'].float()
        thinker_tokens = output_tensor['thinker_tokens'].detach().clone().to(torch.int)
        ar_tokens = output_tensor['ar_tokens'].detach().clone().to(torch.int)
        mlp_tokens = output_tensor['mlp_tokens'].detach().clone().to(torch.int)
        w_thinker = getattr(args, 'qwen3_omni_e2e_thinker_loss_weight', 1.0)
        w_ar = getattr(args, 'qwen3_omni_e2e_ar_loss_weight', 1.0)
        w_mlp = getattr(args, 'qwen3_omni_e2e_mlp_loss_weight', 1.0)
        total_loss = w_thinker * thinker_loss_sum + w_ar * ar_loss_sum + w_mlp * mlp_loss_sum
        local_num_tokens = thinker_tokens + ar_tokens + mlp_tokens
        reporting_lm_loss = torch.stack([total_loss.detach(), local_num_tokens.to(total_loss.dtype)])
        reporting = {
            'lm loss': reporting_lm_loss,
            'loss thinker': thinker_loss.detach(),
            'loss ar': ar_loss.detach(),
            'loss mlp': mlp_loss.detach(),
        }
        return total_loss, local_num_tokens, reporting

    def forward_step(self, data_iterator, model):
        timers = get_timers()
        vp_stage = model.module.module.vp_stage
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator, vp_stage)
        timers('batch-generator').stop()
        labels = data.get('labels')
        with self.stimer:
            output_tensor = model(**data)
        if isinstance(output_tensor, dict) and {
                'thinker_loss_sum', 'ar_loss_sum', 'mlp_loss_sum', 'thinker_tokens', 'ar_tokens', 'mlp_tokens'
        }.issubset(output_tensor.keys()):
            loss_func = self.loss_func_e2e
        else:
            loss_func = partial(self.loss_func, labels=labels, packed_seq_params=data.get('packed_seq_params'))
        return output_tensor, loss_func
