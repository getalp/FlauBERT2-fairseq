# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F
import torch

from fairseq import (
    utils,
    metrics
)
from fairseq.criterions import (
    LegacyFairseqCriterion,
    register_criterion
)
from typing import (
    List,
    Dict,
    Any
)

from fairseq.criterions.fairseq_criterion import LegacyFairseqCriterion


@register_criterion('span_bert_loss')
class SpanPairLoss(LegacyFairseqCriterion):
    """Implementation for loss of SpanBert
        Combine masked language model loss with the SBO loss. 
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args
        self.aux_loss_weight = getattr(args, 'pair_loss_weight', 0)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        lm_targets = sample["lm_target"]
        masked_tokens = lm_targets.ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        if masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        net_output = model(**sample['net_input'], masked_tokens=masked_tokens)[0]

        if masked_tokens is not None:
            lm_targets = lm_targets[masked_tokens]

        # mlm loss
        lm_logits = net_output[0]
        lm_logits = lm_logits.view(-1, lm_logits.size(-1))
        lm_loss = F.cross_entropy(
            lm_logits,
            lm_targets.view(-1),
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )

        # SBO loss
        pair_target_logits = net_output[1]
        pair_target_logits = pair_target_logits.view(-1, pair_target_logits.size(-1))
        pair_targets = sample['pair_targets'].view(-1)
        pair_loss = F.cross_entropy(
            pair_target_logits,
            pair_targets,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )

        nsentences = masked_tokens.size(0)
        ntokens = lm_targets.numel()
        npairs = pair_targets.ne(self.padding_idx).sum() + 1

        sample_size = nsentences if self.args.sentence_avg else ntokens
        loss = lm_loss / ntokens + (self.aux_loss_weight * pair_loss / npairs)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'lm_loss': utils.item(lm_loss.data) if reduce else lm_loss.data,
            'pair_loss':  utils.item(pair_loss.data) if reduce else pair_loss.data,
            'ntokens': ntokens,
            'npairs': npairs,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'aux_loss_weight': self.aux_loss_weight
        }
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        """Aggregate logging outputs from data parallel training."""
        lm_loss_sum = sum(log.get('lm_loss', 0) for log in logging_outputs)
        pair_loss_sum = sum(log.get('pair_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        npairs = sum(log.get('npairs', 0) for log in logging_outputs)
        aux_loss_weight = max(log.get('aux_loss_weight', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        lm_loss = lm_loss_sum / ntokens / math.log(2)
        pair_loss = aux_loss_weight * pair_loss_sum / npairs / math.log(2)
        agg_loss = lm_loss + pair_loss

        agg_output = {
            'loss': agg_loss,
            'lm_loss': lm_loss,
            'pair_loss': pair_loss,
            'nll_loss': lm_loss,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        for k, v in agg_output.items():
            if k in {"nsentences", "ntokens", "sample_size"}:
                continue
            metrics.log_scalar(k, v)
