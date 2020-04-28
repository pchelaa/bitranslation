# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss


@register_criterion('label_smoothed_cross_entropy_with_kl')
class LabelSmoothedCrossEntropyCriterionWithKL(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        encoder_output, decoder_output, z = model(**sample['net_input'])
        loss, nll_loss = self.compute_rec_loss(model, decoder_output, sample, reduce=reduce)
        kl_loss = self.compute_kl_loss(model, encoder_output, decoder_output, z, reduce=reduce)
        loss += kl_loss
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'kl_loss': kl_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_rec_loss(self, model, decoder_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(decoder_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, decoder_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    # MY_CHANGES
    def compute_kl_loss(self, model, encoder_output, decoder_output, z, reduce):
        # DEBUG REGIME
        if True:
            batch_size = encoder_output.encoder_out.size(1)
            n_samples = 5  #samples from prior
            log_probs_prior = 0.5 * encoder_output.encoder_out.new_ones(batch_size, n_samples)
        else:
            log_probs_prior = model.get_prior_log_probability(
                encoder_output, decoder_output, z, log_probs=True
            )

        # THIS MUST BE CHANGED!
        log_probs_posterior = log_probs_prior
        KL = (log_probs_posterior - log_probs_prior).mean(dim=1)
        KL = KL.sum() if reduce else KL
        return KL

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
