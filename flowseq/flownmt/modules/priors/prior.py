import math
from typing import Dict, Tuple, Union
import torch
import torch.nn as nn

from flownmt.flows.nmt import NMTFlow
from flownmt.modules.priors.length_predictors import LengthPredictor


class Prior(nn.Module):
    """
    class for Prior with a NMTFlow inside
    """
    _registry = dict()

    def __init__(self, flow: NMTFlow):
        super(Prior, self).__init__()
        assert flow.inverse, 'prior flow should have inverse mode'
        self.flow = flow
        self.length_unit = max(2, 2 ** (self.flow.levels - 1))
        self.features = self.flow.features

    def sync(self):
        self.flow.sync()

    def decode(self, epsilon: torch.Tensor, tgt_mask: torch.Tensor,
               src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            epsilon: Tensor
                epslion [batch, tgt_length, nz]
            tgt_mask: Tensor
                tensor of target masks [batch, tgt_length]
            src: Tensor
                source encoding [batch, src_length, hidden_size]
            src_mask: Tensor
                tensor of source masks [batch, src_length]

        Returns: Tensor1, Tensor2
            Tensor1: decoded latent code z [batch, tgt_length, nz]
            Tensor2: log probabilities [batch]

        """
        # [batch, tgt_length, nz]
        z, logdet = self.flow.fwdpass(epsilon, tgt_mask, src, src_mask)
        # [batch, tgt_length, nz]
        log_probs = epsilon.mul(epsilon) + math.log(math.pi * 2.0)
        # apply mask
        log_probs = log_probs.mul(tgt_mask.unsqueeze(2))
        # [batch]
        log_probs = log_probs.view(z.size(0), -1).sum(dim=1).mul(-0.5) + logdet
        return z, log_probs

    def sample(self, src: torch.Tensor,
               src_mask: torch.Tensor,
               length = 1,
               nsamples = 1,
               tau=0.0,
               include_zero=False) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """

        Args:
            nlengths: int
                number of lengths per sentence
            nsamples: int
                number of samples per sentence per length
            src: Tensor
                source encoding [batch, src_length, hidden_size]
            ctx: Tensor
                tensor for global state [batch, hidden_size]
            src_mask: Tensor
                tensor of masks [batch, src_length]
            tau: float (default 0.0)
                temperature of density
            include_zero: bool (default False)
                include zero sample

        Returns: (Tensor1, Tensor2, Tensor3), (Tensor4, Tensor5), (Tensor6, Tensor7, Tensor8)
            Tensor1: samples from the prior [batch * nlengths * nsamples, tgt_length, nz]
            Tensor2: log probabilities [batch * nlengths * nsamples]
            Tensor3: target masks [batch * nlengths * nsamples, tgt_length]
            Tensor4: lengths [batch * nlengths]
            Tensor5: log probabilities of lengths [batch * nlengths]
            Tensor6: source encoding with shape [batch * nlengths * nsamples, src_length, hidden_size]
            Tensor7: tensor for global state [batch * nlengths * nsamples, hidden_size]
            Tensor8: source masks with shape [batch * nlengths * nsamples, src_length]

        """
        # batch = src.size(0)
        # batch_nlen = batch * nlengths
        # [batch, nlenths]
        # lengths, log_probs_length = self.predict_length(ctx, src_mask, topk=nlengths)
        # [batch * nlengths]
        # log_probs_length = log_probs_length.view(-1)
        # lengths = lengths.view(-1)
        # max_length = lengths.max().item()
        # [batch * nlengths, max_length]
        # tgt_mask = torch.arange(max_length).to(src.device).unsqueeze(0).expand(batch_nlen, max_length).lt(lengths.unsqueeze(1)).float()


        # MY_CHANGES:

        max_length = length
        if max_length % 4 != 0:
            max_length += 4 - (max_length % 4)

        batch = src.size(0)
        tgt_mask = torch.arange(max_length).to(src.device).unsqueeze(0).expand(batch, max_length).lt(length).float()

        # [batch, nsamples, tgt_length, nz]
        epsilon = src.new_empty(batch, nsamples, max_length, self.features).normal_()
        if include_zero:
            epsilon[:, 0].zero_()
        # [batch * nsamples, tgt_length, nz]
        epsilon = epsilon.view(-1, max_length, self.features)

        if nsamples > 1:
            # [batch, nsamples, src_length, hidden_size]
            src = src.unsqueeze(1) + src.new_zeros(batch, nsamples, *src.size()[1:])
            # [batch * nsamples, src_length, hidden_size]
            src = src.view(batch * nsamples, *src.size()[2:])
            # [batch, nsamples, src_length]
            src_mask = src_mask.unsqueeze(1) + src_mask.new_zeros(batch, nsamples, src_mask.size(1))
            # [batch * nsamples, src_length]
            src_mask = src_mask.view(batch * nsamples, src_mask.size(2))
            # [batch * nsamples, tgt_length]
            tgt_mask = tgt_mask.unsqueeze(1) + tgt_mask.new_zeros(batch, nsamples, tgt_mask.size(1))
            # [batch * nsamples, tgt_length]
            tgt_mask = tgt_mask.view(batch * nsamples, tgt_mask.size(2))

        # [batch * nlength * nsamples, tgt_length, nz]
        z, log_probs = self.decode(epsilon, tgt_mask, src, src_mask)

        return z[:, :length, :], log_probs

    def log_probability(self,
                        z: torch.Tensor,
                        tgt_mask: torch.Tensor,
                        src: torch.Tensor,
                        src_mask: torch.Tensor,
                        ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """

        Args:
            z: Tensor
                tensor of latent code [batch, length, nz]
            tgt_mask: Tensor
                tensor of target masks [batch, length]
            src: Tensor
                source encoding [batch, src_length, hidden_size]
            src_mask: Tensor
                tensor of source masks [batch, src_length]

        Returns: log probabilities of z [batch]

        """

        # [batch, length, nz]
        epsilon, logdet = self.flow.bwdpass(z, tgt_mask, src, src_mask)
        # [batch, tgt_length, nz]
        log_probs = epsilon.mul(epsilon) + math.log(math.pi * 2.0)
        # apply mask
        log_probs = log_probs.mul(tgt_mask.unsqueeze(2))
        log_probs = log_probs.view(z.size(0), -1).sum(dim=1).mul(-0.5) + logdet
        return log_probs

    def init(self, z, tgt_mask, src, src_mask, init_scale=1.0):
        return self.flow.bwdpass(z, tgt_mask, src, src_mask, init=True, init_scale=init_scale)

    @classmethod
    def register(cls, name: str):
        Prior._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return Prior._registry[name]

    @classmethod
    def from_params(cls, params: Dict) -> "Prior":
        flow_params = params.pop('flow')
        flow = NMTFlow.from_params(flow_params)
        return Prior(flow)


Prior.register('normal')
