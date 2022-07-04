import torch
import numpy as np
from torch import nn


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, target):
        batch_size = target.shape[0]

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        labels = target.contiguous().view(-1, 1)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # todo why

        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # log_prob_sum = (mask * log_prob).sum(1)
        # cardinality = mask.sum(1)
        # vid_idx = cardinality.nonzero().detach()
        # log_prob_sum = log_prob_sum[vid_idx]
        # cardinality = cardinality[vid_idx]

        # loss
        # loss = - (self.temperature / self.base_temperature) * (log_prob_sum / cardinality)
        loss = - (self.temperature / self.base_temperature) * (mask * log_prob).sum(1) / mask.sum(1)
        return loss.mean()
