import torch
import numpy as np
from torch import nn


class ContrastiveLoss(nn.Module):

    def __init__(self, num_classes, temperature=0.07, base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_classes = num_classes
        self.log_vars = nn.Parameter(torch.zeros((num_classes)))

    def forward(self, preds, target):
        losses = []
        for i, features in enumerate(preds):
            batch_size = target.shape[0]

            # labels = target.numpy()
            # idx = np.where(target == i)[0]
            # each_class = batch_size / self.num_classes
            # div_scale = each_class // (each_class / (self.num_classes - 1))
            # div_scale = int(div_scale - 1)
            #
            # for j in range(self.num_classes):
            #     if j == i:
            #         continue
            #     sub_idx = np.where(labels == j)[0]
            #     np.random.shuffle(sub_idx)
            #
            #     if len(sub_idx) > div_scale:
            #         sub_idx = sub_idx[:round(len(sub_idx) / div_scale)]
            #     idx = np.concatenate((idx, sub_idx), axis=0)
            #
            # size = len(idx)
            # labels = target[idx]
            # features = features[idx]

            device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
            labels = torch.eq(target, i).float().to(device)

            labels = labels.contiguous().view(-1, 1)
            mask = torch.logical_and(labels, labels.T).float().to(device)

            # compute logits
            anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

            # In last layer, the L2 normalization has done
            # features_norm = torch.norm(features, dim=1, keepdim=True)
            # cov_features_norm = torch.matmul(features_norm, features_norm.T)
            #
            # logits = torch.div(anchor_dot_contrast, cov_features_norm)

            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()  # todo why

            # mask-out self-contrast cases
            logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            log_prob_sum = (mask * log_prob).sum(1)
            cardinality = mask.sum(1)
            vid_idx = cardinality.nonzero().detach()
            log_prob_sum = log_prob_sum[vid_idx]
            cardinality = cardinality[vid_idx]
            # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * (log_prob_sum / cardinality)
            losses.append(loss.mean())

        final_loss = 0
        for i, loss in enumerate(losses):
            final_loss += torch.exp(-self.log_vars[i]) * loss + self.log_vars[i]
        return final_loss
