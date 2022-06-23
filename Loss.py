import torch
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
            device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
            labels = torch.eq(target, i).long()

            batch_size = features.shape[0]
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

            # compute logits
            anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

            features_norm = torch.norm(features, dim=1, keepdim=True)
            cov_features_norm = torch.matmul(features_norm, features_norm.T)

            logits = torch.div(anchor_dot_contrast, cov_features_norm)

            # # for numerical stability
            # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            # logits = anchor_dot_contrast - logits_max.detach()  # todo why

            # mask-out self-contrast cases
            logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            losses.append(loss.mean())

        final_loss = 0
        for i, loss in enumerate(losses):
            final_loss += torch.exp(-self.log_vars[i]) * loss + self.log_vars[i]
        return final_loss
