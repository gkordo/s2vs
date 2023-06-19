import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, similarity, labels):
        assert similarity.shape == labels.shape

        # Get all negatives pairs
        identity = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
        non_matches = labels == 0
        nontrivial_matches = labels * (~identity)

        # Compute the InfoNCE loss (Eq. 1 & 2)
        logits = (similarity / self.temperature).exp()
        partitions = logits + ((non_matches * logits).sum(dim=1) + 1e-6).unsqueeze(1)
        probabilities = logits / partitions
        infonce_loss = (
                (-probabilities.log() * nontrivial_matches).sum(dim=1)
                / nontrivial_matches.sum(dim=1)
        ).mean()
        return infonce_loss

    def __repr__(self, ):
        return '{}(temperature={})'.format(self.__class__.__name__, self.temperature)


class SSHNLoss(nn.Module):

    def __init__(self, eps=1e-3):
        super(SSHNLoss, self).__init__()
        self.eps = eps

    def forward(self, similarity, labels):
        assert similarity.shape == labels.shape

        # Get the hardest negative value for each video in the batch
        non_matches = labels == 0
        small_value = torch.tensor(-100.0).to(similarity)
        max_non_match_sim, _ = torch.where(non_matches, similarity, small_value).max(
            dim=1, keepdim=True
        )

        # Compute the SSHN loss (Eq. 3)
        hardneg_loss = -(1-max_non_match_sim).clamp(min=self.eps).log().mean()
        self_loss = -torch.diagonal(similarity).clamp(min=self.eps).log().mean()

        return hardneg_loss, self_loss

    def __repr__(self, ):
        return '{}()'.format(self.__class__.__name__)


class SimilarityRegularizationLoss(nn.Module):

    def __init__(self, min_val=-1., max_val=1., reduction='sum'):
        super(SimilarityRegularizationLoss, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        assert reduction in ['sum', 'mean'], 'Invalid reduction value'
        self.reduction = reduction

    def forward(self, similarity):
        loss = torch.sum(torch.abs(torch.clamp(similarity - self.min_val, max=0.)))
        loss += torch.sum(torch.abs(torch.clamp(similarity - self.max_val, min=0.)))
        if self.reduction == 'mean':
            loss = loss / similarity.numel()
        return loss

    def __repr__(self, ):
        return '{}(min_val={}, max_val={})'.format(self.__class__.__name__, self.min_val, self.max_val)
