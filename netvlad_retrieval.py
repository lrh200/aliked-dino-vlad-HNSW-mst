"""NetVLAD aggregation and ranked-list style retrieval losses."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class NetVLADLayer(nn.Module):
    """NetVLAD over token descriptors.

    Input:  (B, N, 512)
    Output: (B, K*512) where K=64 by default.
    """

    def __init__(self, dim: int = 512, num_clusters: int = 64):
        super().__init__()
        self.dim = dim
        self.num_clusters = num_clusters

        self.clusters = nn.Parameter(torch.randn(num_clusters, dim))
        self.assignment = nn.Conv1d(dim, num_clusters, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        bsz, n, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected descriptor dim {self.dim}, got {dim}")

        x_t = x.transpose(1, 2)  # (B, D, N)
        soft_assign = F.softmax(self.assignment(x_t), dim=1)  # (B, K, N)

        x_expand = x.unsqueeze(1)  # (B, 1, N, D)
        c_expand = self.clusters.unsqueeze(0).unsqueeze(2)  # (1, K, 1, D)
        residual = x_expand - c_expand  # (B, K, N, D)

        weighted = residual * soft_assign.unsqueeze(-1)
        vlad = weighted.sum(dim=2)  # (B, K, D)

        vlad = F.normalize(vlad, p=2, dim=-1)  # intra-normalization
        vlad = vlad.reshape(bsz, self.num_clusters * self.dim)
        vlad = F.normalize(vlad, p=2, dim=-1)  # global normalization
        return vlad


class RankedListLoss(nn.Module):
    """Ranked-list loss with positive/negative margin constraints.

    L_m = sum_hinge(d_neg < alpha - m) + sum_hinge(d_pos > alpha - m)
    plus a positive ranking term encouraging harder-ranked positives to be farther.
    """

    def __init__(self, alpha: float = 1.0, margin: float = 0.2, rank_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.rank_weight = rank_weight

    def forward(
        self,
        query: Tensor,
        positives: Tensor,
        negatives: Tensor,
        positive_order: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute ranked-list loss.

        Args:
            query: (B, D)
            positives: (B, P, D)
            negatives: (B, N, D)
            positive_order: optional (B, P) relevance rank (smaller is better).
        """
        q = F.normalize(query, p=2, dim=-1)
        p = F.normalize(positives, p=2, dim=-1)
        n = F.normalize(negatives, p=2, dim=-1)

        d_pos = torch.cdist(q.unsqueeze(1), p).squeeze(1)  # (B, P)
        d_neg = torch.cdist(q.unsqueeze(1), n).squeeze(1)  # (B, N)

        thresh = self.alpha - self.margin
        neg_loss = F.relu(thresh - d_neg).mean()
        pos_loss = F.relu(d_pos - thresh).mean()

        rank_loss = torch.tensor(0.0, device=query.device)
        if d_pos.shape[1] > 1:
            if positive_order is None:
                # Default: current order means from easier(nearer) to harder(farther)
                positive_order = torch.arange(d_pos.shape[1], device=query.device).unsqueeze(0).expand_as(d_pos)
            # pairwise ranking: if rank_i < rank_j then d_i <= d_j
            for i in range(d_pos.shape[1] - 1):
                for j in range(i + 1, d_pos.shape[1]):
                    rel_i = positive_order[:, i]
                    rel_j = positive_order[:, j]
                    must_be_closer = rel_i < rel_j
                    if must_be_closer.any():
                        diff = d_pos[must_be_closer, i] - d_pos[must_be_closer, j]
                        rank_loss = rank_loss + F.relu(diff).mean()

        return neg_loss + pos_loss + self.rank_weight * rank_loss
