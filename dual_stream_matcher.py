"""Dual-stream feature fusion and matching utilities.

This module implements the Chapter-1 style dual-stream matcher:
- Project ALIKED (128D) and DINOv3 (384D) to a common 256D space.
- Channel correction with a shared MLP and sigmoid gating.
- Texture/semantic cosine similarity matrix computation and fusion.
- Mutual nearest-neighbor extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FeatureFusionLayer(nn.Module):
    """Project and cross-correct texture/semantic features.

    Args:
        tex_in_dim: Input dim for ALIKED texture descriptors.
        sem_in_dim: Input dim for DINOv3 semantic descriptors.
        hidden_dim: Shared embedding dimension after projection.
    """

    def __init__(self, tex_in_dim: int = 128, sem_in_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.tex_proj = nn.Linear(tex_in_dim, hidden_dim)
        self.sem_proj = nn.Linear(sem_in_dim, hidden_dim)

        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.gate = nn.Sigmoid()

    def forward(self, tex_feat: Tensor, sem_feat: Tensor) -> Tuple[Tensor, Tensor]:
        """Return corrected texture/semantic descriptors.

        Args:
            tex_feat: (B, N, 128) texture descriptors.
            sem_feat: (B, N, 384) semantic descriptors.

        Returns:
            D_t_corr: (B, N, 256)
            D_s_corr: (B, N, 256)
        """
        d_t = self.tex_proj(tex_feat)
        d_s = self.sem_proj(sem_feat)

        fused = torch.cat([d_t, d_s], dim=-1)
        weights = self.gate(self.channel_mlp(fused))
        w_t, w_s = torch.chunk(weights, chunks=2, dim=-1)

        d_t_corr = d_t + w_t * d_s
        d_s_corr = d_s + w_s * d_t
        return d_t_corr, d_s_corr


@dataclass
class MatchResult:
    """Container for pairwise matching outputs."""

    score_matrix: Tensor
    matches: Tensor


class DualStreamMatcher(nn.Module):
    """Compute fused similarity matrices and extract MNN correspondences."""

    def __init__(self):
        super().__init__()
        self.fusion = FeatureFusionLayer()

    @staticmethod
    def _cosine_matrix(a: Tensor, b: Tensor) -> Tensor:
        """Pairwise cosine similarity for batched token descriptors.

        Args:
            a: (B, N1, C)
            b: (B, N2, C)
        """
        a_n = F.normalize(a, p=2, dim=-1)
        b_n = F.normalize(b, p=2, dim=-1)
        return torch.matmul(a_n, b_n.transpose(-1, -2))

    def compute_match_matrix(
        self,
        tex_a: Tensor,
        sem_a: Tensor,
        tex_b: Tensor,
        sem_b: Tensor,
    ) -> Tensor:
        """Compute C_M = C_t * C_s for two images."""
        d_t_a, d_s_a = self.fusion(tex_a, sem_a)
        d_t_b, d_s_b = self.fusion(tex_b, sem_b)

        c_t = self._cosine_matrix(d_t_a, d_t_b)
        c_s = self._cosine_matrix(d_s_a, d_s_b)
        return c_t * c_s

    @staticmethod
    def mutual_nearest_neighbors(score_matrix: Tensor, min_score: float = -1.0) -> Tensor:
        """Extract mutual nearest neighbors from a similarity matrix.

        Args:
            score_matrix: (B, N1, N2)
            min_score: Optional similarity threshold.

        Returns:
            Tensor of shape (K, 4): [batch_idx, idx_a, idx_b, score].
        """
        device = score_matrix.device
        bsz, n1, _ = score_matrix.shape

        best_b_for_a = torch.argmax(score_matrix, dim=-1)  # (B, N1)
        best_a_for_b = torch.argmax(score_matrix, dim=-2)  # (B, N2)

        matches = []
        for b in range(bsz):
            a_idx = torch.arange(n1, device=device)
            b_idx = best_b_for_a[b]
            mutual = best_a_for_b[b, b_idx] == a_idx
            if min_score > -1.0:
                mutual = mutual & (score_matrix[b, a_idx, b_idx] >= min_score)
            valid_a = a_idx[mutual]
            valid_b = b_idx[mutual]
            if valid_a.numel() == 0:
                continue
            scores = score_matrix[b, valid_a, valid_b]
            batch_col = torch.full_like(valid_a, b)
            matches.append(torch.stack([batch_col.float(), valid_a.float(), valid_b.float(), scores], dim=-1))

        if not matches:
            return torch.empty((0, 4), device=device)
        return torch.cat(matches, dim=0)
