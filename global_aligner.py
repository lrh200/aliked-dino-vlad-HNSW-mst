"""Global image alignment with affine init + perspective-penalized homography."""

from __future__ import annotations

from collections import deque
from typing import Dict, Hashable, Tuple

import cv2
import networkx as nx
import numpy as np
from scipy.optimize import least_squares


class GlobalAligner:
    def __init__(self, lambda_perspective: float = 1e-2):
        self.lambda_perspective = lambda_perspective

    @staticmethod
    def _to_homogeneous(points: np.ndarray) -> np.ndarray:
        ones = np.ones((points.shape[0], 1), dtype=points.dtype)
        return np.hstack([points, ones])

    def _residual(self, h_flat: np.ndarray, parent_pts: np.ndarray, child_pts: np.ndarray) -> np.ndarray:
        h = h_flat.reshape(3, 3)
        child_h = self._to_homogeneous(child_pts)
        warped = (h @ child_h.T).T
        warped = warped[:, :2] / warped[:, 2:3]

        reproj = (parent_pts - warped).reshape(-1)
        perspective_penalty = np.sqrt(self.lambda_perspective) * np.array([h[2, 0], h[2, 1]])
        return np.concatenate([reproj, perspective_penalty])

    def _optimize_homography(self, parent_pts: np.ndarray, child_pts: np.ndarray) -> np.ndarray:
        affine, _ = cv2.estimateAffine2D(child_pts, parent_pts)
        if affine is None:
            h0 = np.eye(3, dtype=np.float64)
        else:
            h0 = np.eye(3, dtype=np.float64)
            h0[:2, :3] = affine

        result = least_squares(self._residual, h0.reshape(-1), args=(parent_pts, child_pts), method="lm")
        h_opt = result.x.reshape(3, 3)
        if np.isclose(h_opt[2, 2], 0.0):
            h_opt[2, 2] = 1.0
        return h_opt / h_opt[2, 2]

    def compute_global_transforms(
        self,
        spanning_tree: nx.DiGraph,
        match_points: Dict[Tuple[Hashable, Hashable], Tuple[np.ndarray, np.ndarray]],
        root: Hashable,
    ) -> Dict[Hashable, np.ndarray]:
        """Compute absolute transforms T_global for each node relative to root."""
        if root not in spanning_tree.nodes:
            raise ValueError("Root is not in spanning tree")

        t_global: Dict[Hashable, np.ndarray] = {root: np.eye(3, dtype=np.float64)}
        q = deque([root])

        while q:
            parent = q.popleft()
            for child in spanning_tree.successors(parent):
                key = (parent, child)
                if key not in match_points:
                    reverse_key = (child, parent)
                    if reverse_key not in match_points:
                        raise KeyError(f"No match points found for edge ({parent}, {child})")
                    child_pts, parent_pts = match_points[reverse_key]
                else:
                    parent_pts, child_pts = match_points[key]

                h_pc = self._optimize_homography(parent_pts, child_pts)
                t_global[child] = t_global[parent] @ h_pc
                q.append(child)

        return t_global
