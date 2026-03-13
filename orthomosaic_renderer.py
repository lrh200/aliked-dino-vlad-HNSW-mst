"""Orthomosaic rendering with warping and multi-band blending."""

from __future__ import annotations

from typing import Dict, Hashable, List, Tuple

import cv2
import numpy as np


class OrthomosaicRenderer:
    def _compute_canvas(
        self,
        images: Dict[Hashable, np.ndarray],
        global_transforms: Dict[Hashable, np.ndarray],
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        all_corners = []
        for key, img in images.items():
            h, w = img.shape[:2]
            corners = np.array([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(corners, global_transforms[key]).reshape(-1, 2)
            all_corners.append(warped)

        all_corners = np.vstack(all_corners)
        min_xy = np.floor(all_corners.min(axis=0)).astype(int)
        max_xy = np.ceil(all_corners.max(axis=0)).astype(int)

        width = int(max_xy[0] - min_xy[0] + 1)
        height = int(max_xy[1] - min_xy[1] + 1)

        translation = np.array(
            [[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        return (width, height), translation

    def render(
        self,
        images: Dict[Hashable, np.ndarray],
        global_transforms: Dict[Hashable, np.ndarray],
        num_bands: int = 5,
    ) -> np.ndarray:
        canvas_size, translation = self._compute_canvas(images, global_transforms)

        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands(num_bands)
        blender.prepare((0, 0, canvas_size[0], canvas_size[1]))

        for key, img in images.items():
            h, w = img.shape[:2]
            warp_h = translation @ global_transforms[key]
            warped_img = cv2.warpPerspective(img, warp_h, canvas_size)

            mask = np.ones((h, w), dtype=np.uint8) * 255
            warped_mask = cv2.warpPerspective(mask, warp_h, canvas_size)

            blender.feed(warped_img.astype(np.int16), warped_mask, (0, 0))

        result, result_mask = blender.blend(None, None)
        if result is None:
            raise RuntimeError("Blending failed; no output generated.")

        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        return result
