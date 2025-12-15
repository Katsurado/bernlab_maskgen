"""
mask_generator.py

Inference wrapper for semantic segmentation model.
Handles large images via tiling or downsampling to enable CPU inference.
"""

from typing import Literal, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Internal imports
from .model import Net
from .utils import set_device, load_model


class MaskGenerator:
    """
    High-level interface for generating binary masks from images.
    
    Abstracts away model loading, device management, and memory-efficient
    inference strategies (tiling or downsampling) for large images.
    
    Example:
        >>> gen = MaskGenerator("checkpoint.pth")
        >>> mask = gen.generate("photo.jpg")
        >>> mask.save("mask.png")
    """

    STRIDE_FACTOR: int = 32  # model's total spatial reduction factor

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        use_ema: bool = True,
    ) -> None:
        """
        Load model weights and prepare for inference.

        Args:
            checkpoint_path: Path to .pth checkpoint file.
            device: Target device. "auto" selects best available.
            use_ema: If True, load EMA weights (recommended for inference).
        """
        ...

    def generate(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        strategy: Literal["tile", "downsample"] = "tile",
        tile_size: int = 512,
        tile_overlap: int = 64,
        threshold: float = 0.5,
        return_prob: bool = False,
    ) -> Union[Image.Image, np.ndarray]:
        """
        Generate a binary mask for the input image.

        Args:
            image: Input RGB image (path, PIL Image, or HWC uint8 array).
            strategy: 
                "tile" - process in overlapping patches, blend results.
                "downsample" - resize to fit memory, then upscale mask.
            tile_size: Patch size for tiling (must be divisible by 32).
            tile_overlap: Overlap between tiles for blending.
            threshold: Probability cutoff for binary mask.
            return_prob: If True, return float32 probability map instead of binary.

        Returns:
            Binary mask as PIL Image (mode "L"), or float32 ndarray if return_prob=True.
        """
        ...

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _load_checkpoint(
        self, path: Path, use_ema: bool
    ) -> None:
        """Load model architecture and weights from checkpoint."""
        ...

    def _resolve_device(
        self, device: str
    ) -> torch.device:
        """Map device string to torch.device, handling 'auto'."""
        ...

    def _preprocess(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Convert input to normalized NCHW tensor.
        
        Returns:
            (tensor, original_size) where original_size = (H, W).
        """
        ...

    def _postprocess(
        self, logits: torch.Tensor, 
        original_size: tuple[int, int],
        threshold: float,
        return_prob: bool,
    ) -> Union[Image.Image, np.ndarray]:
        """Sigmoid → resize → threshold → PIL/numpy conversion."""
        ...

    def _infer_whole(
        self, tensor: torch.Tensor
    ) -> torch.Tensor:
        """Run single forward pass (for small images or downsampled)."""
        ...

    def _infer_tiled(
        self, tensor: torch.Tensor,
        tile_size: int,
        overlap: int,
    ) -> torch.Tensor:
        """
        Tile-based inference with overlap-blending.
        
        Uses linear ramp weights at tile boundaries for smooth stitching.
        """
        ...

    def _infer_downsampled(
        self, tensor: torch.Tensor,
        max_dim: int,
    ) -> torch.Tensor:
        """Downsample → infer → upsample strategy."""
        ...

    @staticmethod
    def _pad_to_multiple(
        tensor: torch.Tensor, factor: int
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """Pad tensor so H and W are divisible by factor. Returns (padded, pad_spec)."""
        ...

    @staticmethod
    def _unpad(
        tensor: torch.Tensor, pad_spec: tuple[int, int, int, int]
    ) -> torch.Tensor:
        """Remove padding applied by _pad_to_multiple."""
        ...