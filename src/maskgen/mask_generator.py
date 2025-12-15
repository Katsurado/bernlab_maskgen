"""
mask_generator.py

Inference wrapper for semantic segmentation model.
Handles large images via tiling or downsampling to enable CPU inference.
"""

from typing import Literal, Union
from pathlib import Path

import numpy as np
import torch
import math
import torch.nn.functional as F
import torch.optim.swa_utils as swa

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

    DEFAULT_CONFIG: dict = {
        'channels': 128,
        'layers': [1, 2, 4, 2],
        'stochastic_depth': 0.1,  # doesn't matter for inference, but needed for init
        "ema": 0.9999,
    }

    def __init__(
        self,
        checkpoint_path: str,
        device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        use_ema: bool = True,
        config: dict | None = None,  # override if needed
    ) -> None:
        """
        Args:
            checkpoint_path: Path to .pth checkpoint file.
            device: Target device. "auto" selects best available.
            use_ema: If True, load EMA weights (recommended).
            config: Model config. Uses DEFAULT_CONFIG if None.
        """
        self.config:dict = self.DEFAULT_CONFIG if config is None else config
        self.device:torch.device = self._resolve_device(device)
        self._load_checkpoint(checkpoint_path, use_ema)
        self.model.to(self.device)  # move weights to correct device
        self.model.eval()            # disable dropout, stochastic depth, etc.

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
        self, path: str, use_ema: bool
    ) -> None:
        """Load model architecture and weights from checkpoint."""
        config = self.config

        model = Net(self.config['channels'], self.config)
        ema_model = swa.AveragedModel(
            model, multi_avg_fn=swa.get_ema_multi_avg_fn(self.config['ema'])
        )

        model, ema_model, _, _, _, _ = load_model(model,ema_model, path=path)

        if use_ema:
            self.model = ema_model
        else:
            self.model = model


    def _resolve_device(
        self, device: str
    ) -> torch.device:
        """Map device string to torch.device, handling 'auto'."""
        return torch.device(set_device()) if device == 'auto' else torch.device(device)

    def _preprocess(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """
        Convert input to unnormalized NCHW tensor.

        Handles multiple input formats and prepares image for model inference.
        No normalization is applied - raw pixel values [0, 255] are cast to float32.

        Args:
            image: Input RGB image in one of the following formats:
                - str or Path: File path to image (loaded via PIL)
                - PIL.Image.Image: Already-loaded PIL image (converted to RGB if needed)
                - np.ndarray: HWC uint8 array with shape (H, W, 3), values in [0, 255]

        Returns:
            - tensor: torch.Tensor of shape (1, 3, H, W), dtype float32, 
                    values in [0, 255], on self.device
        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If image format is unrecognized or has wrong shape/channels

        Processing steps:
            1. Load/convert to PIL Image if needed
            2. Convert to RGB (handles RGBA, grayscale, etc.)
            3. Convert to numpy HWC uint8
            4. Cast to float32, permute to CHW, add batch dim
            5. Move to self.device
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            image = Image.open(path)

        if isinstance(image, Image.Image):
            image = image.convert("RGB")  # handles RGBA, L, P, etc.
            image = np.array(image)

        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Unrecognized image type: {type(image).__name__}. "
                f"Expected str, Path, PIL.Image, or np.ndarray."
            )

        if image.ndim != 3:
            raise ValueError(
                f"Expected 3D array (H, W, C), got shape {image.shape}"
            )

        if image.shape[2] != 3:
            raise ValueError(
                f"Expected 3 channels (RGB), got {image.shape[2]} channels"
            )

        tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW
        tensor = tensor.unsqueeze(0)  # add batch dim -> (1, 3, H, W)
        tensor = tensor.to(self.device)

        return tensor

    def _postprocess(
        self,
        logits: torch.Tensor,
        threshold: float,
        return_prob: bool,
    ) -> Union[Image.Image, np.ndarray]:
        """
        Convert raw model output to a usable mask.

        The model outputs "logits" - raw unbounded scores where:
            - Positive values → model thinks "foreground"
            - Negative values → model thinks "background"
            - Magnitude → confidence

        This function converts those logits into either a probability map
        or a binary mask, resized to match the original input image.

        Args:
            logits: Raw model output, shape (1, 1, H, W), unbounded floats.
                    May be different size than original if padding was applied.
            threshold: Cutoff for binary mask
                    Pixels with probability > threshold become 1 (white).
                    Lower threshold → more foreground, higher → less.
            return_prob: If True, return float32 probability map [0, 1].
                        If False, return binary PIL Image (mode "L").

        Returns:
            If return_prob=True:
                np.ndarray of shape (H, W), dtype float32, values in [0, 1]
            If return_prob=False:
                PIL.Image in mode "L" (grayscale), values 0 or 255

        Processing pipeline:
            1. Sigmoid: logits -> probabilities [0, 1]
            2. Squeeze: (1, 1, H, W) -> (H, W)
            3. Threshold (if binary): prob > threshold -> 1
            4. Convert: to numpy/PIL
        """
        probs = torch.sigmoid(logits)
        probs = probs[0, 0].cpu().numpy()

        if return_prob:
            return probs
        
        # apply a threashold
        mask = (probs >= threshold).astype(np.uint8)

        # convert 0/1 to 0/255
        mask = mask * 255
        return Image.fromarray(mask)


    def _infer_whole(
        self, tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Run single forward pass on entire image. 
        WARNING: for large images (the size you would expect from a reactor) this will need A LOT OF vRAM.
        In general, avoid whole image inference unless on NVIDIA A100 80G or larger gpus.

        Args:
            tensor: Preprocessed input, shape (1, 3, H, W). Must already be
                    padded to multiple of STRIDE_FACTOR.

        Returns:
            Logits tensor of shape (1, 1, H, W).
        """
        self._validate_inference_input(tensor)

        with torch.no_grad():
            output = self.model(tensor)
            
        return output


    def _infer_tiled(
        self, tensor: torch.Tensor,
        tile_size: int,
        overlap: int,
    ) -> torch.Tensor:
        """
        Tile-based inference with overlap-blending.

        For large images that don't fit in memory, this method splits the image
        into overlapping tiles, runs inference on each, and blends the results.
        This trades speed for memory efficiency.

        Args:
            tensor: Preprocessed input, shape (1, 3, H, W). Does NOT need to be
                    pre-padded; each tile is padded individually.
            tile_size: Size of each square tile in pixels. Must be divisible by
                    STRIDE_FACTOR (32). Larger tiles = faster but more memory.
                    Recommended: 512 for ~4GB VRAM, 256 for CPU.
            overlap: Overlap between adjacent tiles in pixels. Larger overlap
                    reduces seam artifacts but increases computation.
                    Recommended: 64-128 pixels.

        Returns:
            Logits tensor of shape (1, 1, H, W), same spatial size as input.

        Algorithm:
            1. Calculate tile grid positions with overlap
            2. For each tile position:
            a. Extract tile from input
            b. Pad tile to multiple of 32 (if needed)
            c. Run inference
            d. Unpad result
            e. Accumulate into output buffer with blending weights
            3. Normalize by total weight at each pixel

        Blending strategy:
            Uses linear ramp weights at tile edges. Each pixel's final value is
            a weighted average of all tiles that cover it:

            Tile A:  [1.0  1.0  1.0  0.75 0.5  0.25 0.0]  (fades out)
            Tile B:            [0.0  0.25 0.5  0.75 1.0  1.0  1.0]  (fades in)
                            └─── overlap region ───┘

            This eliminates hard seams between tiles.

        Memory usage:
            Only one tile is in GPU memory at a time. Output buffer is kept on
            CPU and assembled incrementally.

        Example:
            >>> tensor = torch.randn(1, 3, 4000, 6000)  # Large image
            >>> logits = self._infer_tiled(tensor, tile_size=512, overlap=64)
            >>> logits.shape
            torch.Size([1, 1, 4000, 6000])
        """
        ...

    def _infer_downsampled(
        self, tensor: torch.Tensor,
        max_dim: int,
    ) -> torch.Tensor:
        """
        Downsample-infer-upsample strategy for memory-constrained inference.

        Alternative to tiling: shrink the image to fit in memory, run inference,
        then upscale the result. Faster than tiling but may lose fine details.

        Args:
            tensor: Preprocessed input, shape (1, 3, H, W).
            max_dim: Maximum allowed dimension (height or width). Image is scaled
                    so its largest dimension equals max_dim, preserving aspect ratio.
                    Recommended: 1024-2048 depending on available memory.

        Returns:
            Logits tensor of shape (1, 1, H, W), upscaled to original input size.

        Trade-offs vs tiling:
            Pros:
                - Much faster (single forward pass)
                - No blending artifacts
            Cons:
                - Loses fine details at high downscale ratios
                - Small objects may disappear entirely

        Example:
            >>> tensor = torch.randn(1, 3, 4000, 6000)  # 24MP image
            >>> logits = self._infer_downsampled(tensor, max_dim=1024)
            # Internally: 4000x6000 -> 683x1024 -> infer -> 4000x6000
            >>> logits.shape
            torch.Size([1, 1, 4000, 6000])
        """
        ...

    @staticmethod
    def _pad_to_multiple(
        tensor: torch.Tensor, factor: int
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """
        Pad tensor so H and W are divisible by factor.

        The model requires input dimensions to be divisible by its total stride
        (32 for this architecture). This function pads the input to the next
        valid size, using zero-padding on the right and bottom edges only.

        Args:
            tensor: Input tensor of shape (N, C, H, W).
            factor: The multiple to pad to (typically 32 for this model).
                    H and W will be padded to ceil(H/factor)*factor.

        Returns:
            tuple containing:
                - padded: Tensor of shape (N, C, H', W') where H' and W' are
                        the smallest multiples of factor >= H and W.
                - pad_spec: Tuple (left, right, top, bottom) describing padding
                            applied. Used by _unpad to restore original size.
                            For this implementation, left=top=0 always.

        Example:
            >>> tensor = torch.randn(1, 3, 1000, 1500)  # not divisible by 32
            >>> padded, pad_spec = _pad_to_multiple(tensor, 32)
            >>> padded.shape
            torch.Size([1, 3, 1024, 1504])
            >>> pad_spec
            (0, 4, 0, 24)  # (left, right, top, bottom)

        Note:
            Padding is applied to right/bottom only (not symmetric) because:
            1. Simpler to compute and undo
            2. For fully-convolutional models, padding location doesn't affect
            predictions in the original (non-padded) region
        """
        _, _, h, w = tensor.shape
        # Calculate target dimensions (next multiple of factor)
        h_new = math.ceil(h / factor) * factor
        w_new = math.ceil(w / factor) * factor

        pad_right = w_new - w
        pad_bottom = h_new - h

        # F.pad format: (left, right, top, bottom) — last dim first!
        pad_spec = (0, pad_right, 0, pad_bottom)

        padded = F.pad(tensor, pad_spec, mode='constant', value=0)
        return padded, pad_spec


    @staticmethod
    def _unpad(
        tensor: torch.Tensor, pad_spec: tuple[int, int, int, int]
    ) -> torch.Tensor:
        """
        Remove padding applied by _pad_to_multiple.

        Reverses the padding operation by slicing off padded edges.
        Used after model inference to restore output to original dimensions.

        Args:
            tensor: Padded tensor of shape (N, C, H', W') from model output.
            pad_spec: Tuple (left, right, top, bottom) from _pad_to_multiple.
                    Describes how many pixels were added to each edge.

        Returns:
            Tensor of shape (N, C, H, W) with padding removed,
            where H = H' - top - bottom, W = W' - left - right.

        Example:
            >>> padded = torch.randn(1, 1, 1024, 1504)
            >>> pad_spec = (0, 4, 0, 24)
            >>> unpadded = _unpad(padded, pad_spec)
            >>> unpadded.shape
            torch.Size([1, 1, 1000, 1500])

        Note:
            When padding is 0 for an edge, we use None in the slice to avoid
            the -0 problem:
                tensor[:, :, 0:-0, :]  # Wrong! -0 == 0, returns empty tensor
                tensor[:, :, 0:None, :]  # Correct! None means "to the end"
        """
        
        left, right, top, bottom = pad_spec
        
        tensor = tensor[
            :,
            :,
            top : -bottom if bottom else None,
            left : -right if right else None,
        ]

        return tensor
    
    def _validate_inference_input(
        self, tensor: torch.Tensor
    ) -> None:
        """
        Sanity checks before running inference.

        Validates that the input tensor and model are properly configured
        for inference. Raises descriptive errors for common mistakes.

        Args:
            tensor: Preprocessed input tensor to validate.

        Raises:
            RuntimeError: If tensor is not on self.device.
            RuntimeError: If model is not on self.device.
            RuntimeError: If model is in training mode.
            ValueError: If tensor has wrong shape (not 4D).
            ValueError: If tensor has wrong number of channels (not 3).
        """

        # Tensor device check
        if tensor.device != self.device:
            raise RuntimeError(
                f"Tensor is on {tensor.device}, expected {self.device}."
            )

        # Model device check
        model_device = next(self.model.parameters()).device
        if model_device != self.device:
            raise RuntimeError(
                f"Model is on {model_device}, expected {self.device}."
            )

        # Model mode check
        if self.model.training:
            raise RuntimeError(
                "Model is in training mode. Call model.eval() first."
            )

        # Shape check
        if tensor.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (N, C, H, W), got {tensor.dim()}D."
            )

        # Channel check
        if tensor.shape[1] != 3:
            raise ValueError(
                f"Expected 3 input channels, got {tensor.shape[1]}."
            )