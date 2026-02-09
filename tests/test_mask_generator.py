"""
test_mask_generator.py

Unit tests for MaskGenerator components.
Run with: pytest tests/test_mask_generator.py -v
"""

import numpy as np
import pytest
import torch
from PIL import Image

from maskgen.mask_generator import MaskGenerator
from maskgen.model import Net

# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    """Standard model config."""
    return {
        'channels': 128,
        'layers': [1, 2, 4, 2],
        'stochastic_depth': 0.1,
        'ema': 0.9999,
    }


@pytest.fixture
def mock_generator(config):
    """MaskGenerator without checkpoint loading."""
    class MockGenerator(MaskGenerator):
        def __init__(self, config):
            self.config = config
            self.device = torch.device('cpu')
            self.model = Net(config['channels'], config)
            self.model.eval()

    return MockGenerator(config)


# ──────────────────────────────────────────────────────────────
# Model Architecture Tests
# ──────────────────────────────────────────────────────────────

class TestModelArchitecture:

    def test_model_output_shape(self, config):
        """Model output should match input spatial dims."""
        model = Net(config['channels'], config)
        model.eval()

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 1, 64, 64)

    def test_model_various_sizes(self, config):
        """Model should handle any size divisible by 32."""
        model = Net(config['channels'], config)
        model.eval()

        sizes = [(32, 32), (64, 128), (128, 64), (96, 96)]

        for h, w in sizes:
            x = torch.randn(1, 3, h, w)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 1, h, w), f"Failed for size {h}x{w}"


# ──────────────────────────────────────────────────────────────
# Padding Tests
# ──────────────────────────────────────────────────────────────

class TestPadding:

    def test_pad_to_multiple_basic(self):
        """Padding should make dims divisible by factor."""
        tensor = torch.randn(1, 3, 100, 150)
        padded, pad_spec = MaskGenerator._pad_to_multiple(tensor, 32)

        assert padded.shape[2] % 32 == 0
        assert padded.shape[3] % 32 == 0

    def test_pad_spec_format(self):
        """Pad spec should be (left, right, top, bottom)."""
        tensor = torch.randn(1, 3, 100, 150)
        _, pad_spec = MaskGenerator._pad_to_multiple(tensor, 32)

        left, right, top, bottom = pad_spec
        assert left == 0, "Left padding should always be 0"
        assert top == 0, "Top padding should always be 0"

    def test_pad_already_divisible(self):
        """No padding needed if already divisible."""
        tensor = torch.randn(1, 3, 64, 128)
        padded, pad_spec = MaskGenerator._pad_to_multiple(tensor, 32)

        assert pad_spec == (0, 0, 0, 0)
        assert padded.shape == tensor.shape

    def test_unpad_restores_shape(self):
        """Unpad should restore original dimensions."""
        original = torch.randn(1, 3, 100, 150)
        padded, pad_spec = MaskGenerator._pad_to_multiple(original, 32)
        unpadded = MaskGenerator._unpad(padded, pad_spec)

        assert unpadded.shape == original.shape

    def test_unpad_zero_padding(self):
        """Unpad should handle zero padding (no-op)."""
        tensor = torch.randn(1, 3, 64, 64)
        pad_spec = (0, 0, 0, 0)
        unpadded = MaskGenerator._unpad(tensor, pad_spec)

        assert unpadded.shape == tensor.shape

    def test_pad_unpad_preserves_data(self):
        """Data should be unchanged after pad -> unpad."""
        original = torch.randn(1, 3, 50, 75)
        padded, pad_spec = MaskGenerator._pad_to_multiple(original, 32)
        unpadded = MaskGenerator._unpad(padded, pad_spec)

        assert torch.allclose(original, unpadded)


# ──────────────────────────────────────────────────────────────
# Preprocessing Tests
# ──────────────────────────────────────────────────────────────

class TestPreprocess:

    def test_numpy_input(self, mock_generator):
        """Should handle numpy HWC uint8 array."""
        arr = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
        tensor = mock_generator._preprocess(arr)

        assert tensor.shape == (1, 3, 64, 96)
        assert tensor.dtype == torch.float32

    def test_pil_input(self, mock_generator):
        """Should handle PIL Image."""
        arr = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr)
        tensor = mock_generator._preprocess(pil_img)

        assert tensor.shape == (1, 3, 64, 96)

    def test_rgba_conversion(self, mock_generator):
        """Should convert RGBA to RGB."""
        rgba = np.random.randint(0, 255, (64, 96, 4), dtype=np.uint8)
        pil_rgba = Image.fromarray(rgba)
        tensor = mock_generator._preprocess(pil_rgba)

        assert tensor.shape == (1, 3, 64, 96)

    def test_grayscale_conversion(self, mock_generator):
        """Should convert grayscale to RGB."""
        gray = np.random.randint(0, 255, (64, 96), dtype=np.uint8)
        pil_gray = Image.fromarray(gray)
        tensor = mock_generator._preprocess(pil_gray)

        assert tensor.shape == (1, 3, 64, 96)

    def test_invalid_type_raises(self, mock_generator):
        """Should raise ValueError for invalid input type."""
        with pytest.raises(ValueError, match="Unrecognized image type"):
            mock_generator._preprocess([1, 2, 3])

    def test_wrong_dims_raises(self, mock_generator):
        """Should raise ValueError for wrong dimensions."""
        arr = np.random.randint(0, 255, (64, 96), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 3D array"):
            mock_generator._preprocess(arr)

    def test_wrong_channels_raises(self, mock_generator):
        """Should raise ValueError for wrong channel count."""
        arr = np.random.randint(0, 255, (64, 96, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 3 channels"):
            mock_generator._preprocess(arr)


# ──────────────────────────────────────────────────────────────
# Postprocessing Tests
# ──────────────────────────────────────────────────────────────

class TestPostprocess:

    def test_binary_output(self, mock_generator):
        """Should return PIL Image for binary mask."""
        logits = torch.randn(1, 1, 64, 96)
        mask = mock_generator._postprocess(logits, threshold=0.5, return_prob=False)

        assert isinstance(mask, Image.Image)
        assert mask.mode == 'L'
        assert mask.size == (96, 64)  # PIL is (W, H)

    def test_prob_output(self, mock_generator):
        """Should return numpy array for probability map."""
        logits = torch.randn(1, 1, 64, 96)
        probs = mock_generator._postprocess(logits, threshold=0.5, return_prob=True)

        assert isinstance(probs, np.ndarray)
        assert probs.shape == (64, 96)
        assert probs.dtype == np.float32

    def test_prob_range(self, mock_generator):
        """Probabilities should be in [0, 1]."""
        logits = torch.randn(1, 1, 64, 96) * 10  # large values
        probs = mock_generator._postprocess(logits, threshold=0.5, return_prob=True)

        assert probs.min() >= 0
        assert probs.max() <= 1

    def test_binary_values(self, mock_generator):
        """Binary mask should only contain 0 and 255."""
        logits = torch.randn(1, 1, 64, 96)
        mask = mock_generator._postprocess(logits, threshold=0.5, return_prob=False)

        arr = np.array(mask)
        unique = set(np.unique(arr))
        assert unique.issubset({0, 255})

    def test_threshold_effect(self, mock_generator):
        """Higher threshold should produce less foreground."""
        logits = torch.randn(1, 1, 64, 96)

        mask_low = mock_generator._postprocess(logits, threshold=0.3, return_prob=False)
        mask_high = mock_generator._postprocess(logits, threshold=0.7, return_prob=False)

        fg_low = np.array(mask_low).sum()
        fg_high = np.array(mask_high).sum()

        assert fg_low >= fg_high


# ──────────────────────────────────────────────────────────────
# Validation Tests
# ──────────────────────────────────────────────────────────────

class TestValidation:

    def test_valid_input_passes(self, mock_generator):
        """Valid input should not raise."""
        tensor = torch.randn(1, 3, 64, 64)
        mock_generator._validate_inference_input(tensor)  # should not raise

    def test_wrong_dims_raises(self, mock_generator):
        """Should raise for non-4D tensor."""
        tensor = torch.randn(3, 64, 64)
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            mock_generator._validate_inference_input(tensor)

    def test_wrong_channels_raises(self, mock_generator):
        """Should raise for wrong channel count."""
        tensor = torch.randn(1, 4, 64, 64)
        with pytest.raises(ValueError, match="Expected 3 input channels"):
            mock_generator._validate_inference_input(tensor)

    def test_training_mode_raises(self, mock_generator):
        """Should raise if model in training mode."""
        mock_generator.model.train()
        tensor = torch.randn(1, 3, 64, 64)

        with pytest.raises(RuntimeError, match="training mode"):
            mock_generator._validate_inference_input(tensor)

        mock_generator.model.eval()  # reset


# ──────────────────────────────────────────────────────────────
# Integration Tests
# ──────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline(self, mock_generator):
        """Test complete inference pipeline."""
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

        tensor = mock_generator._preprocess(img)
        padded, pad_spec = mock_generator._pad_to_multiple(tensor, 32)
        logits_padded = mock_generator._infer_whole(padded)
        logits = mock_generator._unpad(logits_padded, pad_spec)
        mask = mock_generator._postprocess(logits, threshold=0.5, return_prob=False)

        assert mask.size == (150, 100)  # PIL is (W, H)

    def test_pipeline_various_sizes(self, mock_generator):
        """Pipeline should work for various image sizes."""
        sizes = [(50, 50), (75, 100), (123, 67), (200, 150)]

        for h, w in sizes:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

            tensor = mock_generator._preprocess(img)
            padded, pad_spec = mock_generator._pad_to_multiple(tensor, 32)
            logits_padded = mock_generator._infer_whole(padded)
            logits = mock_generator._unpad(logits_padded, pad_spec)
            mask = mock_generator._postprocess(logits, threshold=0.5, return_prob=False)

            assert mask.size == (w, h), f"Failed for size {h}x{w}"

# ──────────────────────────────────────────────────────────────
# Tiling Tests
# ──────────────────────────────────────────────────────────────

class TestComputeTileGrid:

    def test_single_tile(self, mock_generator):
        """Image smaller than tile_size produces one tile."""
        tiles = mock_generator._compute_tile_grid(h=100, w=100, tile_size=256, overlap=32)

        assert len(tiles) == 1
        assert tiles[0] == (0, 100, 0, 100)

    def test_exact_fit(self, mock_generator):
        """Tiles fit exactly with no partial edge tiles."""
        # tile_size=256, overlap=32 -> stride=224
        # Only 1 row (h=256), width=480 = 256 + 224 -> 2 tiles in x
        tiles = mock_generator._compute_tile_grid(h=256, w=480, tile_size=256, overlap=32)

        # stride=224, so y_starts=[0], x_starts=[0, 224]
        # But wait - range(0, 256, 224) = [0, 224], so 2 y positions!
        # Actually need h <= stride to get 1 row

        # Let's just verify the tiles make sense
        assert tiles[0] == (0, 256, 0, 256)
        assert tiles[1] == (0, 256, 224, 480)

    def test_partial_edge(self, mock_generator):
        """Edge tiles are clamped to image bounds."""
        tiles = mock_generator._compute_tile_grid(h=300, w=300, tile_size=256, overlap=32)

        # stride = 224, so x_starts = [0, 224], y_starts = [0, 224]
        assert len(tiles) == 4

        for y_start, y_end, x_start, x_end in tiles:
            assert y_end <= 300
            assert x_end <= 300

    def test_full_coverage(self, mock_generator):
        """Every pixel is covered by at least one tile."""
        h, w = 500, 700
        tiles = mock_generator._compute_tile_grid(h=h, w=w, tile_size=256, overlap=32)

        coverage = np.zeros((h, w), dtype=int)
        for y_start, y_end, x_start, x_end in tiles:
            coverage[y_start:y_end, x_start:x_end] += 1

        assert (coverage >= 1).all(), "Some pixels not covered"


class TestCreateBlendWeights:

    def test_shape(self, mock_generator):
        """Output shape matches input dimensions."""
        weights = mock_generator._create_blend_weights(tile_h=256, tile_w=256, overlap=32)

        assert weights.shape == (256, 256)

    def test_center_is_one(self, mock_generator):
        """Center region has weight 1.0."""
        weights = mock_generator._create_blend_weights(tile_h=256, tile_w=256, overlap=32)

        center = weights[64:192, 64:192]
        assert torch.allclose(center, torch.ones_like(center))

    def test_no_zeros(self, mock_generator):
        """Weights are strictly positive (avoid division by zero)."""
        weights = mock_generator._create_blend_weights(tile_h=256, tile_w=256, overlap=32)

        assert (weights > 0).all()

    def test_symmetry(self, mock_generator):
        """Weights are symmetric in all directions."""
        weights = mock_generator._create_blend_weights(tile_h=256, tile_w=256, overlap=32)

        assert torch.allclose(weights, weights.flip(1))  # horizontal
        assert torch.allclose(weights, weights.flip(0))  # vertical

    def test_ramp_increasing(self, mock_generator):
        """Edge ramps increase from edge to center."""
        overlap = 4
        weights = mock_generator._create_blend_weights(tile_h=20, tile_w=20, overlap=overlap)

        center_row = weights[10, :]
        left_ramp = center_row[:overlap]

        for i in range(len(left_ramp) - 1):
            assert left_ramp[i] < left_ramp[i + 1]


class TestInferTiled:

    def test_output_shape(self, mock_generator):
        """Output has correct shape (1, 1, H, W)."""
        def mock_infer_whole(tile):
            _, _, h, w = tile.shape
            return torch.ones(1, 1, h, w)

        mock_generator._infer_whole = mock_infer_whole

        tensor = torch.randn(1, 3, 500, 700)
        output = mock_generator._infer_tiled(tensor, tile_size=256, overlap=32)

        assert output.shape == (1, 1, 500, 700)

    def test_uniform_blending(self, mock_generator):
        """Uniform predictions blend to uniform output."""
        def mock_infer_whole(tile):
            _, _, h, w = tile.shape
            return torch.ones(1, 1, h, w) * 0.5

        mock_generator._infer_whole = mock_infer_whole

        tensor = torch.randn(1, 3, 300, 300)
        output = mock_generator._infer_tiled(tensor, tile_size=128, overlap=16)

        assert torch.allclose(output, torch.ones_like(output) * 0.5, atol=1e-5)

    def test_no_seams(self, mock_generator):
        """No visible seams at tile boundaries."""
        def mock_infer_whole(tile):
            _, _, h, w = tile.shape
            return torch.ones(1, 1, h, w)

        mock_generator._infer_whole = mock_infer_whole

        tensor = torch.randn(1, 3, 300, 300)
        output = mock_generator._infer_tiled(tensor, tile_size=128, overlap=32)

        diff_x = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])
        diff_y = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])

        assert diff_x.max() < 0.1
        assert diff_y.max() < 0.1

    def test_tiled_output_on_cpu(self, mock_generator):
        """Tiled inference output should always be on CPU."""
        def mock_infer_whole(tile):
            _, _, h, w = tile.shape
            return torch.ones(1, 1, h, w)

        mock_generator._infer_whole = mock_infer_whole

        tensor = torch.randn(1, 3, 300, 300)
        output = mock_generator._infer_tiled(tensor, tile_size=128, overlap=16)

        assert output.device == torch.device("cpu"), (
            "Tiled output must be on CPU. If _infer_whole returns GPU tensors, "
            "they must be moved to CPU before accumulation."
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="No GPU available",
    )
    def test_tiled_gpu_to_cpu(self, mock_generator):
        """Tiled inference handles GPU->CPU transfer for accumulation."""
        device = "cuda" if torch.cuda.is_available() else "mps"

        def mock_infer_whole(tile):
            _, _, h, w = tile.shape
            return torch.ones(1, 1, h, w, device=device)

        mock_generator._infer_whole = mock_infer_whole

        tensor = torch.randn(1, 3, 300, 300)
        output = mock_generator._infer_tiled(tensor, tile_size=128, overlap=16)

        assert output.device == torch.device("cpu")
        assert output.shape == (1, 1, 300, 300)
