"""
test_mask_generator.py

Unit tests for MaskGenerator components.
Run with: pytest tests/test_mask_generator.py -v
"""

import numpy as np
import torch
from PIL import Image
import pytest
import sys

sys.path.insert(0, 'src')  # so it can find maskgen

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