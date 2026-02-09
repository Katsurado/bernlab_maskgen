# CLAUDE.md

## Overview

maskgen generates binary segmentation masks for microscopy images. It is developed by Kangping Liu and the Bernhard lab at CMU. The primary users are lab members who need automated mask generation as a preprocessing step before running `Experiment`-based image analysis pipelines.

## Repo Structure

```
src/maskgen/
  __init__.py          # Public API: MaskGenerator, download_weights
  mask_generator.py    # Inference wrapper (strategies: whole, tile, downsample)
  model.py             # ConvNeXt V2 U-Net architecture (Net class)
  data.py              # ImageData dataset for training
  train.py             # Importable training module (no argparse)
  utils.py             # Device detection, checkpoint save/load, param groups
  hub.py               # Download weights from GitHub Releases
tests/
  test_mask_generator.py  # Unit + integration tests
notebooks/
  train_colab.ipynb       # Colab training notebook
  demo.ipynb              # Inference demo with all 3 strategies
```

## Commands

Install (editable, all extras):
```bash
pip install -e ".[dev,train]"
```

Run tests:
```bash
pytest tests/ -v
```

Lint:
```bash
ruff check src/maskgen/ tests/
```

## Architecture

- **MaskGenerator** (`mask_generator.py`): High-level inference API. Loads a checkpoint, manages device, exposes `generate()` and `generate_and_save()`.
- **Net** (`model.py`): ConvNeXt V2 U-Net encoder-decoder. Uses GlobalResponseNorm, Stochastic Depth, and LayerNorm. Input: `(N, 3, H, W)` float32, output: `(N, 1, H, W)` logits. Requires spatial dims divisible by 32.
- **EMA**: Exponential Moving Average model (`torch.optim.swa_utils.AveragedModel`) is used during training and is the default for inference (`use_ema=True`).
- **Inference strategies**: `whole` (single pass), `tile` (overlapping patches with blend weights), `downsample` (resize-infer-upscale). Strategy is selected via a dict passed to `generate()`.
- **Training** (`train.py`): `train()` takes a config dict. Uses AdamW, cosine annealing with linear warmup, mixed precision, optional W&B logging.

## Code Style

- Formatter/linter: ruff (config in `pyproject.toml`)
- Line length: 120
- Lint rules: E, F, W, I (ignoring E501)
- Memory format: `torch.channels_last` for both training and inference
- Imports: explicit (no wildcard imports)
- Type hints: use `from __future__ import annotations` for `str | Path` union syntax

## Testing

- Tests use a `MockGenerator` pattern: subclass `MaskGenerator`, skip `__init__` checkpoint loading, use random weights on CPU. This avoids needing real `.pth` files in CI.
- No real model weights are needed to run the test suite.
- Tests cover: model output shapes, padding/unpadding, preprocessing (various input types), postprocessing, tiling grid computation, blend weights, and full inference pipeline.

## CI

- GitHub Actions (`.github/workflows/ci.yml`)
- **lint** job: `ruff check src/maskgen/ tests/` on ubuntu-latest
- **test** job: `pytest tests/ -v` on ubuntu-latest, macos-latest, windows-latest (Python 3.10)
- Triggered on push/PR to `main`
