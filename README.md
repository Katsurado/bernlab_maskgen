# maskgen

[![CI](https://github.com/Katsurado/bernlab_maskgen/actions/workflows/ci.yml/badge.svg)](https://github.com/Katsurado/bernlab_maskgen/actions/workflows/ci.yml)

Binary mask generation for microscopy images. Uses a ConvNeXt V2 U-Net with EMA to segment foreground from background in large microscopy images, with tiling and downsampling strategies for memory-efficient inference.

## Install

```bash
pip install "maskgen @ git+https://github.com/Katsurado/bernlab_maskgen.git"
```

With training dependencies:

```bash
pip install "maskgen[train] @ git+https://github.com/Katsurado/bernlab_maskgen.git"
```

With development dependencies:

```bash
pip install "maskgen[dev] @ git+https://github.com/Katsurado/bernlab_maskgen.git"
```

## Quick Start

```python
from maskgen import MaskGenerator, download_weights

gen = MaskGenerator(download_weights())
mask = gen.generate("image.tif", strategy={"name": "tile", "tile_size": 512, "overlap": 64})
mask.save("mask.png")
```

## Inference Strategies

| Strategy | Config | When to use |
|----------|--------|-------------|
| `whole` | `{"name": "whole"}` | Small images or large GPU memory (A100) |
| `tile` | `{"name": "tile", "tile_size": 512, "overlap": 64}` | Large images, preserves fine detail |
| `downsample` | `{"name": "downsample", "max_dim": 1024}` | Fast preview, less detail at high reduction |

## Training

See [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) for a Colab-ready training workflow.

Minimal example:

```python
from maskgen.train import train

config = {
    "channels": 128,
    "layers": [1, 2, 4, 2],
    "stochastic_depth": 0.1,
    "ema": 0.9999,
    "img_size": 512,
    "crop_per_img": 4,
    "batch_size": 4,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "warmup_epoch": 5,
    "scheduler_params": {"T-max": 100},
    "epochs": 100,
    "gradient_clip": 1.0,
}
train(config, checkpoint_dir="./checkpoints", data_root="./data")
```

## Lab Integration

Add this cell before your `Experiment` workflow to generate masks automatically:

```python
from maskgen import MaskGenerator, download_weights

gen = MaskGenerator(download_weights())
gen.generate_and_save(
    "path/to/first_image.tif",
    "path/to/masks/mask.png",
    strategy={"name": "tile", "tile_size": 512, "overlap": 64},
)
```

The mask is saved to disk and picked up by the existing `Experiment` config as usual.

## API Reference

### `MaskGenerator(checkpoint_path, device="auto", use_ema=True, config=None)`

Load model from a `.pth` checkpoint. `device` can be `"auto"`, `"cpu"`, `"cuda"`, or `"mps"`.

### `MaskGenerator.generate(image, strategy=None, threshold=0.5, return_prob=False)`

Generate a binary mask from an image. `image` accepts a file path, PIL Image, or HWC uint8 numpy array. Returns a PIL Image (mode `"L"`) or float32 numpy array if `return_prob=True`.

### `MaskGenerator.generate_and_save(image, output_path, strategy=None, threshold=0.5)`

Generate a mask and save it to `output_path`. Creates parent directories if needed. Returns the mask.

### `download_weights(url=DEFAULT_URL, cache_dir="~/.cache/maskgen", filename="best.pth", force=False)`

Download model weights from GitHub Releases. Caches locally and skips re-download unless `force=True`. Returns the path to the downloaded file.

### `train(config, checkpoint_dir, data_root, resume_from=None, use_wandb=True)`

Train the model. See [Training](#training) for the config dict format.

## Development

```bash
git clone https://github.com/Katsurado/bernlab_maskgen.git
cd bernlab_maskgen
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

## CI

GitHub Actions runs lint (ruff) and tests (pytest) on every push and PR to `main`, across Ubuntu, macOS, and Windows with Python 3.10.
