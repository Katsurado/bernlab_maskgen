# Plan: Package, CI, Colab Training, and Lab Integration for maskgen

## Context

maskgen is a research codebase with no packaging, no CI, and a training script that only works locally via argparse. The goal is to make it: (1) pip-installable from GitHub, (2) easy to train on Colab with Google Drive, (3) dead-simple to use in notebooks, and (4) integrate cleanly into the lab's existing `image_processing` Experiment workflow.

The lab workflow (see `run_image_processing-5-15-24-Ninetales (1).ipynb`) uses an `Experiment` class that reads a YAML config and expects pre-made mask images in a folder. Masks are typically generated from a single image (first or last in the dataset). maskgen should slot in as a cell right before the Experiment runs — generate the mask, save it, then proceed as normal.

---

## Files to Change

| File | Action | What |
|------|--------|------|
| `.gitignore` | Modify | Expand to cover data/, model/, wandb/, .egg-info, etc. |
| `pyproject.toml` | Create | Hatchling build, deps, `[train]` and `[dev]` extras |
| `src/maskgen/__init__.py` | Modify | Export `MaskGenerator` + `train` module |
| `src/maskgen/train.py` | Create | Refactored training module from `src/train.py` |
| `src/maskgen/data.py` | Modify | Remove unused imports (matplotlib, F, DataLoader) |
| `tests/test_mask_generator.py` | Modify | Remove `sys.path.insert` hack |
| `notebooks/train_colab.ipynb` | Create | Colab-ready training notebook |
| `src/maskgen/mask_generator.py` | Modify | Add `generate_and_save()` convenience method |
| `src/maskgen/hub.py` | Create | `download_weights()` helper to fetch .pth from GitHub Releases |
| `.github/workflows/ci.yml` | Create | pytest + ruff on push/PR |

---

## Step 1: `.gitignore`

Expand to cover all artifacts: `data/`, `model/`, `chechpoints/`, `checkpoints/`, `stuf/`, `wandb/`, `src/wandb/`, `*.egg-info/`, `dist/`, `build/`, `.pytest_cache/`, `src/config.json`, `src/*.ipynb`.

## Step 2: `pyproject.toml`

- Build backend: `hatchling` (understands src layout natively)
- Core deps: `torch>=2.0`, `torchvision>=0.15`, `numpy`, `pillow`, `tqdm`
- `[train]` extras: `wandb`, `torchinfo`
- `[dev]` extras: `pytest`, `ruff`
- Package path: `src/maskgen`
- Ruff config: `line-length = 120`, select `E, F, W, I`, ignore `E501`
- Python: `>=3.10`

## Step 3: `src/maskgen/data.py` cleanup

Remove 3 unused imports that ruff would flag and that block clean import chains:
- `import matplotlib.pyplot as plt` (line 7, never used)
- `import torch.nn.functional as F` (line 5, never used)
- `from torch.utils.data import DataLoader` (line 11, only `Dataset` is used)

## Step 4: `src/maskgen/train.py` (new module)

Refactor `src/train.py` into an importable module. Key changes from original:

1. **No argparse** — `train()` takes a config dict + explicit `data_root` and `checkpoint_dir` paths
2. **W&B is optional** — `try: import wandb` with fallback to print-only logging
3. **`torchinfo` is optional** — wrapped in try/except
4. **Fix bugs from original:**
   - `config['clip_grad']` → `config.get('clip_grad', config.get('gradient_clip'))` (KeyError fix)
   - `batch_bar.update` → `batch_bar.update()` (missing parentheses, progress bar never advanced)
5. **Colab-friendly** — auto-detect `num_workers` (2 on Colab, up to 8 locally)
6. **Resume support** — optional `resume_from` parameter to continue from a checkpoint
7. **Explicit imports** — no more `from maskgen.utils import *`

Function signature:
```python
def train(config: dict,
          checkpoint_dir: str | Path = "./checkpoints",
          data_root: str | Path = "./data",
          resume_from: str | Path | None = None,
          use_wandb: bool = True) -> None:
```

## Step 5: `src/maskgen/__init__.py`

```python
from . import data, model, utils, train
from .mask_generator import MaskGenerator
from .hub import download_weights

__all__ = ["data", "model", "utils", "train", "MaskGenerator", "download_weights"]
```

After this: `from maskgen import MaskGenerator` and `from maskgen.train import train` both work.

## Step 6: `tests/test_mask_generator.py`

Remove the `sys.path.insert(0, 'src')` hack (lines 12-14). With `pip install -e .` this is unnecessary.

## Step 7: `notebooks/train_colab.ipynb`

~6 cells:
1. `!pip install "maskgen[train] @ git+https://github.com/<user>/maskgen.git"`
2. Mount Google Drive, set `DATA_ROOT` and `CHECKPOINT_DIR` paths on Drive
3. Define config dict (all hyperparams, optional W&B settings)
4. `from maskgen.train import train; train(config, ...)`
5. Test inference: `from maskgen import MaskGenerator, download_weights` → download weights → generate mask

## Step 8: Easy notebook API — `MaskGenerator.generate_and_save()`

Add a convenience method to `MaskGenerator` for the common lab use case: generate a mask from one image and save it to a folder. This keeps the core `generate()` method unchanged.

```python
# New method on MaskGenerator:
def generate_and_save(
    self,
    image: str | Path | Image.Image | np.ndarray,
    output_path: str | Path,
    strategy: dict | None = None,
    threshold: float = 0.5,
) -> Image.Image:
    """Generate mask and save to disk. Returns the mask."""
    mask = self.generate(image, strategy=strategy, threshold=threshold)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)
    return mask
```

**Lab integration** — in the experiment notebook, this becomes one cell before `Experiment`:

```python
# Cell: Generate Masks
from maskgen import MaskGenerator, download_weights

gen = MaskGenerator(download_weights())
gen.generate_and_save(
    "path/to/first_image.tif",
    "path/to/masks/mask.png",
    strategy={"name": "tile", "tile_size": 512, "overlap": 64},
)
```

Then the existing `Experiment` picks up the mask from that folder as usual — zero changes to `image_processing`.

## Step 9: `src/maskgen/hub.py` (new — weights download helper)

Model weights are hosted on GitHub Releases (not in the git repo). This module provides a simple download function:

```python
from maskgen.hub import download_weights

# Downloads best.pth from the latest GitHub Release to ./weights/best.pth
path = download_weights()

gen = MaskGenerator(path)
```

Implementation:
- Uses `urllib.request` (stdlib, no extra deps) to download from a GitHub Releases URL
- Default URL points to the repo's latest release asset (`best.pth`)
- Caches to a local directory (default `~/.cache/maskgen/`) — skips download if file exists
- Shows a progress bar via `tqdm`
- Also export from `__init__.py` so `from maskgen import download_weights` works

After the package is set up, you upload the weights once with:
```
gh release create v0.1.0 model/best.pth --title "v0.1.0" --notes "Initial release with trained weights"
```

## Step 10: `.github/workflows/ci.yml`

Two parallel jobs:
- **lint**: `ruff check src/ tests/` (ubuntu-latest only, no need to lint on every OS)
- **test**: `pip install -e ".[dev,train]"` then `pytest tests/ -v`
  - Matrix: `ubuntu-latest`, `macos-latest`, `windows-latest` — all on Python 3.10
  - Ensures the package works on Linux (Colab/servers), macOS (your dev machine), and Windows (lab members)

---

## Bugs Fixed Along the Way

1. `config['clip_grad']` KeyError — config uses `gradient_clip`, code uses `clip_grad`
2. `batch_bar.update` missing `()` — progress bar never advanced during training
3. `from maskgen.utils import *` — silently imports `os`, replaced with explicit imports
4. Unused `matplotlib` import in `data.py` — would cause ImportError without `[train]` extras

---

## Verification

1. `pip install -e ".[dev,train]"` succeeds
2. `python -c "from maskgen import MaskGenerator"` works
3. `python -c "from maskgen.train import train"` works
4. `pytest tests/ -v` passes
5. `ruff check src/ tests/` passes
