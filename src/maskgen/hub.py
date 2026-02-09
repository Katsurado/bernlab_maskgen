"""
hub.py

Download model weights from GitHub Releases.

Usage:
    from maskgen.hub import download_weights

    path = download_weights()  # downloads to ~/.cache/maskgen/best.pth
    gen = MaskGenerator(path)
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from tqdm.auto import tqdm

DEFAULT_URL = "https://github.com/Katsurado/bernlab_maskgen/releases/latest/download/best.pth"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "maskgen"


class _DownloadProgressBar(tqdm):
    """tqdm wrapper for urllib reporthook."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int = -1):
        if total_size > 0:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_weights(
    url: str = DEFAULT_URL,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    filename: str = "best.pth",
    force: bool = False,
) -> str:
    """
    Download model weights from a URL (typically a GitHub Release asset).

    Args:
        url: Direct download URL for the .pth file.
        cache_dir: Local directory to cache the file.
        filename: Name for the cached file.
        force: Re-download even if the file already exists.

    Returns:
        Path to the downloaded .pth file (as a string).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / filename

    if dest.exists() and not force:
        print(f"Using cached weights: {dest}")
        return str(dest)

    print(f"Downloading weights from {url} ...")
    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)

    print(f"Saved to {dest}")
    return str(dest)
