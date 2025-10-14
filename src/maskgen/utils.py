import torch
import os


def set_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return device

def list_dir(dirpath):
    """
    Return all non-hidden entries in dirpath.
    Skips files/folders starting with '.' (e.g., .DS_Store) to
    ensure compatibility with Mac
    """
    return [name for name in os.listdir(dirpath) if not name.startswith('.')]