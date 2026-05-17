"""CIFAR-10 dataloaders with a shared, project-wide dataset root.

Where datasets live
-------------------
By default, every dataset (CIFAR-10 today; ImageNet, Kodak, etc. later) goes
under one root directory shared across runs and configs:

    <repo>/datasets/<dataset_name>/

So CIFAR-10 ends up at ``<repo>/datasets/cifar10/`` and is downloaded exactly
once. The location can be overridden in three ways, in priority order:

  1. The ``data.root`` field in the YAML config (passed to ``get_loaders``).
  2. The ``DJSCC_DATA_ROOT`` environment variable. Useful on HPC nodes that
     want a path like ``/scratch/$USER/datasets`` without touching configs.
  3. The default ``<repo>/datasets``.

A small file-based lock prevents races when several processes try to
download the same dataset in parallel.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ``<repo>/datasets/`` regardless of the caller's CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = _REPO_ROOT / "datasets"


def resolve_data_root(cfg_root: Optional[str] = None) -> Path:
    """Pick the shared dataset root using config > env > default."""
    if cfg_root:
        return Path(cfg_root).expanduser().resolve()
    env_root = os.environ.get("DJSCC_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return _DEFAULT_DATA_ROOT


def _dataset_dir(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_with_lock(dataset_dir: Path, build_fn, lock_timeout: float = 600.0):
    """Wrap ``build_fn()`` (which performs the download) with a file lock.

    Avoids two processes simultaneously downloading into the same directory.
    The lock is best-effort (sleeps until the lockfile is removed); on
    timeout it proceeds anyway and lets torchvision's own "already exists"
    short-circuit handle the rest.
    """
    lock = dataset_dir / ".download.lock"
    start = time.time()
    # Wait for any in-flight download to finish.
    while lock.exists() and (time.time() - start) < lock_timeout:
        time.sleep(1.0)
    try:
        lock.touch(exist_ok=True)
        return build_fn()
    finally:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass


def _make_transforms(augment: bool):
    train_tfms = ([transforms.RandomHorizontalFlip(),
                   transforms.RandomCrop(32, padding=4)]
                  if augment else [])
    train_tfms.append(transforms.ToTensor())   # -> [0, 1]
    train_tf = transforms.Compose(train_tfms)
    test_tf = transforms.Compose([transforms.ToTensor()])
    return train_tf, test_tf


def get_loaders(root: Optional[str] = None,
                batch_size: int = 128,
                num_workers: int = 2,
                augment: bool = True):
    """Build CIFAR-10 dataloaders.

    Args
    ----
    root: optional override of the shared dataset root. If None, falls back
        to ``DJSCC_DATA_ROOT`` env var, then to ``<repo>/datasets``.
    """
    data_root = resolve_data_root(root)
    cifar_dir = _dataset_dir(data_root, "cifar10")

    train_tf, test_tf = _make_transforms(augment)

    def _build():
        train_set = datasets.CIFAR10(root=str(cifar_dir), train=True,
                                     download=True, transform=train_tf)
        test_set = datasets.CIFAR10(root=str(cifar_dir), train=False,
                                    download=True, transform=test_tf)
        return train_set, test_set

    train_set, test_set = _download_with_lock(cifar_dir, _build)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
