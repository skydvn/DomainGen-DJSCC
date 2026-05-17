"""CIFAR-10 dataloaders (images in [0, 1], no normalization)."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _transform():
    return transforms.Compose([
        transforms.ToTensor(),  # -> [0, 1]
    ])


def get_loaders(root: str = "./data_cifar",
                batch_size: int = 128,
                num_workers: int = 2,
                augment: bool = True):
    train_tfms = [transforms.RandomHorizontalFlip(),
                  transforms.RandomCrop(32, padding=4)] if augment else []
    train_tfms.append(transforms.ToTensor())
    train_tf = transforms.Compose(train_tfms)
    test_tf = _transform()

    train_set = datasets.CIFAR10(root=root, train=True, download=True,
                                 transform=train_tf)
    test_set = datasets.CIFAR10(root=root, train=False, download=True,
                                transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
