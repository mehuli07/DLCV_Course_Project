import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from config import DATA_DIR, DATASET_CFG


# transforms

def _build_transforms(dataset_name: str, train: bool, img_size: int):
    cfg  = DATASET_CFG[dataset_name]
    mean = cfg["mean"]
    std  = cfg["std"]

    if train:
        tfms = [
            T.RandomCrop(img_size, padding=4),
            T.RandomHorizontalFlip(),
        ]
        if img_size >= 64:                    
            tfms = [T.RandomResizedCrop(img_size)] + tfms[1:]
    else:
        tfms = [T.Resize(img_size)] if img_size != 32 else []

    tfms += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(tfms)


# loaders 

def get_loader(
    dataset_name: str,
    train: bool,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool | None = None,
) -> DataLoader:
    

    cfg      = DATASET_CFG[dataset_name]
    img_size = cfg["img_size"]
    tfms     = _build_transforms(dataset_name, train, img_size)

    if dataset_name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            DATA_DIR, train=train, transform=tfms, download=True
        )
    elif dataset_name == "cifar100":
        ds = torchvision.datasets.CIFAR100(
            DATA_DIR, train=train, transform=tfms, download=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    _shuffle = train if shuffle is None else shuffle
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )

