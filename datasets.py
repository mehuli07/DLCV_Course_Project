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
    elif dataset_name == "tiny_imagenet":
        split = "train" if train else "val"
        ds = _TinyImageNet(DATA_DIR, split=split, transform=tfms)
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


# Tiny-ImageNet 

class _TinyImageNet(torch.utils.data.Dataset):

    def __init__(self, root: str, split: str = "train", transform=None):
        self.root      = os.path.join(root, "tiny-imagenet-200")
        self.split     = split
        self.transform = transform

        
        with open(os.path.join(self.root, "wnids.txt")) as f:
            self.classes = [l.strip() for l in f]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        if self.split == "train":
            for cls in self.classes:
                img_dir = os.path.join(self.root, "train", cls, "images")
                if not os.path.isdir(img_dir):
                    continue
                for fname in os.listdir(img_dir):
                    if fname.endswith(".JPEG"):
                        samples.append(
                            (os.path.join(img_dir, fname), self.class_to_idx[cls])
                        )
        else:  
            ann_file = os.path.join(self.root, "val", "val_annotations.txt")
            img_dir  = os.path.join(self.root, "val", "images")
            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    fname, cls = parts[0], parts[1]
                    if cls in self.class_to_idx:
                        samples.append(
                            (os.path.join(img_dir, fname), self.class_to_idx[cls])
                        )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
