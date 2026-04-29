import copy
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from config import TRAIN_CFG, CKPT_DIR
from pruning import apply_mask, get_sparsity


#  Utility helpers


def _build_optimizer(model: nn.Module, lr: float = None) -> torch.optim.Optimizer:
    cfg = TRAIN_CFG
    lr = lr if lr is not None else cfg["lr"]

    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
            nesterov=True,
        )
    else:
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=cfg["weight_decay"],
        )


def _build_scheduler(optimizer, num_epochs: int, milestones: List[float] | None = None):
    cfg = TRAIN_CFG
    if cfg["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        # step decay
        ms = [int(m * num_epochs) for m in (milestones or cfg["lr_milestones"])]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=ms, gamma=cfg["lr_gamma"]
        )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def _one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    masks:     Dict | None,
    aws_norm_init: Dict | None = None,   
) -> float:
    
    model.train()
    total_loss = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # AWS forward-pass with interpolated norm params
        if aws_norm_init is not None:
            _aws_interpolate(model, aws_norm_init, device)

        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

        # re-zero pruned weights after gradient step
        if masks is not None:
            apply_mask(model, masks)

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def _aws_interpolate(
    model:          nn.Module,
    norm_init:      Dict[str, torch.Tensor],
    device:         torch.device,
) -> None:

    alpha = torch.rand(1).item()
    with torch.no_grad():   
        for name, param in model.named_parameters():
            if name in norm_init:
                pinit = norm_init[name].to(device)
                param.data.copy_(alpha * param.data + (1.0 - alpha) * pinit)


#  1. Standard Trainer


class StandardTrainer:
    

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        num_epochs:   int,
        masks:        Dict | None = None,
        tag:          str = "standard",
        save_path:    str | None = None,
    ) -> List[Dict]:
        model = self.model.to(self.device)
        optimizer = _build_optimizer(model)
        scheduler = _build_scheduler(optimizer, num_epochs)
        history = []

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            loss = _one_epoch(
                model, train_loader, optimizer, self.criterion, self.device, masks
            )
            scheduler.step()
            acc = evaluate(model, val_loader, self.device)
            elapsed = time.time() - t0

            row = dict(epoch=epoch, loss=loss, val_acc=acc,
                       sparsity=get_sparsity(model, masks or {}))
            history.append(row)
            print(
                f"[{tag}] Ep {epoch:3d}/{num_epochs}  "
                f"loss={loss:.4f}  val_acc={acc:.2f}%  "
                f"sparsity={row['sparsity']:.3f}  ({elapsed:.1f}s)"
            )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        return history


#  2. LRR Trainer


class LRRTrainer:

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def run_iteration(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        masks:        Dict,
        num_epochs:   int,
        iter_idx:     int,
        tag:          str = "lrr",
    ) -> Tuple[float, List[Dict]]:

        model     = self.model.to(self.device)
        optimizer = _build_optimizer(model)
        scheduler = _build_scheduler(optimizer, num_epochs)
        history   = []

        apply_mask(model, masks)    

        for epoch in range(1, num_epochs + 1):
            loss = _one_epoch(
                model, train_loader, optimizer, self.criterion,
                self.device, masks
            )
            scheduler.step()
            acc = evaluate(model, val_loader, self.device)
            history.append(dict(iter=iter_idx, epoch=epoch, loss=loss, val_acc=acc))

        print(
            f"[{tag}] Iter {iter_idx}  "
            f"remaining={get_sparsity(model, masks):.3f}  "
            f"val_acc={history[-1]['val_acc']:.2f}%"
        )
        return history[-1]["val_acc"], history

    def final_train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        masks:        Dict,
        num_epochs:   int,
        save_path:    str | None = None,
        tag:          str = "lrr_final",
    ) -> List[Dict]:
        trainer = StandardTrainer(self.model, self.device)
        return trainer.fit(
            train_loader, val_loader, num_epochs, masks, tag, save_path
        )


#  3. AWS Trainer

class AWSTrainer:

    def __init__(self, model: nn.Module, device: torch.device):
        self.model     = model
        self.device    = device
        self.criterion = nn.CrossEntropyLoss()
        # store the initial (un-modified) norm-layer parameters
        self._init_norm_params: Dict[str, torch.Tensor] = {}
        self._capture_norm_init()

    def _capture_norm_init(self) -> None:
        for name, param in self.model.named_parameters():
            if self._is_norm(name):
                self._init_norm_params[name] = param.detach().clone().cpu()

    def _is_norm(self, name: str) -> bool:
        return "norm" in name or "bn" in name

    def run_iteration(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        masks:        Dict,
        num_epochs:   int,
        iter_idx:     int,
        tag:          str = "aws",
    ) -> Tuple[float, List[Dict]]:
        model = self.model.to(self.device)
        optimizer = _build_optimizer(model)
        scheduler = _build_scheduler(optimizer, num_epochs)
        history = []

        apply_mask(model, masks)

        norm_init_gpu = {
            k: v.to(self.device)
            for k, v in self._init_norm_params.items()
        }

        for epoch in range(1, num_epochs + 1):
            loss = _one_epoch(
                model, train_loader, optimizer, self.criterion,
                self.device, masks, aws_norm_init=norm_init_gpu
            )
            scheduler.step()
            acc = evaluate(model, val_loader, self.device)
            history.append(dict(iter=iter_idx, epoch=epoch, loss=loss, val_acc=acc))

        print(
            f"[{tag}] Iter {iter_idx}  "
            f"remaining={get_sparsity(model, masks):.3f}  "
            f"val_acc={history[-1]['val_acc']:.2f}%"
        )
        return history[-1]["val_acc"], history

    def final_train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        masks:        Dict,
        num_epochs:   int,
        save_path:    str | None = None,
        tag:          str = "aws_final",
    ) -> List[Dict]:
        
        trainer = StandardTrainer(self.model, self.device)
        return trainer.fit(
            train_loader, val_loader, num_epochs, masks, tag, save_path
        )
