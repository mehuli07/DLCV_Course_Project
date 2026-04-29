import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


# helpers 

def _is_prunable(name: str) -> bool:
    skip = ("bias", "norm", "bn", "cls_token", "pos_embed")
    return not any(k in name for k in skip)


def get_prunable_params(
    model: nn.Module,
) -> List[Tuple[str, torch.Tensor]]:
    return [
        (n, p)
        for n, p in model.named_parameters()
        if _is_prunable(n) and p.requires_grad
    ]


# mask creation 

def compute_mask(
    model: nn.Module,
    prune_rate: float,
    existing_masks: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    prunable = get_prunable_params(model)
    device = next(model.parameters()).device

    # collect magnitudes of currently alive weights
    all_mag = []
    for name, param in prunable:
        w = param.detach()
        if existing_masks is not None and name in existing_masks:
            alive = existing_masks[name].to(device)
            all_mag.append(w[alive.bool()].abs().flatten())
        else:
            all_mag.append(w.abs().flatten())

    all_mag_cat = torch.cat(all_mag)
    n_alive = all_mag_cat.numel()
    n_prune = max(1, int(prune_rate * n_alive))
    threshold = all_mag_cat.kthvalue(n_prune).values.item()

    new_masks = {}
    for name, param in prunable:
        w = param.detach().abs()
        mask = (w > threshold).float()
        if existing_masks is not None and name in existing_masks:
            mask = mask * existing_masks[name].to(device).float()
        new_masks[name] = mask.to(device)

    return new_masks


def apply_mask(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.data.mul_(masks[name].to(param.device))


# signed mask 

def extract_signed_mask(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    
    signed = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                masked = param.detach() * masks[name].to(param.device)
                signed[name] = masked.sign()   
    return signed


def apply_signed_mask(
    model: nn.Module,
    signed_mask: Dict[str, torch.Tensor],
    abs_init_params: Dict[str, torch.Tensor] | None = None,
) -> None:
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in signed_mask:
                s = signed_mask[name].to(param.device)
                if abs_init_params is not None and name in abs_init_params:
                    magnitude = abs_init_params[name].abs().to(param.device)
                else:
                    magnitude = param.data.abs()
                param.data.copy_(magnitude * s)


# sparsity query 

def get_sparsity(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
) -> float:
    
    total = alive = 0
    for name, param in get_prunable_params(model):
        t = param.numel()
        total += t
        if name in masks:
            alive += masks[name].sum().item()
        else:
            alive += t   
    return alive / max(total, 1)


def get_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# persistence

def save_masks(masks: Dict[str, torch.Tensor], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: v.cpu() for k, v in masks.items()}, path)


def load_masks(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def save_signed_mask(signed: Dict[str, torch.Tensor], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: v.cpu() for k, v in signed.items()}, path)


def load_signed_mask(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")
