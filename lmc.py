import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np

from trainer import evaluate, _build_optimizer, _build_scheduler, _one_epoch
from pruning import apply_mask
from config import TRAIN_CFG


# interpolation

def _interpolate_params(
    model_a: nn.Module,
    model_b: nn.Module,
    alpha:   float,
) -> nn.Module:
    
    interp = copy.deepcopy(model_a)
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    sd_i = {}
    for key in sd_a:
        sd_i[key] = alpha * sd_a[key].float() + (1.0 - alpha) * sd_b[key].float()
    interp.load_state_dict(sd_i)
    return interp


# error barrier

def error_barrier(
    model_a:  nn.Module,
    model_b:  nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    n_points: int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    
    alphas = np.linspace(0, 1, n_points)
    errors = []

    err_a = 100.0 - evaluate(model_a, loader, device)
    err_b = 100.0 - evaluate(model_b, loader, device)
    mean_err = (err_a + err_b) / 2.0

    for alpha in alphas:
        interp     = _interpolate_params(model_a, model_b, alpha).to(device)
        err_interp = 100.0 - evaluate(interp, loader, device)
        errors.append(err_interp)
        del interp

    barriers = np.array(errors) - mean_err
    return alphas, barriers


# stability

def sgd_noise_stability(
    model_init:   nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    masks:        Dict | None = None,
    num_epochs:   int = None,
    n_trials:     int = 2,
    n_points:     int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    
    num_epochs = num_epochs or TRAIN_CFG["final_epochs"]
    criterion  = nn.CrossEntropyLoss()

    trained_models = []
    for _ in range(n_trials):
        m = copy.deepcopy(model_init).to(device)
        opt  = _build_optimizer(m)
        sched = _build_scheduler(opt, num_epochs)
        for _ in range(num_epochs):
            _one_epoch(m, train_loader, opt, criterion, device, masks)
            sched.step()
        if masks:
            apply_mask(m, masks)
        trained_models.append(m)

    alphas, barriers = error_barrier(
        trained_models[0], trained_models[1], val_loader, device, n_points
    )
    return alphas, barriers


# LMC between two solutions

def lmc_between(
    model_a:  nn.Module,
    model_b:  nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    n_points: int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    
    return error_barrier(model_a, model_b, loader, device, n_points)


# max barrier 

def max_error_barrier(barriers: np.ndarray) -> float:
    
    return float(np.max(barriers))
