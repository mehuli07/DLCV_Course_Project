import argparse
import copy
import os
import json
import torch
import torch.nn as nn

from config import TRAIN_CFG, PRUNE_CFG, CKPT_DIR, RESULTS_DIR
from datasets import get_loader
from models import build_model
from pruning import (
    compute_mask, apply_mask, extract_signed_mask,
    apply_signed_mask, get_sparsity,
    save_masks, load_masks, save_signed_mask, load_signed_mask,
)
from trainer import StandardTrainer, LRRTrainer, AWSTrainer, evaluate
from lmc import sgd_noise_stability, lmc_between


#  Step helpers

def warm_up(model, train_loader, val_loader, device, tag, save_path=None):
    """Initial warm-up training"""
    trainer = StandardTrainer(model, device)
    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=TRAIN_CFG["warmup_epochs"],
        tag=tag,
        save_path=save_path,
    )
    return history


def run_lrr(model, train_loader, val_loader, device, tag="lrr"):
    """Run T iterations of LRR and return (masks, signed_mask, history)"""
    trainer  = LRRTrainer(model, device)
    masks    = {}        
    all_hist = []

    for t in range(1, PRUNE_CFG["T"] + 1):
        masks    = compute_mask(model, PRUNE_CFG["prune_rate"], masks)
        acc, hist = trainer.run_iteration(
            train_loader, val_loader, masks,
            num_epochs=TRAIN_CFG["iter_epochs"],
            iter_idx=t, tag=tag,
        )
        all_hist.extend(hist)

    signed_mask = extract_signed_mask(model, masks)
    return masks, signed_mask, all_hist


def run_aws(model, train_loader, val_loader, device, tag="aws"):
    """Run T iterations of AWS and return (masks, signed_mask, history)"""
    trainer  = AWSTrainer(model, device)
    masks    = {}
    all_hist = []

    for t in range(1, PRUNE_CFG["T"] + 1):
        masks    = compute_mask(model, PRUNE_CFG["prune_rate"], masks)
        acc, hist = trainer.run_iteration(
            train_loader, val_loader, masks,
            num_epochs=TRAIN_CFG["iter_epochs"],
            iter_idx=t, tag=tag,
        )
        all_hist.extend(hist)

    signed_mask = extract_signed_mask(model, masks)
    return masks, signed_mask, all_hist


def final_train_and_eval(
    model, train_loader, val_loader, masks, device, tag, save_path=None
):
    """100-epoch final training; returns test accuracy"""
    trainer = StandardTrainer(model, device)
    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=TRAIN_CFG["final_epochs"],
        masks=masks,
        tag=tag,
        save_path=save_path,
    )
    acc = evaluate(model, val_loader, device)
    print(f"  [{tag}] Final test acc = {acc:.2f}%")
    return acc, history

#  Main experiment


def run_part_a(arch: str, dataset: str, device: torch.device):
    print(f"\n{'='*60}")
    print(f"  Part A — arch={arch}  dataset={dataset}")
    print(f"{'='*60}\n")

    ckpt = lambda name: os.path.join(CKPT_DIR, f"partA_{arch}_{dataset}_{name}.pt")
    res_path = os.path.join(RESULTS_DIR, f"partA_{arch}_{dataset}.json")

    # data
    train_loader = get_loader(dataset, train=True,
                              batch_size=TRAIN_CFG["batch_size"],
                              num_workers=TRAIN_CFG["num_workers"])
    val_loader   = get_loader(dataset, train=False,
                              batch_size=TRAIN_CFG["batch_size"],
                              num_workers=TRAIN_CFG["num_workers"])

    results = {}

    # (1) Dense baseline 
    print("--- (1) Dense baseline ---")
    dense_model = build_model(arch, dataset).to(device)
    warm_up(dense_model, train_loader, val_loader, device,
            tag="dense_warmup", save_path=ckpt("dense_warmup"))
    acc_dense, _ = final_train_and_eval(
        dense_model, train_loader, val_loader, None, device,
        tag="dense", save_path=ckpt("dense_final"),
    )
    results["dense"] = acc_dense

    
    init_state = copy.deepcopy(dense_model.state_dict())
    
    abs_init = {
        n: p.detach().abs().clone().cpu()
        for n, p in dense_model.named_parameters()
    }

    # LRR pipeline
    print("\n--- LRR pruning ---")
    lrr_model = build_model(arch, dataset).to(device)
    lrr_model.load_state_dict(copy.deepcopy(init_state))  # same init
    warm_up(lrr_model, train_loader, val_loader, device, tag="lrr_warmup")

    lrr_masks, lrr_signed, lrr_hist = run_lrr(
        lrr_model, train_loader, val_loader, device
    )
    save_masks(lrr_masks, ckpt("lrr_masks").replace(".pt", ""))
    save_signed_mask(lrr_signed, ckpt("lrr_signed").replace(".pt", ""))

    # (2) A(θ^LRR_T ⊙ m^LRR_T)
    print("\n--- (2) LRR subnetwork final training ---")
    acc_lrr, _ = final_train_and_eval(
        lrr_model, train_loader, val_loader, lrr_masks, device,
        tag="lrr_subnet", save_path=ckpt("lrr_subnet"),
    )
    results["lrr_subnet"] = acc_lrr

    # AWS pipeline
    print("\n--- AWS pruning ---")
    aws_model = build_model(arch, dataset).to(device)
    aws_model.load_state_dict(copy.deepcopy(init_state))
    warm_up(aws_model, train_loader, val_loader, device, tag="aws_warmup")

    aws_masks, aws_signed, aws_hist = run_aws(
        aws_model, train_loader, val_loader, device
    )
    save_masks(aws_masks, ckpt("aws_masks").replace(".pt", ""))
    save_signed_mask(aws_signed, ckpt("aws_signed").replace(".pt", ""))

    # (3) A(θ^AWS_T ⊙ m^AWS_T)
    print("\n--- (3) AWS subnetwork final training ---")
    acc_aws, _ = final_train_and_eval(
        aws_model, train_loader, val_loader, aws_masks, device,
        tag="aws_subnet", save_path=ckpt("aws_subnet"),
    )
    results["aws_subnet"] = acc_aws


    # (4) A(|θ_init| ⊙ s^LRR_T)
    print("\n--- (4) Random init + LRR signed mask ---")
    m4 = build_model(arch, dataset).to(device)
    apply_signed_mask(m4, lrr_signed, abs_init)
    acc_4, _ = final_train_and_eval(
        m4, train_loader, val_loader, lrr_masks, device,
        tag="abs_init_lrr_sign", save_path=ckpt("abs_init_lrr_sign"),
    )
    results["abs_init_lrr_sign"] = acc_4

    # (5) A(|θ_init| ⊙ s^AWS_T)
    print("\n--- (5) Random init + AWS signed mask ---")
    m5 = build_model(arch, dataset).to(device)
    apply_signed_mask(m5, aws_signed, abs_init)
    acc_5, _ = final_train_and_eval(
        m5, train_loader, val_loader, aws_masks, device,
        tag="abs_init_aws_sign", save_path=ckpt("abs_init_aws_sign"),
    )
    results["abs_init_aws_sign"] = acc_5

    # (6) A(θ_init ⊙ m^LRR_T)  — mask only, no sign
    print("\n--- (6) Random init + LRR mask (no sign) ---")
    m6 = build_model(arch, dataset).to(device)   # fresh random init
    apply_mask(m6, lrr_masks)
    acc_6, _ = final_train_and_eval(
        m6, train_loader, val_loader, lrr_masks, device,
        tag="init_lrr_mask_nosign", save_path=ckpt("init_lrr_mask_nosign"),
    )
    results["init_lrr_mask_nosign"] = acc_6

    # LMC / SGD-noise analysis 
    print("\n--- LMC / SGD-noise stability (AWS signed mask) ---")
    alphas_stab, barriers_stab = sgd_noise_stability(
        m5, train_loader, val_loader, device, masks=aws_masks,
        num_epochs=TRAIN_CFG["final_epochs"],
    )
    alphas_lmc, barriers_lmc = lmc_between(m5, aws_model, val_loader, device)

    results["lmc"] = {
        "stability_barriers": barriers_stab.tolist(),
        "lmc_barriers":       barriers_lmc.tolist(),
        "alphas":             alphas_lmc.tolist(),
    }

    # Summary 
    print("\n" + "="*50)
    print("  SUMMARY — Part A")
    print("="*50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:30s} : {v:.2f}%")
    print("="*50)

    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {res_path}")
    return results


#  Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",    default="vit",     choices=["vit", "resnet50"])
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",    type=int, default=TRAIN_CFG["seed"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    run_part_a(args.arch, args.dataset, device)
