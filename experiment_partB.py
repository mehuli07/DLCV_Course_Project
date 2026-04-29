import argparse
import copy
import os
import json
import torch

from config import TRAIN_CFG, PRUNE_CFG, CKPT_DIR, RESULTS_DIR
from datasets import get_loader
from models import build_model
from pruning import (
    apply_mask, apply_signed_mask,
    load_masks, load_signed_mask,
    compute_mask, extract_signed_mask,
    save_masks, save_signed_mask,
)
from trainer import StandardTrainer, evaluate
from experiment_partA import run_part_a, warm_up, run_lrr, run_aws


#  Head replacement 

def _replace_head(model, num_classes_tgt: int) -> None:
    
    import torch.nn as nn
    head = model.head if hasattr(model, "head") else model.backbone.fc
    in_feat = head.in_features
    new_head = nn.Linear(in_feat, num_classes_tgt)
    if hasattr(model, "head"):
        model.head = new_head
    else:
        model.backbone.fc = new_head



def _load_or_train_source(arch, src_dataset, device):
    
    ckpt = lambda name: os.path.join(
        CKPT_DIR, f"partA_{arch}_{src_dataset}_{name}"
    )

    lrr_mask_path = ckpt("lrr_masks")
    lrr_sign_path = ckpt("lrr_signed")
    aws_mask_path = ckpt("aws_masks")
    aws_sign_path = ckpt("aws_signed")

    if all(os.path.exists(p) for p in [lrr_mask_path, lrr_sign_path,
                                        aws_mask_path, aws_sign_path]):
        print("Loading pre-computed source masks from disk ...")
        return (
            load_masks(lrr_mask_path),
            load_signed_mask(lrr_sign_path),
            load_masks(aws_mask_path),
            load_signed_mask(aws_sign_path),
        )
    else:
        print("Source masks not found — running Part A first ...")
        run_part_a(arch, src_dataset, device)
        return (
            load_masks(lrr_mask_path),
            load_signed_mask(lrr_sign_path),
            load_masks(aws_mask_path),
            load_signed_mask(aws_sign_path),
        )


#  Transfer and fine-tune


def transfer_and_finetune(
    arch:         str,
    src_dataset:  str,
    tgt_dataset:  str,
    device:       torch.device,
    masks:        dict,
    signed_mask:  dict | None,
    abs_init:     dict | None,
    tag:          str,
    save_path:    str | None = None,
) -> float:
    
    from config import DATASET_CFG
    num_classes_tgt = DATASET_CFG[tgt_dataset]["num_classes"]

    train_loader = get_loader(tgt_dataset, train=True,
                              batch_size=TRAIN_CFG["batch_size"],
                              num_workers=TRAIN_CFG["num_workers"])
    val_loader   = get_loader(tgt_dataset, train=False,
                              batch_size=TRAIN_CFG["batch_size"],
                              num_workers=TRAIN_CFG["num_workers"])

    # build model for target dataset
    model = build_model(arch, tgt_dataset).to(device)

    if signed_mask is not None:
        apply_signed_mask(model, signed_mask, abs_init)
    elif masks is not None:
        apply_mask(model, masks)

    trainer = StandardTrainer(model, device)
    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=TRAIN_CFG["final_epochs"],
        masks=masks,
        tag=tag,
        save_path=save_path,
    )
    acc = evaluate(model, val_loader, device)
    print(f"  [{tag}] Transfer acc on {tgt_dataset} = {acc:.2f}%")
    return acc


#  Main experiment


def run_part_b(arch: str, src_dataset: str, tgt_dataset: str, device: torch.device):
    print(f"\n{'='*60}")
    print(f"  Part B — {src_dataset} → {tgt_dataset}  arch={arch}")
    print(f"{'='*60}\n")

    ckpt = lambda name: os.path.join(
        CKPT_DIR, f"partB_{arch}_{src_dataset}_{tgt_dataset}_{name}.pt"
    )
    res_path = os.path.join(
        RESULTS_DIR, f"partB_{arch}_{src_dataset}_{tgt_dataset}.json"
    )

    results = {}

    # (i) Baseline: train from scratch on target
    print("--- (i) Baseline: scratch training on target ---")
    train_loader_tgt = get_loader(tgt_dataset, train=True,
                                  batch_size=TRAIN_CFG["batch_size"],
                                  num_workers=TRAIN_CFG["num_workers"])
    val_loader_tgt   = get_loader(tgt_dataset, train=False,
                                  batch_size=TRAIN_CFG["batch_size"],
                                  num_workers=TRAIN_CFG["num_workers"])

    baseline_model = build_model(arch, tgt_dataset).to(device)
    trainer        = StandardTrainer(baseline_model, device)
    trainer.fit(train_loader_tgt, val_loader_tgt,
                num_epochs=TRAIN_CFG["final_epochs"],
                tag="baseline_tgt", save_path=ckpt("baseline"))
    acc_baseline = evaluate(baseline_model, val_loader_tgt, device)
    results["baseline"] = acc_baseline
    print(f"  Baseline acc = {acc_baseline:.2f}%")

    # Load source masks 
    lrr_masks, lrr_signed, aws_masks, aws_signed = _load_or_train_source(
        arch, src_dataset, device
    )

    # abs init for the target model (fresh random init)
    ref_model = build_model(arch, tgt_dataset).to(device)
    abs_init  = {
        n: p.detach().abs().clone().cpu()
        for n, p in ref_model.named_parameters()
    }
    del ref_model

    # (ii) Transfer LRR mask
    print("\n--- (ii) Transfer LRR mask → fine-tune on target ---")
    acc_lrr_mask = transfer_and_finetune(
        arch, src_dataset, tgt_dataset, device,
        masks=lrr_masks, signed_mask=None, abs_init=None,
        tag="transfer_lrr_mask", save_path=ckpt("lrr_mask"),
    )
    results["transfer_lrr_mask"] = acc_lrr_mask

    # (iii) Transfer LRR signed mask
    print("\n--- (iii) Transfer LRR signed mask → fine-tune on target ---")
    acc_lrr_sign = transfer_and_finetune(
        arch, src_dataset, tgt_dataset, device,
        masks=lrr_masks, signed_mask=lrr_signed, abs_init=abs_init,
        tag="transfer_lrr_sign", save_path=ckpt("lrr_sign"),
    )
    results["transfer_lrr_sign"] = acc_lrr_sign

    # (iv) Transfer AWS mask 
    print("\n--- (iv) Transfer AWS mask → fine-tune on target ---")
    acc_aws_mask = transfer_and_finetune(
        arch, src_dataset, tgt_dataset, device,
        masks=aws_masks, signed_mask=None, abs_init=None,
        tag="transfer_aws_mask", save_path=ckpt("aws_mask"),
    )
    results["transfer_aws_mask"] = acc_aws_mask

    # (v) Transfer AWS signed mask 
    print("\n--- (v) Transfer AWS signed mask → fine-tune on target ---")
    acc_aws_sign = transfer_and_finetune(
        arch, src_dataset, tgt_dataset, device,
        masks=aws_masks, signed_mask=aws_signed, abs_init=abs_init,
        tag="transfer_aws_sign", save_path=ckpt("aws_sign"),
    )
    results["transfer_aws_sign"] = acc_aws_sign

    # Summary 
    print("\n" + "="*50)
    print(f"  SUMMARY — Part B  ({src_dataset} → {tgt_dataset})")
    print("="*50)
    for k, v in results.items():
        print(f"  {k:35s} : {v:.2f}%")
    print("="*50)

    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {res_path}")
    return results


#  Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",        default="vit",     choices=["vit", "resnet50"])
    parser.add_argument("--src",         default="cifar10",  choices=["cifar10"])
    parser.add_argument("--tgt",         default="cifar100",
                        choices=["cifar100"])
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    run_part_b(args.arch, args.src, args.tgt, device)
