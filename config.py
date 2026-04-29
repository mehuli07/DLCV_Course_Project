"""
config.py — Central configuration for the DLCV project:
"Understanding Trainability of Sparse Vision Transformers via Winning Sign
and Cross-Dataset Generalization"
"""

import os

# ─────────────────────────── paths ───────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT, "data")
CKPT_DIR    = os.path.join(ROOT, "checkpoints")
LOG_DIR     = os.path.join(ROOT, "logs")
RESULTS_DIR = os.path.join(ROOT, "results")

for d in [DATA_DIR, CKPT_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────── datasets ────────────────────────
DATASET_CFG = {
    "cifar10": {
        "num_classes": 10,
        "img_size":    32,
        "mean":        (0.4914, 0.4822, 0.4465),
        "std":         (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "num_classes": 100,
        "img_size":    32,
        "mean":        (0.5071, 0.4865, 0.4409),
        "std":         (0.2673, 0.2564, 0.2762),
    },
    "tiny_imagenet": {
        "num_classes": 200,
        "img_size":    64,
        "mean":        (0.4802, 0.4481, 0.3975),
        "std":         (0.2770, 0.2691, 0.2821),
    },
}

# ─────────────────────────── model ───────────────────────────
VIT_CFG = {
    "patch_size":   4,     # suitable for 32×32 CIFAR
    "embed_dim":    192,
    "depth":        9,
    "num_heads":    3,
    "mlp_ratio":    4.0,
    "drop_rate":    0.1,
    "attn_drop":    0.0,
}

# ─────────────────────────── training ────────────────────────
TRAIN_CFG = {
    "batch_size":   128,
    "num_workers":  4,
    "seed":         42,

    # warm-up training before first LRR/AWS iteration
    "warmup_epochs": 10,

    # epochs per LRR/AWS iteration
    "iter_epochs":   10,

    # final fine-tuning after T-th iteration
    "final_epochs": 100,

    # optimiser
    "optimizer":    "sgd",      # "sgd" | "adam"
    "lr":           0.1,
    "momentum":     0.9,
    "weight_decay": 5e-4,

    # cosine LR scheduler (used in final training)
    "scheduler":    "cosine",

    # step-decay milestones as fractions of final_epochs
    "lr_milestones": [0.5, 0.75],
    "lr_gamma":      0.1,
}

# ─────────────────────────── pruning ─────────────────────────
PRUNE_CFG = {
    # number of LRR/AWS iterative pruning rounds
    "T": 10,

    # fraction of non-zero weights removed per round  (≈20 %)
    "prune_rate": 0.20,

    # remaining-parameter ratios to evaluate at (for plots)
    "eval_ratios": [0.8, 0.33, 0.13, 0.06, 0.02],
}

# ─────────────────────────── AWS ─────────────────────────────
AWS_CFG = {
    # α ~ Uniform(0,1) used to interpolate norm-layer params
    # No extra hyper-params needed beyond the training config.
    "enabled": True,
}
