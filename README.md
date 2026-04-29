# DLCV_Course_Project


**Understanding Trainability of Sparse Vision Transformers via Winning Sign
and Cross-Dataset Generalization**

---

## Repository layout

```
dlcv_project/
├── config.py              # All hyper-parameters in one place
├── datasets.py            # DataLoader factories (CIFAR-10/100)
├── models.py              # ViT (from scratch) + ResNet-50 (torchvision)
├── pruning.py             # Mask creation, signed-mask, sparsity helpers
├── trainer.py             # Standard / LRR / AWS training loops
├── lmc.py                 # Linear Mode Connectivity & SGD-noise stability
├── experiment_partA.py    # Part A: Winning Sign in ViT
├── experiment_partB.py    # Part B: Cross-Dataset Generalization
├── experiment_partC.py    # Part C: CNN Baseline (ResNet-50)
└── run_all.sh             # End-to-end pipeline script
```

---

## Setup

```bash
pip install torch torchvision matplotlib numpy
```

---

## Running

### Full pipeline
```bash
bash run_all.sh cuda        # or cpu
```

### Individual parts
```bash
# Part A — ViT winning sign on CIFAR-10
python experiment_partA.py --arch vit --dataset cifar10 --device cuda

# Part B — Transfer CIFAR-10 masks to CIFAR-100
python experiment_partB.py --arch vit --src cifar10 --tgt cifar100 --device cuda

# Part C — ResNet-50 baseline
python experiment_partC.py --dataset cifar10 --device cuda

```

---

## What each experiment evaluates

### Part A: Winning Sign in Vision Transformers

Six configurations following the AWS paper (Oh et al., ICLR 2025):

| Config | Description |
|--------|-------------|
| `dense` | A(θ_init) — Dense baseline |
| `lrr_subnet` | A(θ^LRR ⊙ m^LRR) — LRR subnetwork |
| `aws_subnet` | A(θ^AWS ⊙ m^AWS) — AWS subnetwork |
| `abs_init_lrr_sign` | A(\|θ_init\| ⊙ s^LRR) — LRR signed mask on random init |
| `abs_init_aws_sign` | A(\|θ_init\| ⊙ s^AWS) — AWS signed mask on random init |
| `init_lrr_mask_nosign` | A(θ_init ⊙ m^LRR) — Mask only, no sign info |

LMC and SGD-noise stability are computed for the AWS signed-mask configuration.

### Part B: Cross-Dataset Generalization

Masks/signed masks learned on **CIFAR-10** are transferred to:
- CIFAR-100

Configurations: scratch baseline, LRR mask, LRR signed mask, AWS mask, AWS signed mask.

### Part C: CNN Baseline

Runs the same Part-A six configurations on **ResNet-50** for architectural comparison.

---

## Key design choices

### AWS training (Algorithm 1 — Oh et al. 2025)
During each forward pass, normalization-layer parameters ψ are replaced by:

```
(ψ_t, ψ_init)_α = α·ψ_t + (1−α)·ψ_init,   α ~ U(0,1)
```

This is implemented in `trainer.py → AWSTrainer._one_epoch`.

### Signed mask
```python
s = sign0(θ ⊙ m)    # ∈ {-1, 0, +1}
# Applied as:
θ_new = |θ_init| ⊙ s
```

Implemented in `pruning.py → extract_signed_mask / apply_signed_mask`.

### Global magnitude pruning
`pruning.py → compute_mask` implements global unstructured pruning:
removes the lowest-magnitude *active* weights across all layers simultaneously.

---


## References

1. Frankle & Carbin, "The Lottery Ticket Hypothesis", ICLR 2019
2. Renda et al., "Comparing Rewinding and Fine-tuning", ICLR 2020
3. Oh, Baik & Lee, "Find a Winning Sign: Sign Is All We Need", ICLR 2025
