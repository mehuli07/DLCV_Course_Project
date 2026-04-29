#!/usr/bin/env bash


DEVICE=${1:-cuda}
SEED=42

echo "============================================================"
echo "  DLCV Project — Sparse ViT Winning Sign"
echo "  Device: $DEVICE"
echo "============================================================"

# Part A — ViT on CIFAR-10
echo ""
echo "[1/4] Part A: ViT on CIFAR-10"
python experiment_partA.py --arch vit --dataset cifar10 --device $DEVICE --seed $SEED

# Part B — Transfer to CIFAR-100
echo ""
echo "[2/4] Part B: Transfer CIFAR-10 → CIFAR-100"
python experiment_partB.py --arch vit --src cifar10 --tgt cifar100 --device $DEVICE --seed $SEED

# Part C — ResNet-50 baseline
echo ""
echo "[3/4] Part C: ResNet-50 on CIFAR-10"
python experiment_partC.py --dataset cifar10 --device $DEVICE --seed $SEED


