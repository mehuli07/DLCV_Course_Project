import argparse
import torch

from config import TRAIN_CFG
from experiment_partA import run_part_a


def run_part_c(dataset: str, device: torch.device):
    print(f"\n{'='*60}")
    print(f"  Part C — CNN Baseline (ResNet-50)  dataset={dataset}")
    print(f"{'='*60}\n")
    return run_part_a(arch="resnet50", dataset=dataset, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",    type=int, default=TRAIN_CFG["seed"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    run_part_c(args.dataset, torch.device(args.device))
