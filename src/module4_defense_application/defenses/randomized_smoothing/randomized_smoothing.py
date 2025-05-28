import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from defenses.randomized_smoothing.generate_randomized_smoothing_report import generate_randomized_smoothing_report


# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))

from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile


class NoisyDataset(Dataset):
    def __init__(self, base_dataset, sigma):
        self.base_dataset = base_dataset
        self.sigma = sigma

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        noise = torch.randn_like(x) * self.sigma
        return x + noise, y

    def __len__(self):
        return len(self.base_dataset)


def run_randomized_smoothing_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Randomized Smoothing defense for {attack_type}...")

    cfg = profile["defense_config"]["evasion_attacks"][attack_type]["randomized_smoothing"]
    sigma = cfg.get("sigma", 0.25)

    # Create noisy version of training set
    noisy_trainset = NoisyDataset(trainset, sigma=sigma)

    # Load model
    model = load_model_cfg_from_profile(profile)

    # Train on noisy data
    print(f"[*] Training model with Gaussian noise σ = {sigma} on training set...")
    train_model(model, noisy_trainset, valset, epochs=3, class_names=class_names)

    # Evaluate on clean test set
    acc, per_class = evaluate_model(model, testset, class_names=class_names)

    # Save results
    os.makedirs(f"results/evasion/{attack_type}", exist_ok=True)
    result_data = {
        "defense": "randomized_smoothing",
        "attack": attack_type,
        "sigma": sigma,
        "accuracy_after_defense": acc,
        "per_class_accuracy": per_class,
        "params": cfg
    }

    result_path = f"results/evasion/{attack_type}/randomized_smoothing_results.json"
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"[✔] Results saved to {result_path}")

    # Generate Markdown report
    md_path = f"results/evasion/{attack_type}/randomized_smoothing_report.md"
    generate_randomized_smoothing_report(json_file=result_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
