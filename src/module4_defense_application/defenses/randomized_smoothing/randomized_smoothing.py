import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import shutil

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from defenses.randomized_smoothing.generate_randomized_smoothing_report import generate_randomized_smoothing_report

# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))
from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile


class NoisyDataset(Dataset):
    def __init__(self, base_dataset, sigma):
        self.base_dataset = base_dataset
        self.sigma = sigma
        self.cached_originals = []
        self.cached_noisy = []

        for i in range(min(5, len(base_dataset))):
            x, _ = base_dataset[i]
            self.cached_originals.append(x.clone())
            self.cached_noisy.append(x + torch.randn_like(x) * sigma)

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

    # Directories
    base_dir = f"results/evasion/{attack_type}/randomized_smoothing"
    examples_dir = os.path.join(base_dir, "noisy_examples")
    hist_dir = os.path.join(base_dir, "histograms")
    shutil.rmtree(examples_dir, ignore_errors=True)
    shutil.rmtree(hist_dir, ignore_errors=True)
    os.makedirs(examples_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)

    # Create noisy version of training set
    noisy_trainset = NoisyDataset(trainset, sigma=sigma)

    # Save visual examples
    print("[*] Saving original vs noisy examples...")
    for i, (orig, noisy) in enumerate(zip(noisy_trainset.cached_originals, noisy_trainset.cached_noisy)):
        fig, axs = plt.subplots(1, 2, figsize=(4, 2))
        axs[0].imshow(to_pil_image(orig), cmap='gray')
        axs[0].set_title("Original")
        axs[0].axis("off")
        axs[1].imshow(to_pil_image(noisy), cmap='gray')
        axs[1].set_title("Noisy")
        axs[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(examples_dir, f"example_{i}.png"), dpi=150)
        plt.close()

    # Save noise histogram
    print("[*] Saving histogram of noise distribution...")
    noise = (torch.stack(noisy_trainset.cached_noisy) - torch.stack(noisy_trainset.cached_originals)).flatten().numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(noise, bins=40, color='skyblue', edgecolor='black')
    plt.title("Distribution of Added Gaussian Noise")
    plt.xlabel("Noise Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(hist_dir, "noise_distribution.png"))
    plt.close()

    # Load model
    model = load_model_cfg_from_profile(profile)

    # Train on noisy data
    print(f"[*] Training model with Gaussian noise σ = {sigma} on training set...")
    train_model(model, noisy_trainset, valset, epochs=3, class_names=class_names)

    # Evaluate on clean test set
    acc, per_class = evaluate_model(model, testset, class_names=class_names)

    # Save results
    os.makedirs(base_dir, exist_ok=True)
    result_data = {
        "defense": "randomized_smoothing",
        "attack": attack_type,
        "sigma": sigma,
        "accuracy_after_defense": acc,
        "per_class_accuracy": per_class,
        "params": cfg
    }

    result_path = os.path.join(base_dir, "randomized_smoothing_results.json")
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"[✔] Results saved to {result_path}")

    # Generate Markdown report
    md_path = os.path.join(base_dir, "randomized_smoothing_report.md")
    generate_randomized_smoothing_report(json_file=result_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
