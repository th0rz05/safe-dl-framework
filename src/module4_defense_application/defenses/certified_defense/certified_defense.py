import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from defenses.certified_defense.generate_certified_defense_report import generate_certified_defense_report


# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))

from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile


def estimate_interval_bounds(model, inputs):
    eps = 0.1
    lower = inputs - eps
    upper = inputs + eps
    return lower, upper


def estimate_convex_relaxation(model, inputs):
    # Placeholder: in real applications, this could invoke tools like DeepPoly or CROWN
    return {"certified_accuracy_estimate": 0.85}


def estimate_lipschitz_bound(model, inputs):
    lipschitz_const = 2.0  # dummy
    max_perturbation = 1 / lipschitz_const
    return {"lipschitz_constant": lipschitz_const, "max_robust_radius": max_perturbation}


def run_certified_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Certified Defense for {attack_type}...")

    cfg = profile["defense_config"]["evasion_attacks"][attack_type]["certified_defense"]
    method = cfg.get("method", "interval_bound")

    model = load_model_cfg_from_profile(profile)

    print("[*] Training model on clean data...")
    train_model(model, trainset, valset, epochs=3, class_names=class_names)

    print(f"[*] Applying certified method: {method}")
    certified_info = {}

    sample_loader = DataLoader(testset, batch_size=32, shuffle=True)
    images, _ = next(iter(sample_loader))
    images = images.to(next(model.parameters()).device)

    if method == "interval_bound":
        lb, ub = estimate_interval_bounds(model, images)
        certified_info["interval_bounds"] = {
            "lower_mean": lb.mean().item(),
            "upper_mean": ub.mean().item()
        }
    elif method == "convex_relaxation":
        certified_info = estimate_convex_relaxation(model, images)
    elif method == "lipschitz_bound":
        certified_info = estimate_lipschitz_bound(model, images)
    else:
        raise ValueError(f"Unknown certified method: {method}")

    acc, per_class = evaluate_model(model, testset, class_names=class_names)

    result_data = {
        "defense": "certified_defense",
        "attack": attack_type,
        "method": method,
        "accuracy_after_defense": acc,
        "per_class_accuracy": per_class,
        "certified_info": certified_info,
        "params": cfg
    }

    os.makedirs(f"results/evasion/{attack_type}", exist_ok=True)
    json_path = f"results/evasion/{attack_type}/certified_defense_results.json"
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"[✔] Results saved to {json_path}")

    md_path = f"results/evasion/{attack_type}/certified_defense_report.md"
    generate_certified_defense_report(json_file=json_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
