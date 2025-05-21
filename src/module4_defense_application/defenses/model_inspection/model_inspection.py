import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from torch.nn import Module
from torch.utils.data import DataLoader, Subset

# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))

from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile
from backdoor_utils import simulate_static_patch_attack, simulate_learned_trigger_attack
from defenses.model_inspection.generate_model_inspection_report import generate_model_inspection_report


def inspect_layer_weights(model: Module, layer_names: list):
    stats = {}
    suspicious = []

    for name, param in model.named_parameters():
        if not any(layer in name for layer in layer_names):
            continue
        weights = param.detach().cpu().numpy().flatten()
        mean = np.mean(weights)
        std = np.std(weights)
        max_abs = np.max(np.abs(weights))
        num_large = np.sum(np.abs(weights) > 1.5)

        stats[name] = {
            "mean": float(mean),
            "std": float(std),
            "max_abs": float(max_abs),
            "num_large_weights": int(num_large)
        }

        if max_abs > 2.0 or std > 1.0:
            suspicious.append(name)

        os.makedirs("results/tmp_model_inspection_weights", exist_ok=True)
        plt.figure()
        plt.hist(weights, bins=50, alpha=0.7)
        plt.title(f"Layer: {name}")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"results/tmp_model_inspection_weights/{name.replace('.', '_')}_hist.png")
        plt.close()

    return suspicious, stats


def run_model_inspection_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Model Inspection defense for {attack_type}...")

    cfg = profile["defense_config"]["backdoor"][attack_type]["model_inspection"]
    layers_to_inspect = cfg.get("layers", [])

    model = load_model_cfg_from_profile(profile)

    if attack_type == "static_patch":
        poisoned_trainset, _, _ = simulate_static_patch_attack(profile, trainset, testset, class_names)
    elif attack_type == "learned_trigger":
        poisoned_trainset, _, _ = simulate_learned_trigger_attack(profile, trainset, testset, class_names)
    else:
        raise ValueError(f"Unsupported backdoor attack: {attack_type}")

    print("[*] Training model on poisoned dataset...")
    train_model(model, poisoned_trainset, valset, epochs=3, class_names=class_names)

    print(f"[*] Inspecting weights of layers: {', '.join(layers_to_inspect)}")
    suspicious_layers, stats = inspect_layer_weights(model, layers_to_inspect)

    acc, per_class = evaluate_model(model, testset, class_names=class_names)

    os.makedirs(f"results/backdoor/{attack_type}", exist_ok=True)
    result = {
        "defense": "model_inspection",
        "attack": attack_type,
        "layers_inspected": layers_to_inspect,
        "suspicious_layers": suspicious_layers,
        "layer_stats": stats,
        "accuracy_after_defense": acc,
        "per_class_accuracy": per_class,
        "params": cfg
    }

    json_path = f"results/backdoor/{attack_type}/model_inspection_results.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[✔] Results saved to {json_path}")

    md_path = f"results/backdoor/{attack_type}/model_inspection_report.md"
    generate_model_inspection_report(json_file=json_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
