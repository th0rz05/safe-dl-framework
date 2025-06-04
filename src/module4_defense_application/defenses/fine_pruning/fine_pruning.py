import os
import sys
import json
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn

from torch.utils.data import DataLoader
from collections import defaultdict
from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile
from defenses.fine_pruning.generate_fine_pruning_report import generate_fine_pruning_report

# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))
from backdoor_utils import simulate_static_patch_attack, simulate_learned_trigger_attack

def get_last_linear_layer(model):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            return name, module
    raise ValueError("No linear layer found in model.")

def fine_prune_neurons(layer, ratio=0.2):
    with torch.no_grad():
        weights = layer.weight.abs().sum(dim=1)  # sum over input dim
        num_neurons = weights.size(0)
        num_to_prune = int(num_neurons * ratio)
        _, indices = torch.topk(weights, num_to_prune, largest=False)

        mask = torch.ones_like(weights)
        mask[indices] = 0
        layer.weight[mask == 0] = 0
        if layer.bias is not None:
            layer.bias[mask == 0] = 0

    return indices.tolist(), num_to_prune, ratio

def run_fine_pruning_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Fine-Pruning defense for {attack_type}...")

    cfg = profile["defense_config"]["backdoor"][attack_type]["fine_pruning"]
    pruning_ratio = cfg.get("amount", 0.2)

    model = load_model_cfg_from_profile(profile)

    if attack_type == "static_patch":
        poisoned_trainset, patched_testset, trigger_info = simulate_static_patch_attack(profile, trainset, testset, class_names)
    elif attack_type == "learned_trigger":
        poisoned_trainset, patched_testset, trigger_info = simulate_learned_trigger_attack(profile, trainset, testset, class_names)
    else:
        raise ValueError(f"Unsupported backdoor attack: {attack_type}")

    print("[*] Training model on clean dataset before pruning...")
    train_model(model, trainset, valset=valset, epochs=3, class_names=class_names)

    print("[*] Identifying last linear layer for fine pruning...")
    last_layer_name, last_layer = get_last_linear_layer(model)

    print(f"[*] Pruning neurons in layer: {last_layer_name} with ratio = {pruning_ratio}")
    pruned_neurons, num_pruned, _ = fine_prune_neurons(last_layer, ratio=pruning_ratio)

    print("[*] Retraining model after pruning...")
    train_model(model, trainset, valset=valset, epochs=3, class_names=class_names)

    acc_clean, per_class_clean = evaluate_model(model, testset, class_names=class_names)
    if patched_testset is not None:
        acc_adv, per_class_adv = evaluate_model(model, patched_testset, class_names=class_names)
    else:
        acc_adv, per_class_adv = None, None

    results = {
        "defense": "fine_pruning",
        "attack": attack_type,
        "accuracy_clean": acc_clean,
        "accuracy_adversarial": acc_adv,
        "per_class_accuracy_clean": per_class_clean,
        "per_class_accuracy_adversarial": per_class_adv,
        "pruned_layer": last_layer_name,
        "num_neurons_pruned": num_pruned,
        "pruned_neuron_indices": pruned_neurons,
        "pruning_ratio": pruning_ratio,
        "params": cfg
    }

    os.makedirs(f"results/backdoor/{attack_type}", exist_ok=True)
    result_path = f"results/backdoor/{attack_type}/fine_pruning_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results saved to {result_path}")

    md_path = f"results/backdoor/{attack_type}/fine_pruning_report.md"
    generate_fine_pruning_report(json_file=result_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
