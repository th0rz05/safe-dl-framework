import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import Subset

# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))

from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile
from backdoor_utils import simulate_static_patch_attack, simulate_learned_trigger_attack
from defenses.pruning.generate_pruning_report import generate_pruning_report


def apply_pruning(model, ratio, scope="last_layer_only"):
    pruned_layers = []
    if scope == "last_layer_only":
        # Última camada linear
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise RuntimeError("No Linear layers found in the model.")
        target_layer = linear_layers[-1]
        prune.ln_structured(target_layer, name="weight", amount=ratio, n=1, dim=0)
        pruned_layers.append(target_layer)
    elif scope == "all_layers":
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.ln_structured(module, name="weight", amount=ratio, n=1, dim=0)
                pruned_layers.append(module)
    else:
        raise ValueError(f"Unknown pruning scope: {scope}")
    return pruned_layers


def run_pruning_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Pruning defense for {attack_type}...")

    # Carregar config
    cfg = profile["defense_config"]["backdoor"][attack_type]["pruning"]
    pruning_ratio = cfg.get("pruning_ratio", 0.2)
    scope = cfg.get("scope", "last_layer_only")

    # Simular ataque e treinar modelo
    model = load_model_cfg_from_profile(profile)
    if attack_type == "static_patch":
        poisoned_trainset, _, _ = simulate_static_patch_attack(profile, trainset, testset, class_names)
    elif attack_type == "learned_trigger":
        poisoned_trainset, _, _ = simulate_learned_trigger_attack(profile, trainset, testset, class_names)
    else:
        raise ValueError(f"Unsupported backdoor attack: {attack_type}")

    print("[*] Training model on poisoned dataset...")
    train_model(model, poisoned_trainset, valset, epochs=3, class_names=class_names)

    # Aplicar pruning
    print(f"[*] Applying structured pruning (ratio={pruning_ratio}, scope={scope})...")
    pruned_layers = apply_pruning(model, pruning_ratio, scope)

    # Avaliar
    print("[*] Evaluating pruned model...")
    acc, per_class = evaluate_model(model, testset, class_names=class_names)

    # Guardar resultados
    os.makedirs(f"results/backdoor/{attack_type}", exist_ok=True)
    result_path = f"results/backdoor/{attack_type}/pruning_results.json"
    results = {
        "defense": "pruning",
        "attack": attack_type,
        "accuracy_after_defense": acc,
        "per_class_accuracy": per_class,
        "params": cfg,
        "pruned_layers": [str(layer.__class__.__name__) for layer in pruned_layers]
    }
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results saved to {result_path}")

    # Gerar relatório
    md_path = f"results/backdoor/{attack_type}/pruning_report.md"
    generate_pruning_report(json_file=result_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
