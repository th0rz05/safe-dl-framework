import os
import sys
import json
import torch
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile
from torch.utils.data import DataLoader
from defenses.pruning.generate_pruning_report import generate_pruning_report

# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))
from backdoor_utils import simulate_static_patch_attack, simulate_learned_trigger_attack,evaluate_backdoor_asr

def apply_pruning(model, amount=0.5):
    pruned = 0
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            pruned += torch.sum(module.weight_mask == 0).item()
            total += module.weight_mask.numel()
    return pruned / total if total > 0 else 0.0


def run_pruning_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Pruning defense for {attack_type}...")

    cfg = profile["defense_config"]["backdoor"][attack_type]["pruning"]

    attack_config = profile.get("attack_overrides", {}).get("backdoor", {}).get(attack_type, {})
    target_class = attack_config.get("target_class")

    if target_class is None:
        raise ValueError(f"Target class not found in profile for attack type '{attack_type}'. Cannot evaluate ASR.")


    amount = cfg.get("amount", 0.5)

    model = load_model_cfg_from_profile(profile)

    if attack_type == "static_patch":
        poisoned_trainset, patched_testset, trigger_info = simulate_static_patch_attack(profile, trainset, testset, class_names)
    elif attack_type == "learned_trigger":
        poisoned_trainset, patched_testset, trigger_info = simulate_learned_trigger_attack(profile, trainset, testset, class_names)
    else:
        raise ValueError(f"Unsupported backdoor attack: {attack_type}")

    print("[*] Training model on poisoned dataset...")
    train_model(model, poisoned_trainset, valset=valset, epochs=3, class_names=class_names)

    print(f"[*] Applying pruning with amount = {amount}...")
    pruned_fraction = apply_pruning(model, amount)
    print(f"[✔] Fraction of parameters pruned: {pruned_fraction:.4f}")


    acc_clean, per_class_clean = evaluate_model(model, testset, class_names=class_names)

    # Evaluate Attack Success Rate (ASR) on adversarial test set (patched)
    if patched_testset is not None:
        # Use the new ASR evaluation function
        acc_adv, per_class_adv = evaluate_backdoor_asr(
            model,
            patched_testset,
            target_class=target_class,  # Pass the target_class
            class_names=class_names,
            prefix="[Eval ASR after Defense]"  # Um prefixo mais descritivo
        )
    else:
        acc_adv, per_class_adv = None, None

    results = {
        "defense": "pruning",
        "attack": attack_type,
        "accuracy_clean": acc_clean,
        "asr_after_defense": acc_adv,
        "per_class_accuracy_clean": per_class_clean,
        "per_original_class_asr": per_class_adv,
        "pruned_params_fraction": round(pruned_fraction, 4),
        "params": cfg
    }

    result_path = f"results/backdoor/{attack_type}/pruning_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results saved to {result_path}")

    md_path = f"results/backdoor/{attack_type}/pruning_report.md"
    generate_pruning_report(json_file=result_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
