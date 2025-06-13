import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))
from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile
from backdoor_utils import simulate_static_patch_attack, simulate_learned_trigger_attack,evaluate_backdoor_asr
from defenses.fine_pruning.generate_fine_pruning_report import generate_fine_pruning_report


def get_last_linear_layer(model):
    linear_layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    if not linear_layers:
        raise RuntimeError("No Linear layers found in model.")
    return linear_layers[-1]


def prune_neurons_by_activation(model, dataloader, layer_name, pruning_ratio):
    activations = []

    def hook(module, input, output):
        activations.append(output.detach().cpu())

    layer = dict(model.named_modules()).get(layer_name)
    if not layer:
        raise ValueError(f"Layer {layer_name} not found in model.")
    handle = layer.register_forward_hook(hook)

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            model(x)

    handle.remove()
    A = torch.cat(activations, dim=0)  # [N, D]
    mean_activations = A.mean(dim=0)
    num_neurons = A.shape[1]
    k = int(pruning_ratio * num_neurons)
    prune_indices = torch.topk(mean_activations, k, largest=False).indices.tolist()

    with torch.no_grad():
        weight = layer.weight.data
        weight[prune_indices] = 0.0
        if layer.bias is not None:
            layer.bias.data[prune_indices] = 0.0

    return prune_indices


def run_fine_pruning_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Fine-Pruning defense for {attack_type}...")

    cfg = profile["defense_config"]["backdoor"][attack_type]["fine_pruning"]

    attack_config = profile.get("attack_overrides", {}).get("backdoor", {}).get(attack_type, {})
    target_class = attack_config.get("target_class")

    if target_class is None:
        raise ValueError(f"Target class not found in profile for attack type '{attack_type}'. Cannot evaluate ASR.")


    pruning_ratio = cfg.get("pruning_ratio", 0.2)

    # === Step 1: Simulate attack ===
    model = load_model_cfg_from_profile(profile)
    if attack_type == "static_patch":
        poisoned_trainset, patched_testset, _ = simulate_static_patch_attack(profile, trainset, testset, class_names)
    elif attack_type == "learned_trigger":
        poisoned_trainset, patched_testset, _ = simulate_learned_trigger_attack(profile, trainset, testset, class_names)
    else:
        raise ValueError(f"Unsupported backdoor attack: {attack_type}")

    print("[*] Training backdoored model...")
    train_model(model, poisoned_trainset, valset, epochs=3, class_names=class_names)

    # === Step 2: Prune neurons from last layer ===
    print("[*] Pruning low-activation neurons from last layer...")
    dataloader = DataLoader(valset, batch_size=64, shuffle=False)
    last_layer = get_last_linear_layer(model)
    pruned_neurons = prune_neurons_by_activation(model, dataloader, last_layer, pruning_ratio)

    # === Step 3: Fine-tune model ===
    print("[*] Fine-tuning pruned model on clean training data...")
    train_model(model, trainset, valset=valset, epochs=3, class_names=class_names)

    # === Step 4: Evaluate ===
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


    # === Save ===
    os.makedirs(f"results/backdoor/{attack_type}", exist_ok=True)

    results = {
        "defense": "fine_pruning",
        "attack": attack_type,
        "accuracy_clean": acc_clean,
        "asr_after_defense": acc_adv,
        "per_class_accuracy_clean": per_class_clean,
        "per_original_class_asr": per_class_adv,
        "pruning_ratio": pruning_ratio,
        "pruned_layer": last_layer,
        "num_neurons_pruned": len(pruned_neurons),
        "pruned_neuron_indices": pruned_neurons,
        "params": cfg
    }

    result_path = f"results/backdoor/{attack_type}/fine_pruning_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[✔] Results saved to {result_path}")

    # === Generate Markdown report ===
    md_path = f"results/backdoor/{attack_type}/fine_pruning_report.md"
    generate_fine_pruning_report(json_file=result_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
