import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
from defenses.spectral_signatures.generate_spectral_signatures_report import generate_spectral_signatures_report


# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))
from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile
from backdoor_utils import simulate_static_patch_attack, simulate_learned_trigger_attack

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_layer_activations(model, dataloader, layer_name):
    activations = []
    indices = []

    def hook(module, input, output):
        activations.append(output.detach().cpu())

    target_layer = dict(model.named_modules()).get(layer_name)
    if not target_layer:
        raise ValueError(f"Layer {layer_name} not found in model.")

    handle = target_layer.register_forward_hook(hook)

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        for idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            _ = model(x)
            indices.extend(range(idx * x.size(0), idx * x.size(0) + x.size(0)))

    handle.remove()
    activations_tensor = torch.cat(activations, dim=0)
    return activations_tensor.numpy(), indices

def run_spectral_signatures_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Spectral Signatures defense for {attack_type}...")

    cfg = profile["defense_config"]["backdoor"][attack_type]["spectral_signatures"]
    threshold = cfg.get("threshold", 0.9)

    model = load_model_cfg_from_profile(profile)

    if attack_type == "static_patch":
        poisoned_trainset, _, _ = simulate_static_patch_attack(profile, trainset, testset, class_names)
    elif attack_type == "learned_trigger":
        poisoned_trainset, _, _ = simulate_learned_trigger_attack(profile, trainset, testset, class_names)
    else:
        raise ValueError(f"Unsupported backdoor attack: {attack_type}")

    train_model(model, poisoned_trainset, valset, epochs=3, class_names=class_names)

    linear_layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    if not linear_layers:
        raise RuntimeError("No Linear layers found in model.")
    layer_to_use = linear_layers[-2] if len(linear_layers) >= 2 else linear_layers[0]
    print(f"[*] Extracting activations from layer: {layer_to_use}")

    dataloader = DataLoader(poisoned_trainset, batch_size=64, shuffle=False)
    feats, indices = extract_layer_activations(model, dataloader, layer_to_use)

    label_list = [poisoned_trainset[i][1] for i in range(len(poisoned_trainset))]
    class_to_feats = defaultdict(list)
    class_to_indices = defaultdict(list)
    for feat, label, idx in zip(feats, label_list, indices):
        class_to_feats[label].append(feat)
        class_to_indices[label].append(idx)

    retained_indices = set()
    removed_indices = []

    output_dir = f"results/backdoor/{attack_type}/spectral_histograms"
    ensure_dir(output_dir)

    for cls in class_to_feats:
        X = np.stack(class_to_feats[cls])
        X_centered = X - X.mean(axis=0)

        svd = TruncatedSVD(n_components=1, random_state=0)
        svd.fit(X_centered)
        projections = svd.transform(X_centered).squeeze()

        abs_proj = np.abs(projections)
        cutoff = np.quantile(abs_proj, threshold)
        retained = [idx for idx, val in zip(class_to_indices[cls], abs_proj) if val <= cutoff]
        removed = [idx for idx, val in zip(class_to_indices[cls], abs_proj) if val > cutoff]

        retained_indices.update(retained)
        removed_indices.extend(removed)

        plt.figure()
        plt.hist(abs_proj, bins=40, color='blue', alpha=0.7)
        plt.axvline(cutoff, color='red', linestyle='--', label=f"Cutoff (q={threshold})")
        plt.title(f"Spectral Signatures - Class {class_names[cls]}")
        plt.xlabel("Projection magnitude")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"class_{cls}_hist.png"))
        plt.close()

    print(f"[✔] Removed {len(removed_indices)} suspected samples across all classes.")
    cleaned_trainset = Subset(poisoned_trainset, sorted(retained_indices))

    clean_model = load_model_cfg_from_profile(profile)
    train_model(clean_model, cleaned_trainset, valset, epochs=3, class_names=class_names)

    acc, per_class = evaluate_model(clean_model, testset, class_names=class_names)

    results = {
        "defense": "spectral_signatures",
        "attack": attack_type,
        "accuracy_after_defense": acc,
        "per_class_accuracy": per_class,
        "num_removed": len(removed_indices),
        "params": cfg
    }

    result_path = f"results/backdoor/{attack_type}/spectral_signatures_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[✔] Results saved to {result_path}")

    # Optional report generation
    md_path = f"results/backdoor/{attack_type}/spectral_signatures_report.md"
    generate_spectral_signatures_report(json_file=result_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
