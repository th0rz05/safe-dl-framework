import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
from defenses.activation_clustering.generate_activation_clustering_report import generate_activation_clustering_report

# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))
from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile
from backdoor_utils import simulate_static_patch_attack, simulate_learned_trigger_attack



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


def save_removed_examples(dataset, removed_indices, output_dir, class_names=None, max_examples=5):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    saved = 0
    example_log = []

    for idx in removed_indices:
        try:
            img, label = dataset[idx]
            label_name = class_names[label] if class_names else str(label)

            img = img.cpu()
            if img.dim() == 3 and img.shape[0] in [3, 4]:
                img = img.permute(1, 2, 0)
            img = img.squeeze()

            filename = f"removed_{idx}_{label}.png"
            save_path = os.path.join(output_dir, filename)

            plt.figure()
            plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
            plt.axis("off")
            plt.title(f"Removed: {label_name}")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            example_log.append({
                "index": idx,
                "original_label": label,
                "original_label_name": label_name,
                "image_path": os.path.join(os.path.basename(output_dir), filename)
            })

            saved += 1
            if saved >= max_examples:
                break
        except Exception as e:
            print(f"[!] Failed to save example {idx}: {e}")

    return example_log


def run_activation_clustering_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Activation Clustering defense for {attack_type}...")

    cfg = profile["defense_config"]["backdoor"][attack_type]["activation_clustering"]
    num_clusters = cfg.get("num_clusters", 2)

    model = load_model_cfg_from_profile(profile)

    if attack_type == "static_patch":
        poisoned_trainset, patched_testset, trigger_info = simulate_static_patch_attack(profile, trainset, testset, class_names)
    elif attack_type == "learned_trigger":
        poisoned_trainset, patched_testset, trigger_info = simulate_learned_trigger_attack(profile, trainset, testset, class_names)
    else:
        raise ValueError(f"Unsupported backdoor attack: {attack_type}")

    train_model(model, poisoned_trainset, valset, epochs=3, class_names=class_names)

    linear_layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    if not linear_layers:
        raise RuntimeError("No Linear layers found in model.")
    layer_to_use = linear_layers[-1]
    print(f"[*] Extracting activations from layer: {layer_to_use}")

    dataloader = DataLoader(poisoned_trainset, batch_size=64, shuffle=False)
    feats, indices = extract_layer_activations(model, dataloader, layer_to_use)

    print("[*] Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(feats)
    labels = kmeans.labels_

    cluster_counts = defaultdict(int)
    for lbl in labels:
        cluster_counts[lbl] += 1
    smallest_cluster = min(cluster_counts, key=cluster_counts.get)

    retained_indices = [idx for idx, lbl in zip(indices, labels) if lbl != smallest_cluster]
    removed_indices = [idx for idx, lbl in zip(indices, labels) if lbl == smallest_cluster]

    print(f"[✔] Removed {len(removed_indices)} samples from suspected cluster {smallest_cluster}")

    cleaned_trainset = Subset(poisoned_trainset, retained_indices)

    clean_model = load_model_cfg_from_profile(profile)
    print("[*] Retraining model on cleaned dataset...")
    train_model(clean_model, cleaned_trainset, valset=valset, epochs=3, class_names=class_names)

    # Accuracy in clean test set
    acc_clean, per_class_clean = evaluate_model(clean_model, testset, class_names=class_names)

    # Accuracy in adversarial test set (if exists)
    adv_path = f"module2_attack_simulation/results/backdoor/{attack_type}/adv_testset.pt"
    if os.path.exists(adv_path):
        adv_testset = torch.load(adv_path)
        loader_adv = DataLoader(adv_testset, batch_size=64, shuffle=False)
        acc_adv, per_class_adv = evaluate_model(clean_model, loader_adv, class_names=class_names)
    else:
        acc_adv, per_class_adv = None, None

    os.makedirs(f"results/backdoor/{attack_type}", exist_ok=True)
    example_log = save_removed_examples(poisoned_trainset.dataset, removed_indices,
                                        output_dir=f"results/backdoor/{attack_type}/activation_removed",
                                        class_names=class_names, max_examples=5)

    results = {
        "defense": "activation_clustering",
        "attack": attack_type,
        "accuracy_clean": acc_clean,
        "accuracy_adversarial": acc_adv,
        "per_class_accuracy_clean": per_class_clean,
        "per_class_accuracy_adversarial": per_class_adv,
        "num_removed": len(removed_indices),
        "removed_indices": removed_indices,
        "example_removed": example_log,
        "params": cfg
    }

    result_path = f"results/backdoor/{attack_type}/activation_clustering_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results saved to {result_path}")

    md_path = f"results/backdoor/{attack_type}/activation_clustering_report.md"
    generate_activation_clustering_report(json_file=result_path, md_file=md_path)
