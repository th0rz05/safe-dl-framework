import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from sklearn.cluster import KMeans
from collections import defaultdict

# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))
from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile
from backdoor_utils import simulate_static_patch_attack, simulate_learned_trigger_attack

from defenses.activation_clustering.generate_activation_clustering_report import generate_activation_clustering_report


def extract_layer_activations(model, dataloader, layer_name):
    """
    Extract activations from a given layer using a forward hook.
    Returns activations and their corresponding dataset indices.
    """
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


def run_activation_clustering_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Activation Clustering defense for {attack_type}...")

    # Load parameters
    cfg = profile["defense_config"]["backdoor"][attack_type]["activation_clustering"]
    num_clusters = cfg.get("num_clusters", 2)

    # Load model
    model = load_model_cfg_from_profile(profile)

    # Simulate backdoor attack
    if attack_type == "static_patch":
        poisoned_trainset, patched_testset, trigger_info = simulate_static_patch_attack(profile, trainset, testset, class_names)
    elif attack_type == "learned_trigger":
        poisoned_trainset, patched_testset, trigger_info = simulate_learned_trigger_attack(profile, trainset, testset, class_names)
    else:
        raise ValueError(f"Unsupported backdoor attack: {attack_type}")

    # Train model on poisoned dataset
    train_model(model, poisoned_trainset, valset, epochs=3, class_names=class_names)

    # Detect last Linear layer
    linear_layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    if not linear_layers:
        raise RuntimeError("No Linear layers found in model.")
    layer_to_use = linear_layers[-1]
    print(f"[*] Extracting activations from layer: {layer_to_use}")

    # Extract activations from poisoned training set
    dataloader = DataLoader(poisoned_trainset, batch_size=64, shuffle=False)
    feats, indices = extract_layer_activations(model, dataloader, layer_to_use)

    # Clustering
    print("[*] Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(feats)
    labels = kmeans.labels_

    # Identify cluster sizes and remove the smallest
    cluster_counts = defaultdict(int)
    for lbl in labels:
        cluster_counts[lbl] += 1
    smallest_cluster = min(cluster_counts, key=cluster_counts.get)

    retained_indices = [idx for idx, lbl in zip(indices, labels) if lbl != smallest_cluster]
    removed_indices = [idx for idx, lbl in zip(indices, labels) if lbl == smallest_cluster]

    print(f"[✔] Removed {len(removed_indices)} samples from suspected cluster {smallest_cluster}")

    cleaned_trainset = Subset(poisoned_trainset, retained_indices)

    # Retrain model
    print("[*] Retraining model on cleaned dataset...")
    clean_model = load_model_cfg_from_profile(profile)
    train_model(clean_model, cleaned_trainset, valset=valset, epochs=3, class_names=class_names)

    # Evaluate
    acc, per_class = evaluate_model(clean_model, testset, class_names=class_names)

    # Save results
    os.makedirs(f"results/backdoor/{attack_type}", exist_ok=True)
    results = {
        "defense": "activation_clustering",
        "attack": attack_type,
        "accuracy_after_defense": acc,
        "per_class_accuracy": per_class,
        "num_removed": len(removed_indices),
        "removed_indices": removed_indices,
        "params": cfg
    }

    result_path = f"results/backdoor/{attack_type}/activation_clustering_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results saved to {result_path}")

    # Generate Markdown report
    md_path = f"results/backdoor/{attack_type}/activation_clustering_report.md"
    generate_activation_clustering_report(json_file=result_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
