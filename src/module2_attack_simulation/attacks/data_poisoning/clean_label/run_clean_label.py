import os
import json
import random
from copy import deepcopy
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from attacks.utils import (
    train_model,
    evaluate_model,
    get_class_labels
)

from attacks.data_poisoning.clean_label.generate_clean_label_report import generate_clean_label_report

# === Helper function to save poisoned examples ===
def save_poisoned_examples(dataset, poison_log, num_examples=5, class_names=None):
    os.makedirs("results/data_poisoning/clean_label/examples", exist_ok=True)

    #delete existing images in the directory
    for filename in os.listdir("results/data_poisoning/clean_label/examples"):
        file_path = os.path.join("results/data_poisoning/clean_label/examples", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for entry in poison_log[:num_examples]:
        image = entry["tensor"]  # Now we use the tensor directly

        if image.max() <= 1.0:
            image = image * 255
        image = image.to(torch.uint8)

        if image.shape[0] in [1, 3]:  # CHW to HWC
            image = image.permute(1, 2, 0)

        image_np = image.numpy()
        if image_np.shape[-1] == 1:
            image_np = image_np.squeeze(-1)

        plt.imsave(entry["example_image_path"], image_np)


def extract_features(model, x, no_grad=True):
    model.eval()
    if no_grad:
        with torch.no_grad():
            if hasattr(model, "features"):
                return model.features(x.unsqueeze(0)).squeeze(0)
            elif hasattr(model, "forward_features"):
                return model.forward_features(x.unsqueeze(0)).squeeze(0)
            else:
                return model(x.unsqueeze(0)).squeeze(0)
    else:
        if hasattr(model, "features"):
            return model.features(x.unsqueeze(0)).squeeze(0)
        elif hasattr(model, "forward_features"):
            return model.forward_features(x.unsqueeze(0)).squeeze(0)
        else:
            return model(x.unsqueeze(0)).squeeze(0)

# === Perturbation logic ===
def apply_perturbation(image, method="overlay", epsilon=0.1, model=None, target_image=None, max_iterations=100):
    if method == "overlay":
        image = image.clone()
        _, H, W = image.shape
        patch_size = int(min(H, W) * 0.2)

        x_start = W - patch_size
        y_start = H - patch_size

        alpha = 0.5
        image[:, y_start:H, x_start:W] = (
            alpha * image[:, y_start:H, x_start:W] +
            (1 - alpha) * torch.ones_like(image[:, y_start:H, x_start:W])
        )
        return torch.clamp(image, 0, 1)

    elif method == "noise":
        noise = torch.randn_like(image) * epsilon
        return torch.clamp(image + noise, 0, 1)

    elif method == "feature_collision":
        if model is None or target_image is None:
            raise ValueError("Model and target_image required for feature_collision.")

        # Clone, detach and enable gradient tracking
        x = image.clone().detach().requires_grad_()

        # Compute fixed target features without tracking
        target_features = extract_features(model, target_image, no_grad=True).detach()

        optimizer = torch.optim.SGD([x], lr=epsilon)
        loss_fn = torch.nn.MSELoss()

        for _ in range(max_iterations):
            optimizer.zero_grad()
            features = extract_features(model, x, no_grad=False)  # forward pass with grad
            loss = loss_fn(features, target_features)
            loss.backward()
            optimizer.step()
            x.data = torch.clamp(x.data, 0, 1)

        return x.detach()

    else:
        raise ValueError(f"Unknown perturbation method: {method}")



def poison_dataset(dataset, fraction_poison, target_class, method, epsilon, max_iterations, source_selection, class_names, model=None):
    poisoned_dataset = deepcopy(dataset)
    targets = poisoned_dataset.dataset.targets
    indices = poisoned_dataset.indices

    class_counts = defaultdict(int)
    poisoned_indices = []
    poison_log = []

    # Decide which samples are eligible to be poisoned
    if target_class is None:
        eligible_indices = indices  # Untargeted
    else:
        eligible_indices = [i for i in indices if targets[i] == target_class]  # Targeted

    # If not feature_collision, source_selection should be random
    if method in ["overlay", "noise"] and source_selection != "random":
        print(f"[!] Warning: source_selection '{source_selection}' ignored for '{method}'. Using 'random' instead.")
        source_selection = "random"

    # For feature_collision, source selection based on confidence
    if method == "feature_collision" and source_selection != "random":
        if model is None:
            raise ValueError("Model is required for source_selection other than 'random' with feature_collision.")

        print(f"[*] Computing confidence scores for source selection: {source_selection}")
        model.eval()
        confidences = []

        with torch.no_grad():
            for idx in tqdm(eligible_indices, desc="Scoring confidence"):
                x, _ = poisoned_dataset.dataset[idx]
                x = x.unsqueeze(0)
                logits = model(x)
                prob = torch.softmax(logits, dim=1)
                confidence = prob.max().item()
                confidences.append((idx, confidence))

        # Sort by confidence
        confidences.sort(key=lambda x: x[1], reverse=(source_selection == "most_confident"))
        selected_indices = [idx for idx, _ in confidences[:int(len(confidences) * fraction_poison)]]

    else:
        num_to_poison = int(len(eligible_indices) * fraction_poison)
        selected_indices = random.sample(eligible_indices, num_to_poison)

    for idx in selected_indices:
        original_image, label = poisoned_dataset.dataset[idx]

        # Feature collision needs a target image
        target_image = None
        if method == "feature_collision":
            other_class_indices = [i for i in indices if targets[i] != label]
            target_idx = random.choice(other_class_indices)
            target_image, _ = poisoned_dataset.dataset[target_idx]

        perturbed_image = apply_perturbation(
            original_image,
            method=method,
            epsilon=epsilon,
            model=model,
            target_image=target_image,
            max_iterations=max_iterations
        )

        poison_log.append({
            "index": idx,
            "original_label": int(label),
            "original_label_name": class_names[int(label)],
            "perturbation_norm": float(torch.norm(perturbed_image - original_image, p=2).item()),
            "example_image_path": f"results/data_poisoning/clean_label/examples/poison_{idx}_{class_names[int(label)]}.png",
            "tensor": perturbed_image.detach().cpu().clone()
        })
        poisoned_indices.append(idx)
        class_counts[class_names[int(label)]] += 1

    return poisoned_dataset, poison_log, class_counts

def run_clean_label(trainset, testset, valset, model, profile, class_names):
    print("[*] Running clean label attack from profile configuration...")

    attack_cfg = profile.get("attack_overrides", {}).get("data_poisoning", {}).get("clean_label", {})

    fraction_poison = attack_cfg.get("fraction_poison", 0.05)
    target_class = attack_cfg.get("target_class")
    method = attack_cfg.get("perturbation_method", "overlay")
    max_iterations = attack_cfg.get("max_iterations", 100)
    epsilon = attack_cfg.get("epsilon", 0.1)
    source_selection = attack_cfg.get("source_selection", "random")

    classes = get_class_labels(trainset)

    if target_class is not None:
        print(f"[*] Poisoning only class: {target_class} ({class_names[target_class]})")
    else:
        print("[*] Untargeted mode: poisoning across all classes")

    poisoned_trainset, poison_log, poison_stats = poison_dataset(
        trainset,
        fraction_poison,
        target_class,
        method,
        epsilon,
        max_iterations,
        source_selection,
        class_names,
        model=model
    )

    print("[*] Training model on poisoned dataset...")
    train_model(model, poisoned_trainset, valset, epochs=6)

    acc, per_class_accuracy = evaluate_model(model, testset, class_names=class_names)

    os.makedirs("results/data_poisoning/clean_label/examples", exist_ok=True)
    save_poisoned_examples(poisoned_trainset.dataset, poison_log, num_examples=5, class_names=class_names)

    #remove tensor from poison_log to save space
    for entry in poison_log:
        del entry["tensor"]

    result = {
        "attack_type": "clean_label",
        "perturbation_method": method,
        "fraction_poison": fraction_poison,
        "target_class": target_class,
        "max_iterations": max_iterations,
        "epsilon": epsilon,
        "source_selection": source_selection,
        "num_poisoned_samples": len(poison_log),
        "num_total_samples": len(trainset),
        "accuracy_after_attack": acc,
        "per_class_accuracy": per_class_accuracy,
        "example_poisoned_samples": poison_log[:5]
    }

    with open("results/data_poisoning/clean_label/clean_label_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("[✔] Clean label attack metrics saved to results/data_poisoning/clean_label/clean_label_metrics.json")

    generate_clean_label_report(
        json_file="results/data_poisoning/clean_label/clean_label_metrics.json",
        md_file="results/data_poisoning/clean_label/clean_label_report.md"
    )

    print("[✔] Clean label report generation")
