import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from tqdm.auto import tqdm

# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "module2_attack_simulation")))

from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile
from attacks.data_poisoning.clean_label.run_clean_label import poison_dataset as apply_clean_label


def compute_grad_influence(model, dataset, batch_size=64, device=None):
    """
    Estimate influence using gradient inner product: ∇loss(x)ᵀ ∇loss(all)
    Returns: dict {idx: influence_score}
    """
    model.eval()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    influences = {}
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size)

    for i, (x, y) in enumerate(tqdm(loader, desc="[Grad Influence]", leave=False)):
        x, y = x.to(dev), y.to(dev)
        for j in range(x.size(0)):
            model.zero_grad()
            out = model(x[j].unsqueeze(0))
            loss = criterion(out, y[j].unsqueeze(0))
            grad = torch.autograd.grad(loss, model.parameters(), retain_graph=False, create_graph=False)
            score = sum(torch.sum(g**2).item() for g in grad)
            global_idx = i * batch_size + j
            influences[global_idx] = score

    return influences


def save_removed_examples(dataset, indices, output_dir, class_names=None, max_examples=5):
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        try:
            os.remove(os.path.join(output_dir, f))
        except Exception:
            pass

    saved, examples = 0, []
    for idx in indices:
        img, label = dataset[idx]
        label_name = class_names[label] if class_names else str(label)
        img = img.cpu()
        if img.shape[0] in [3, 4]:
            img = img.permute(1, 2, 0)
        img = img.squeeze()

        path = os.path.join(output_dir, f"removed_{idx}_{label}.png")
        plt.figure()
        plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
        plt.axis("off")
        plt.title(f"Removed: {label_name}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        examples.append({
            "index": idx,
            "original_label": int(label),
            "original_label_name": label_name,
            "image_path": f"influence_examples/removed_{idx}_{label}.png"
        })

        saved += 1
        if saved >= max_examples:
            break

    return examples


def run_influence_functions_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Influence Functions defense for attack '{attack_type}'...")

    cfg = profile["defense_config"]["data_poisoning"][attack_type]["influence_functions"]
    method = cfg.get("method", "grad_influence")
    sample_size = cfg.get("sample_size", 500)

    profile_name = profile.get("name", "default")
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "module2_attack_simulation", "saved_models", f"{profile_name}_clean_model.pth")

    model = load_model_cfg_from_profile(profile)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Poisoned dataset generation (only clean_label supported)
    if attack_type == "clean_label":
        atk_cfg = profile["attack_overrides"]["data_poisoning"]["clean_label"]
        poisoned_ds, _, _ = apply_clean_label(
            trainset,
            atk_cfg.get("fraction_poison", 0.05),
            atk_cfg.get("target_class"),
            atk_cfg.get("perturbation_method", "overlay"),
            atk_cfg.get("epsilon", 0.1),
            atk_cfg.get("max_iterations", 100),
            atk_cfg.get("source_selection", "random"),
            class_names,
            model
        )
    else:
        raise NotImplementedError("Influence functions only support clean_label for now.")

    # Compute influence
    if method == "grad_influence":
        scores = compute_grad_influence(model, poisoned_ds)
    else:
        raise NotImplementedError("Only grad_influence is implemented.")

    values = np.array(list(scores.values()))
    mean, std = values.mean(), values.std()
    to_remove = [idx for idx, score in scores.items() if score > mean + 2 * std]
    retained = [i for i in range(len(poisoned_ds)) if i not in to_remove]
    cleaned_ds = Subset(poisoned_ds, retained)

    # Save removed
    output_dir = os.path.join("results", "data_poisoning", attack_type, "influence_examples")
    removed_examples = save_removed_examples(poisoned_ds.dataset, to_remove, output_dir, class_names)

    # Retrain and evaluate
    model_clean = load_model_cfg_from_profile(profile)
    train_model(model_clean, cleaned_ds, valset, epochs=3, class_names=class_names)
    acc, per_class = evaluate_model(model_clean, testset, class_names=class_names)

    results = {
        "defense": "influence_functions",
        "attack": attack_type,
        "method": method,
        "sample_size": sample_size,
        "num_removed": len(to_remove),
        "accuracy_clean": acc,
        "per_class_accuracy_clean": per_class,
        "example_removed": removed_examples
    }

    out_dir = os.path.join("results", "data_poisoning", attack_type)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "influence_functions_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results JSON saved at {json_path}")

    try:
        from defenses.influence_functions.generate_influence_report import generate_report
        md_path = os.path.join(out_dir, "influence_functions_report.md")
        generate_report(json_path, md_path)
        print(f"[✔] Report generated at {md_path}")
    except ImportError:
        print("[!] Report generator not found.")
