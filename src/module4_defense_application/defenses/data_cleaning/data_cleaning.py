import os
import sys
import json
from torch.utils.data import Subset
import torch
import numpy as np
import matplotlib.pyplot as plt
from defenses.data_cleaning.generate_data_cleaning_report import generate_data_cleaning_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..","..", "module2_attack_simulation")))
from attacks.utils import train_model, evaluate_model, get_class_labels, load_model
from attacks.data_poisoning.label_flipping.run_label_flipping import flip_labels  # Only for label_flipping
from dataset_loader import unnormalize, get_normalization_params



def save_cleaned_examples(dataset, removed_indices, output_dir="results/data_poisoning/cleaned_examples", class_names=None, max_examples=5,profile = None):
    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    #remove old files
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"[!] Failed to remove file {file_path}: {e}")

    mean, std = get_normalization_params(profile['dataset']['name']) if profile else (None, None)

    for idx in removed_indices:
        try:
            img, label = dataset[idx]
            label_name = class_names[label] if class_names else str(label)

            if mean is not None and std is not None:
                img = unnormalize(img.cpu(), mean, std).clamp(0, 1)
            else:
                img = img.cpu().clamp(0, 1)

            if img.shape[0] in [3, 4]:  # RGB or RGBA
                img = img.permute(1, 2, 0)  # [H, W, C]
            img = img.squeeze()

            plt.figure()
            plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
            plt.axis("off")
            plt.title(f"Removed: {label_name}")
            filename = os.path.join(output_dir, f"removed_{idx}_{label}.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()

            saved += 1
            if saved >= max_examples:
                break
        except Exception as e:
            print(f"[!] Failed to save removed sample {idx}: {e}")


def apply_data_cleaning(trainset,model, params):
    """
    Apply data cleaning strategies to remove suspected poisoned or noisy samples.

    Args:
    ----
    - trainset: Subset or Dataset (must have 'loss' accessible if needed)
    - params: dict with keys 'method' and 'threshold'

    Returns:
    ----
    - A filtered Subset or Dataset with potentially noisy samples removed
    """

    print(f"[*] Applying data cleaning with method={params['method']} and threshold={params['threshold']}")

    method = params['method']
    threshold = params['threshold']
    retained_indices = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    loader = torch.utils.data.DataLoader(trainset, batch_size=64)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    losses = []
    all_indices = []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            batch_loss = criterion(logits, y)
            losses.extend(batch_loss.cpu().numpy())

            start_idx = i * loader.batch_size
            end_idx = start_idx + x.size(0)
            batch_indices = trainset.indices[start_idx:end_idx] if isinstance(trainset, Subset) else list(range(start_idx, end_idx))
            all_indices.extend(batch_indices)

    if method == "loss_filtering":
        mean_loss = sum(losses) / len(losses)
        for idx, loss in zip(all_indices, losses):
            if loss <= mean_loss * threshold:
                retained_indices.append(idx)

    elif method == "outlier_detection":
        loss_array = np.array(losses)
        mean_loss = np.mean(loss_array)
        std_loss = np.std(loss_array)

        z_scores = (loss_array - mean_loss) / std_loss
        retained_indices = [idx for idx, z in zip(all_indices, z_scores) if abs(z) <= threshold]

    else:
        raise ValueError(f"Unknown data cleaning method: {method}")

    print(f"[✔] Retained {len(retained_indices)} out of {len(all_indices)} samples.")
    return Subset(trainset.dataset, retained_indices) if isinstance(trainset, Subset) else Subset(trainset, retained_indices)


def run_data_cleaning_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running data_cleaning defense for {attack_type}...")

    # Load defense config for this attack
    defense_cfg = profile["defense_config"]["data_poisoning"][attack_type]["data_cleaning"]

    # Load model for training after cleaning
    model = load_model("label_flipping_model",profile,path = "../module2_attack_simulation/saved_models/")

    # Generate poisoned dataset depending on the attack type
    if attack_type == "label_flipping":
        attack_cfg = profile["attack_overrides"]["data_poisoning"]["label_flipping"]
        poisoned_trainset, flip_log, flip_map = flip_labels(
            trainset,
            flip_rate=attack_cfg["flip_rate"],
            strategy=attack_cfg["strategy"],
            source_class=attack_cfg.get("source_class"),
            target_class=attack_cfg.get("target_class"),
            class_names=class_names
        )
    else:
        raise NotImplementedError(f"Data cleaning for attack '{attack_type}' not implemented yet.")


    # Apply cleaning
    cleaned_dataset = apply_data_cleaning(poisoned_trainset, model, defense_cfg)

    removed_indices = list(set(poisoned_trainset.indices) - set(cleaned_dataset.indices))
    print(f"[*] Saving up to 5 removed examples ({len(removed_indices)} identified)...")
    save_cleaned_examples(poisoned_trainset.dataset, removed_indices,
                          output_dir=f"results/data_poisoning/{attack_type}/cleaned_examples",
                          class_names=class_names, max_examples=5,profile =  profile)

    # Train final model on cleaned data
    model = train_model(model, cleaned_dataset, valset, epochs=1, class_names=class_names)

    # Evaluate
    acc, per_class = evaluate_model(model, testset, class_names=class_names)

    # Save metrics
    os.makedirs(f"results/data_poisoning/{attack_type}", exist_ok=True)

    example_flips = []
    for idx in removed_indices[:5]:
        img, label = poisoned_trainset.dataset[idx]
        label_name = class_names[label] if class_names else str(label)
        path = f"cleaned_examples/removed_{idx}_{label}.png"
        example_flips.append({
            "index": idx,
            "original_label": label,
            "original_label_name": label_name,
            "image_path": path
        })

    results = {
        "defense": "data_cleaning",
        "attack": attack_type,
        "accuracy_clean": acc,
        "per_class_accuracy_clean": per_class,
        "cleaning_params": defense_cfg,
        "num_removed": len(removed_indices),
        "example_removed": example_flips
    }

    path = f"results/data_poisoning/{attack_type}/data_cleaning_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[✔] Defense results saved to {path}")

    # Generate Markdown report
    md_path = f"results/data_poisoning/{attack_type}/data_cleaning_report.md"
    generate_data_cleaning_report(json_file=path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
