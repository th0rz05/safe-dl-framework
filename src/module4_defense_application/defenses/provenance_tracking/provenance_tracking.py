import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Attach module-2 utilities to path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "module2_attack_simulation"
        )
    )
)
from attacks.utils import evaluate_model, load_model_cfg_from_profile, train_model
from attacks.data_poisoning.clean_label.run_clean_label import poison_dataset as apply_clean_label

# -----------------------------------------------------------------------------
# Loss computation helpers with tqdm
# -----------------------------------------------------------------------------

def compute_sample_losses(model: torch.nn.Module,
                          dataset,
                          batch_size: int = 64,
                          device: torch.device = None) -> dict:
    if isinstance(dataset, Subset):
        index_map = dataset.indices
    else:
        index_map = list(range(len(dataset)))

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    model.eval().to(dev)

    losses = {}
    for batch_id, (xs, ys) in enumerate(tqdm(loader, desc="[Sample Loss Eval]", leave=False)):
        xs, ys = xs.to(dev), ys.to(dev)
        batch_loss = criterion(model(xs), ys).detach().cpu().numpy()
        start = batch_id * batch_size
        for i, loss_val in enumerate(batch_loss):
            orig_idx = index_map[start + i]
            losses[int(orig_idx)] = float(loss_val)
    return losses


def compute_batch_losses(model: torch.nn.Module,
                         dataset,
                         batch_size: int = 64,
                         device: torch.device = None) -> list:
    if isinstance(dataset, Subset):
        index_map = dataset.indices
    else:
        index_map = list(range(len(dataset)))

    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    model.eval().to(dev)

    batch_stats = []
    for batch_id, (xs, ys) in enumerate(tqdm(loader, desc="[Batch Loss Eval]", leave=False)):
        xs, ys = xs.to(dev), ys.to(dev)
        losses = criterion(model(xs), ys)
        avg_loss = float(losses.mean().cpu().item())
        start = batch_id * batch_size
        batch_indices = [index_map[start + i] for i in range(len(losses))]
        batch_stats.append((batch_indices, avg_loss))
    return batch_stats

# -----------------------------------------------------------------------------
# Save removed examples
# -----------------------------------------------------------------------------

def save_removed_examples(dataset, removed_indices, output_dir, class_names=None, max_examples=5):
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    example_list = []

    for file in os.listdir(output_dir):
        path = os.path.join(output_dir, file)
        if os.path.isfile(path):
            os.remove(path)

    for idx in removed_indices:
        try:
            img, label = dataset[idx]
            label_name = class_names[label] if class_names else str(label)

            img = img.cpu()
            if img.shape[0] in [3, 4]:
                img = img.permute(1, 2, 0)
            img = img.squeeze()

            plt.figure()
            plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
            plt.axis("off")
            plt.title(f"Removed: {label_name}")
            filename = os.path.join(output_dir, f"removed_{idx}_{label}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            example_list.append({
                "index": idx,
                "original_label": int(label),
                "original_label_name": label_name,
                "image_path": f"provenance_examples/removed_{idx}_{label}.png"
            })

            saved += 1
            if saved >= max_examples:
                break
        except Exception as e:
            print(f"[!] Failed to save removed sample {idx}: {e}")

    return example_list

# -----------------------------------------------------------------------------
# Runner for provenance tracking defense
# -----------------------------------------------------------------------------

def run_provenance_tracking_defense(
    profile: dict,
    trainset,
    testset,
    valset,
    class_names,
    attack_type: str
):
    print(f"[*] Running Provenance-Tracking defense for attack '{attack_type}'...")

    cfg = profile["defense_config"]["data_poisoning"][attack_type]["provenance_tracking"]
    granularity = cfg.get("granularity", "sample")
    batch_size = cfg.get("batch_size", 64)

    atk_cfg = profile.get("attack_overrides", {}).get("data_poisoning", {}).get("clean_label", {})
    profile_name = profile.get("name", "default")

    model_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..","module2_attack_simulation","saved_models", f"{profile_name}_clean_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[!] Model file not found at {model_path}. Make sure it was trained and saved.")

    # Load the correct model architecture based on the profile
    helper = load_model_cfg_from_profile(profile)

    # Load the saved weights
    helper.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    helper.eval()
    print(f"[✔] Loaded helper from {model_path}")
    poisoned_ds, _, _ = apply_clean_label(
        trainset,
        atk_cfg.get("fraction_poison", 0.05),
        atk_cfg.get("target_class"),
        atk_cfg.get("perturbation_method", "overlay"),
        atk_cfg.get("epsilon", 0.1),
        atk_cfg.get("max_iterations", 100),
        atk_cfg.get("source_selection", "random"),
        class_names,
        helper
    )

    model = load_model_cfg_from_profile(profile)
    if granularity == "sample":
        losses = compute_sample_losses(model, poisoned_ds, batch_size=batch_size)
        arr = np.array(list(losses.values()))
        mean, std = arr.mean(), arr.std()
        to_remove = [idx for idx, l in losses.items() if l > mean + 2 * std]
    elif granularity == "batch":
        stats = compute_batch_losses(model, poisoned_ds, batch_size=batch_size)
        losses_arr = np.array([l for _, l in stats])
        m, s = losses_arr.mean(), losses_arr.std()
        outlier_batches = [idxs for idxs, l in stats if abs(l - m) > 2 * s]
        to_remove = [i for batch in outlier_batches for i in batch]
    else:
        raise ValueError(f"Unknown granularity '{granularity}'")

    retained = [i for i in range(len(poisoned_ds)) if i not in to_remove]
    cleaned_ds = Subset(poisoned_ds, retained)
    print(f"[✔] Removed {len(to_remove)} samples out of {len(poisoned_ds)}")

    # Save visual examples
    output_dir = os.path.join("results", "data_poisoning", attack_type, "provenance_examples")
    example_flips = save_removed_examples(poisoned_ds.dataset, to_remove, output_dir, class_names, max_examples=5)

    model_clean = load_model_cfg_from_profile(profile)
    train_model(
        model_clean, cleaned_ds, valset,
        epochs=profile.get("training", {}).get("epochs", 3),
        class_names=class_names
    )
    final_acc, per_class = evaluate_model(
        model_clean, testset, class_names=class_names
    )

    out_dir = os.path.join("results", "data_poisoning", attack_type)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "provenance_tracking_results.json")
    results = {
        "defense": "provenance_tracking",
        "attack": attack_type,
        "granularity": granularity,
        "num_removed": len(to_remove),
        "clean_accuracy": final_acc,
        "per_class_accuracy": per_class,
        "example_removed": example_flips
    }
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results JSON saved at {json_path}")

    try:
        from generate_provenance_tracking_report import generate_report
        md_path = os.path.join(out_dir, "provenance_tracking_report.md")
        generate_report(json_path, md_path)
        print(f"[✔] Report generated at {md_path}")
    except ImportError:
        print("[!] Report generator not found; skipping Markdown report.")
