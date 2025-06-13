import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from defenses.anomaly_detection.generate_anomaly_detection_report import generate_anomaly_detection_report

# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))
from attacks.utils import train_model, evaluate_model, load_model_cfg_from_profile
from backdoor_utils import simulate_static_patch_attack, simulate_learned_trigger_attack,evaluate_backdoor_asr

def extract_features(model, dataloader, layer_name):
    features = []
    indices = []

    def hook(module, input, output):
        features.append(output.detach().cpu())

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
    return torch.cat(features, dim=0).numpy(), indices

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

def run_anomaly_detection_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Anomaly Detection defense for {attack_type}...")

    cfg = profile["defense_config"]["backdoor"][attack_type]["anomaly_detection"]

    attack_config = profile.get("attack_overrides", {}).get("backdoor", {}).get(attack_type, {})
    target_class = attack_config.get("target_class")

    if target_class is None:
        raise ValueError(f"Target class not found in profile for attack type '{attack_type}'. Cannot evaluate ASR.")


    method = cfg.get("method", "isolation_forest")
    contamination = cfg.get("contamination", 0.1)

    model = load_model_cfg_from_profile(profile)

    if attack_type == "static_patch":
        poisoned_trainset, patched_testset, trigger_info = simulate_static_patch_attack(profile, trainset, testset, class_names)
    elif attack_type == "learned_trigger":
        poisoned_trainset, patched_testset, trigger_info = simulate_learned_trigger_attack(profile, trainset, testset, class_names)
    else:
        raise ValueError(f"Unsupported backdoor attack: {attack_type}")

    train_model(model, poisoned_trainset, valset=valset, epochs=3, class_names=class_names)

    layer_to_use = [name for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)][-1]
    print(f"[*] Extracting features from layer: {layer_to_use}")
    loader = DataLoader(poisoned_trainset, batch_size=64, shuffle=False)
    feats, indices = extract_features(model, loader, layer_to_use)

    print(f"[*] Running anomaly detection using {method}...")
    if method == "isolation_forest":
        detector = IsolationForest(contamination=contamination, random_state=0)
    elif method == "lof":
        detector = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    else:
        raise ValueError("Unknown anomaly detection method.")

    if method == "lof":
        preds = detector.fit_predict(feats)
    else:
        detector.fit(feats)
        preds = detector.predict(feats)

    removed_indices = [idx for idx, p in zip(indices, preds) if p == -1]
    retained_indices = [idx for idx, p in zip(indices, preds) if p != -1]

    print(f"[✔] Removed {len(removed_indices)} anomalous samples")
    cleaned_trainset = Subset(poisoned_trainset, retained_indices)

    clean_model = load_model_cfg_from_profile(profile)
    print("[*] Retraining model on cleaned dataset...")
    train_model(clean_model, cleaned_trainset, valset=valset, epochs=3, class_names=class_names)

    acc_clean, per_class_clean = evaluate_model(clean_model, testset, class_names=class_names)
    if patched_testset is not None:
        acc_adv, per_class_adv = evaluate_backdoor_asr(
            clean_model,
            patched_testset,
            target_class=target_class, # Pass the target_class
            class_names=class_names,
            prefix="[Eval ASR after Defense]"
        )
    else:
        acc_adv, per_class_adv = None, None

    os.makedirs(f"results/backdoor/{attack_type}", exist_ok=True)
    example_log = save_removed_examples(poisoned_trainset.dataset, removed_indices,
                                        output_dir=f"results/backdoor/{attack_type}/anomaly_removed",
                                        class_names=class_names, max_examples=5)

    results = {
        "defense": "anomaly_detection",
        "attack": attack_type,
        "accuracy_clean": acc_clean,
        "asr_after_defense": acc_adv,
        "per_class_accuracy_clean": per_class_clean,
        "per_original_class_asr": per_class_adv,
        "num_removed": len(removed_indices),
        "removed_indices": removed_indices,
        "example_removed": example_log,
        "params": cfg
    }

    result_path = f"results/backdoor/{attack_type}/anomaly_detection_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results saved to {result_path}")

    md_path = f"results/backdoor/{attack_type}/anomaly_detection_report.md"
    generate_anomaly_detection_report(json_file=result_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
