import os
import json
import numpy as np
from pathlib import Path
from defenses.per_class_monitoring.generate_per_class_monitoring_report import generate_per_class_monitoring_report


def run_per_class_monitoring_defense(profile, attack_type):
    print(f"[*] Running per_class_monitoring defense for {attack_type}...")

    # Get threshold from profile config
    defense_cfg = profile["defense_config"]["data_poisoning"][attack_type]["per_class_monitoring"]
    threshold = defense_cfg["std_threshold"]

    # Load attack metrics file
    json_path = os.path.join("..", "module2_attack_simulation", "results",
                             "data_poisoning", attack_type, f"{attack_type}_metrics.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[!] Attack metrics file not found: {json_path}")

    with open(json_path, "r") as f:
        attack_data = json.load(f)

    per_class_acc = attack_data["per_class_accuracy"]
    acc_values = np.array(list(per_class_acc.values()))
    acc_mean = acc_values.mean()
    acc_std = acc_values.std()

    flagged = []
    for cls, acc in per_class_acc.items():
        if acc < acc_mean - threshold * acc_std:
            flagged.append({"class": cls, "accuracy": acc})

    print(f"[*] Flagged {len(flagged)} suspicious classes out of {len(per_class_acc)}")

    results = {
        "defense": "per_class_monitoring",
        "attack": attack_type,
        "accuracy_clean": round(acc_mean, 4),
        "accuracy_adversarial": None,
        "per_class_accuracy_clean": per_class_acc,
        "per_class_accuracy_adversarial": None,
        "accuracy_std": round(acc_std, 4),
        "threshold": threshold,
        "flagged_classes": flagged,
    }

    output_dir = f"results/data_poisoning/{attack_type}"
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "per_class_monitoring_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    out_md = os.path.join(output_dir, "per_class_monitoring_report.md")
    generate_per_class_monitoring_report(out_json, out_md)
