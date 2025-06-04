import json
import matplotlib.pyplot as plt
import os

def generate_pruning_report(json_file, md_file):
    with open(json_file, "r") as f:
        results = json.load(f)

    acc_clean = results.get("accuracy_clean", None)
    acc_adv = results.get("accuracy_adversarial", None)
    per_class_clean = results.get("per_class_accuracy_clean", {})
    per_class_adv = results.get("per_class_accuracy_adversarial", {})
    hist_path = results.get("histogram_path", None)

    with open(md_file, "w") as f:
        f.write("# Pruning Defense Report\n\n")
        f.write(f"**Attack type:** `{results.get('attack')}`\n\n")
        f.write(f"**Pruning ratio:** `{results['params'].get('pruning_ratio', 'N/A')}`\n\n")
        f.write(f"**Pruned Parameters Fraction:** `{results.get('pruned_params_fraction', 'N/A')}`\n\n")

        if acc_clean is not None:
            f.write(f"**Accuracy on clean test set:** `{acc_clean:.4f}`\n\n")
        if acc_adv is not None:
            f.write(f"**Accuracy on adversarial test set:** `{acc_adv:.4f}`\n\n")

        if per_class_clean:
            f.write("## Per-Class Accuracy (Clean)\n\n")
            f.write("| Class | Accuracy |\n")
            f.write("|-------|----------|\n")
            for cls, acc in per_class_clean.items():
                f.write(f"| {cls} | {acc:.4f} |\n")
            f.write("\n")

        if per_class_adv:
            f.write("## Per-Class Accuracy (Adversarial)\n\n")
            f.write("| Class | Accuracy |\n")
            f.write("|-------|----------|\n")
            for cls, acc in per_class_adv.items():
                f.write(f"| {cls} | {acc:.4f} |\n")
            f.write("\n")

        if hist_path and os.path.exists(os.path.join("results", hist_path)):
            f.write("## Weight Histogram\n\n")
            f.write(f"![Histogram]({hist_path})\n")
        elif hist_path:
            f.write("## Weight Histogram\n\n")
            f.write(f"*Histogram saved at: `{hist_path}` (image file not found when writing report)*\n")
