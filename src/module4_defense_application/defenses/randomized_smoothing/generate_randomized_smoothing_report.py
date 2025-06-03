import json


def generate_randomized_smoothing_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Randomized Smoothing Defense Report\n")
    lines.append(f"**Attack Evaluated:** {data.get('evaluated_attack', 'Unknown')}")
    lines.append(f"**Defense Method:** {data.get('defense_name', 'randomized_smoothing')}\n")

    lines.append("## Smoothing Parameters")
    lines.append(f"- **Sigma (noise std):** {data.get('sigma', 'N/A')}")
    lines.append(f"- **Number of Samples:** {data.get('num_samples', 'N/A')}\n")

    lines.append("## Evaluation Results")
    lines.append(f"- **Smoothed Accuracy on Clean Test Set:** {data.get('accuracy_clean', 0.0):.4f}")
    lines.append(f"- **Smoothed Accuracy on Adversarial Test Set:** {data.get('accuracy_adversarial', 0.0):.4f}\n")

    lines.append("### Per-Class Accuracy (Clean + Smoothed)")
    for cls, acc in data.get("per_class_accuracy_clean", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    lines.append("\n### Per-Class Accuracy (Adversarial + Smoothed)")
    for cls, acc in data.get("per_class_accuracy_adversarial", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
