import os
import json

def generate_spectral_signatures_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    acc_clean = data.get("accuracy_clean")
    acc_adv = data.get("accuracy_adversarial")
    per_class_clean = data.get("per_class_accuracy_clean", {})
    per_class_adv = data.get("per_class_accuracy_adversarial", {})
    num_removed = data.get("num_removed")
    removed_examples = data.get("example_removed", [])
    histogram_path = data.get("histogram_path")
    params = data.get("params", {})

    lines = [
        f"# Spectral Signatures Defense Report\n",
        f"**Attack Type:** {data.get('attack')}\n",
        f"**Defense:** {data.get('defense')}\n",
        "\n",
        f"## Defense Parameters\n",
    ]
    for k, v in params.items():
        lines.append(f"- `{k}`: {v}")

    lines.append("\n## Accuracy After Defense\n")
    lines.append(f"- **Clean Accuracy:** {acc_clean:.4f}" if acc_clean is not None else "- **Clean Accuracy:** N/A")
    lines.append(f"- **Adversarial Accuracy:** {acc_adv:.4f}" if acc_adv is not None else "- **Adversarial Accuracy:** N/A")

    lines.append("\n## Per-Class Accuracy (Clean)\n")
    for cls, acc in per_class_clean.items():
        lines.append(f"- {cls}: {acc:.4f}")

    if per_class_adv:
        lines.append("\n## Per-Class Accuracy (Adversarial)\n")
        for cls, acc in per_class_adv.items():
            lines.append(f"- {cls}: {acc:.4f}")

    lines.append("\n## Removed Samples Summary\n")
    lines.append(f"- **Total Removed:** {num_removed}")

    if histogram_path:
        lines.append("\n## Spectral Signature Histogram\n")
        lines.append(f"![Spectral Histogram](./{histogram_path})\n")

    if removed_examples:
        lines.append("\n## Examples of Removed Samples\n")
        for ex in removed_examples:
            lines.append(f"- **Index**: {ex['index']}, **Label**: {ex['original_label_name']}\n")
            lines.append(f"  ![Removed](./{ex['image_path']})\n")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))
