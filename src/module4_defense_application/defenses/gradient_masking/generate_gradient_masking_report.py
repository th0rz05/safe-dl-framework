import json
import os

def generate_gradient_masking_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Gradient Masking Defense Report\n")
    lines.append(f"**Attack Type:** {data.get('attack', 'Unknown')}")
    lines.append("**Defense Method:** Gradient Masking")

    params = data.get("params", {})
    lines.append(f"**Masking Strength:** {params.get('strength', 'N/A')}")
    lines.append(f"**Masked Accuracy (Clean):** {data.get('masked_accuracy_clean', 0.0):.4f}")
    lines.append(f"**Masked Accuracy (Adversarial):** {data.get('masked_accuracy_adversarial', 0.0):.4f}\n")

    lines.append("## Per-Class Accuracy (Clean)")
    for cls, acc in data.get("per_class_accuracy_clean", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    lines.append("\n## Per-Class Accuracy (Adversarial)")
    for cls, acc in data.get("per_class_accuracy_adversarial", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Gradient masking report written to {md_file}")
