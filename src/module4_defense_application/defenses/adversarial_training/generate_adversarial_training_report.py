import json
import os

def generate_adversarial_training_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Adversarial Training Defense Report\n")
    lines.append(f"**Attack Evaluated:** {data.get('evaluated_attack', 'Unknown')}")
    lines.append(f"**Defense Method:** {data.get('defense_name', 'Adversarial Training')}\n")

    params = data.get("parameters", {})
    lines.append("## Training Parameters")
    lines.append(f"- **Base Attack Used for Training:** {params.get('base_attack_used_for_training', 'N/A')}")
    lines.append(f"- **Epsilon:** {params.get('epsilon', 'N/A')}")
    lines.append(f"- **Mixed with Clean Samples:** {params.get('mixed_with_clean', False)}\n")

    lines.append("## Evaluation Results\n")
    lines.append(f"- **Clean Test Accuracy:** {data.get('clean_test_accuracy', 0.0):.4f}")
    lines.append(f"- **Adversarial Test Accuracy:** {data.get('adversarial_test_accuracy', 0.0):.4f}\n")

    lines.append("### Per-Class Accuracy (Clean)")
    for cls, acc in data.get("per_class_clean_accuracy", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    lines.append("\n### Per-Class Accuracy (Adversarial)")
    for cls, acc in data.get("per_class_adversarial_accuracy", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
