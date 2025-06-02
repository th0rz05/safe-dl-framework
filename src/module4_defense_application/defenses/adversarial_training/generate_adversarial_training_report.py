import json
import os

def generate_adversarial_training_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Adversarial Training Defense Report\n")
    lines.append(f"**Attack Type:** {data.get('attack', 'Unknown')}")
    lines.append("**Defense Method:** Adversarial Training")

    params = data.get("params", {})
    lines.append(f"**Base Attack Used:** {params.get('base_attack', 'N/A')}")
    lines.append(f"**Epsilon:** {params.get('epsilon', 'N/A')}")
    lines.append(f"**Mix Clean:** {params.get('mix_clean', False)}")
    lines.append(f"**Robust Accuracy:** {data.get('robust_accuracy', 0.0):.4f}\n")

    lines.append("## Accuracy After Defense\n")
    lines.append(f"- **Overall Accuracy:** {data.get('accuracy_after_defense', 0.0):.4f}\n")

    lines.append("### Per-Class Accuracy")
    for cls, acc in data.get("per_class_accuracy", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
