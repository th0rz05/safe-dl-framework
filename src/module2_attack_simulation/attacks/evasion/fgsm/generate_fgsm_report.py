import json
import os

def generate_fgsm_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Evasion Attack Report — FGSM\n")
    lines.append("## Overview\n")
    lines.append(f"- **Attack Type:** {data['attack_type']}")
    lines.append(f"- **Epsilon:** {data['epsilon']}\n")

    lines.append("## Performance Metrics\n")
    lines.append(f"- **Accuracy on Clean Test Set (CDA):** {data['accuracy_clean_testset']:.4f}")
    lines.append(f"- **Accuracy on Adversarial Test Set (ADA):** {data['accuracy_adversarial_testset']:.4f}\n")

    lines.append("### Per‑Class Accuracy (Clean Test Set)\n")
    lines.append("| Class | Accuracy |")
    lines.append("|-------|----------|")
    for cls, acc in data["per_class_clean"].items():
        lines.append(f"| {cls} | {acc:.4f} |")

    lines.append("\n### Per‑Class Accuracy (Adversarial Test Set)\n")
    lines.append("| Class | Accuracy |")
    lines.append("|-------|----------|")
    for cls, acc in data["per_class_adversarial"].items():
        lines.append(f"| {cls} | {acc:.4f} |")

    lines.append("\n## Example Adversarial Samples\n")
    lines.append('<div style="display: flex; gap: 10px;">')
    for sample in data["example_adversarial_samples"]:
        lines.append(
            f'<div style="text-align:center;"><small>{sample["example_image_path"]}</small><br>'
            f'<img src="{sample["example_image_path"]}" style="width: 120px;"></div>'
        )
    lines.append("</div>")

    os.makedirs(os.path.dirname(md_file), exist_ok=True)
    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] FGSM report generated at: {md_file}")
