import json
import os

def generate_per_class_monitoring_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Defense Report — Per-Class Monitoring\n")

    lines.append("## Overview\n")
    lines.append(f"- **Defense:** {data['defense']}")
    lines.append(f"- **Attack Targeted:** {data['attack']}")
    lines.append(f"- **Standard Deviation Threshold:** {data['threshold']}")
    lines.append(f"- **Mean Accuracy:** {data['accuracy_clean']:.4f}")
    lines.append(f"- **Std Dev of Accuracy:** {data['accuracy_std']:.4f}\n")

    lines.append("## Flagged Classes\n")
    if data["flagged_classes"]:
        lines.append("| Class | Accuracy |")
        lines.append("|-------|----------|")
        for item in data["flagged_classes"]:
            lines.append(f"| {item['class']} | {item['accuracy']:.4f} |")
    else:
        lines.append("No classes were flagged as anomalous.")

    lines.append("\n## Full Per-Class Accuracy\n")
    lines.append("| Class | Accuracy |")
    lines.append("|-------|----------|")
    for cls, acc in data["per_class_accuracy_clean"].items():
        lines.append(f"| {cls} | {acc:.4f} |")

    os.makedirs(os.path.dirname(md_file), exist_ok=True)
    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Per-class monitoring report generated at: {md_file}")