import json
import os


def generate_anomaly_detection_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Anomaly Detection Defense Report\n")
    lines.append(f"**Attack Type:** {data.get('attack', 'Unknown')}")
    lines.append(f"**Defense Method:** Anomaly Detection ({data.get('method', 'N/A')})")
    lines.append(f"**Number of Removed Samples:** {data.get('num_removed', 0)}\n")

    lines.append("## Accuracy After Defense\n")
    lines.append(f"- **Overall Accuracy:** {data.get('accuracy_after_defense', 0.0):.4f}\n")

    lines.append("### Per-Class Accuracy")
    for cls, acc in data.get("per_class_accuracy", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")
    lines.append("")

    if data.get("example_removed"):
        lines.append("## Visual Examples of Removed Samples\n")
        lines.append("The following are examples of samples removed by the anomaly detection method.\n")
        for ex in data["example_removed"]:
            orig = ex.get("original_label_name", str(ex.get("original_label", "?")))
            lines.append(f"**Removed Sample — Label: {orig}**\n")
            lines.append(f"![removed]({ex['image_path']})\n")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
