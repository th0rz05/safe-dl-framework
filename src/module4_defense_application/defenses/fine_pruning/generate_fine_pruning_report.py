import json
import os

def generate_fine_pruning_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Fine-Pruning Defense Report\n")
    lines.append(f"**Attack Type:** {data.get('attack', 'Unknown')}")
    lines.append(f"**Defense Method:** Fine-Pruning")
    lines.append(f"**Pruning Ratio:** {data['params'].get('pruning_ratio', 'N/A')}")
    lines.append(f"**Number of Removed Neurons:** {data.get('num_removed_neurons', 'N/A')}\n")

    lines.append("## Accuracy After Defense\n")
    lines.append(f"- **Overall Accuracy:** {data.get('accuracy_after_defense', 0.0):.4f}")

    lines.append("\n### Per-Class Accuracy")
    for cls, acc in data.get("per_class_accuracy", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    lines.append("\n## Visual Examples of Removed Samples")
    lines.append("The following are examples of poisoned samples removed by fine-pruning (if applicable):\n")

    for example in data.get("example_removed", []):
        lines.append(f"- **Index {example['index']}** — Class: {example['original_label_name']}")
        lines.append(f"![removed_example]({example['image_path']})\n")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
