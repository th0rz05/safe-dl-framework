import json
import os

def generate_fine_pruning_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Fine-Pruning Defense Report\n")
    lines.append(f"**Attack Type:** {data.get('attack', 'Unknown')}")
    lines.append(f"**Defense Method:** Fine-Pruning\n")

    lines.append("## Pruning Details\n")
    lines.append(f"- **Pruning Ratio:** {data.get('pruning_ratio', 'N/A')}")
    lines.append(f"- **Pruned Layer:** {data.get('pruned_layer', 'N/A')}")
    lines.append(f"- **Number of Neurons Pruned:** {data.get('num_neurons_pruned', 'N/A')}")

    pruned_indices = data.get("pruned_neuron_indices", [])
    if pruned_indices:
        lines.append(f"- **Pruned Neuron Indices:** {', '.join(map(str, pruned_indices))}")
    lines.append("")

    lines.append("## Accuracy After Defense\n")
    lines.append(f"- **Accuracy on Clean Test Set:** {data.get('accuracy_clean', 0.0):.4f}")
    lines.append(f"- **Accuracy on Adversarial Test Set:** {data.get('accuracy_adversarial', 0.0):.4f}")

    lines.append("\n### Per-Class Accuracy (Clean)")
    for cls, acc in data.get("per_class_accuracy_clean", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    lines.append("\n### Per-Class Accuracy (Adversarial)")
    for cls, acc in data.get("per_class_accuracy_adversarial", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    lines.append("")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
