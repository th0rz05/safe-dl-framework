import json
import os

def generate_pruning_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Pruning Defense Report\n")
    lines.append(f"**Attack Type:** {data.get('attack', 'Unknown')}")
    lines.append(f"**Defense Method:** Structured Pruning")
    lines.append(f"**Scope:** {data['params'].get('scope', 'N/A')}")
    lines.append(f"**Pruning Ratio:** {data['params'].get('pruning_ratio', 'N/A')}")
    lines.append(f"**Number of Pruned Layers:** {len(data.get('pruned_layers', []))}")
    lines.append("**Pruned Layers:** " + ", ".join(data.get("pruned_layers", [])) + "\n")

    lines.append("## Accuracy After Defense\n")
    lines.append(f"- **Overall Accuracy:** {data.get('accuracy_after_defense', 0.0):.4f}")

    lines.append("\n### Per-Class Accuracy")
    for cls, acc in data.get("per_class_accuracy", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
