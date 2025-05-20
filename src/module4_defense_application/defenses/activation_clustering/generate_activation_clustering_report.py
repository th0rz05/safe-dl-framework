import os
import json

def generate_activation_clustering_report(json_file, md_file):
    with open(json_file, "r") as f:
        results = json.load(f)

    lines = []

    lines.append(f"# Activation Clustering Report – {results['attack'].replace('_', ' ').title()}\n")

    lines.append("## 1. Overview")
    lines.append(f"- **Defense:** {results['defense']}")
    lines.append(f"- **Attack Type:** {results['attack']}")
    lines.append(f"- **Number of Removed Samples:** {results['num_removed']}")
    lines.append(f"- **Defense Parameters:**")
    for key, val in results.get("params", {}).items():
        lines.append(f"  - `{key}`: {val}")
    lines.append("")

    lines.append("## 2. Accuracy After Defense")
    lines.append(f"- **Overall Accuracy:** `{results['accuracy_after_defense']:.4f}`\n")

    lines.append("### Per-Class Accuracy")
    for cls, acc in results["per_class_accuracy"].items():
        lines.append(f"- **{cls}**: `{acc:.4f}`")
    lines.append("")

    example_removed = results.get("example_removed", [])
    if example_removed:
        lines.append("## 3. Removed Sample Examples (Cluster-based)\n")
        for ex in example_removed:
            label = ex.get("original_label_name", str(ex.get("original_label", "")))
            lines.append(f"**Removed Sample — Class: {label}**\n")
            lines.append(f"![Removed]({ex['image_path']})\n")
    else:
        lines.append("## 3. Removed Sample Examples\n")
        lines.append("_No visual examples were saved._\n")

    # Save Markdown report
    os.makedirs(os.path.dirname(md_file), exist_ok=True)
    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report generated at {md_file}")
