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

    if results.get("accuracy_clean") is not None:
        lines.append(f"- **Clean Test Set Accuracy:** `{results['accuracy_clean']:.4f}`")
    else:
        lines.append("- **Clean Test Set Accuracy:** _Not available_")

    if results.get("asr_after_defense") is not None:
        lines.append(f"- **ASR After Defense:** `{results['asr_after_defense']:.4f}`")  # Alterado aqui
    else:
        lines.append("- **ASR After Defense:** _Not available_")  # Alterado aqui
    lines.append("")

    lines.append("### Per-Class Accuracy (Clean)")
    for cls, acc in results.get("per_class_accuracy_clean", {}).items():
        lines.append(f"- **{cls}**: `{acc:.4f}`")

    if results.get("per_original_class_asr"):
        lines.append("\n### Per-Original-Class ASR")
        for cls, asr in results["per_original_class_asr"].items():
            lines.append(f"- **Original Class {cls}**: `{asr:.4f}`")

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
