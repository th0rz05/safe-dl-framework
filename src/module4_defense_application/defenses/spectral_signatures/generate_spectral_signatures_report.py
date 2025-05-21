import json
import os

def generate_spectral_signatures_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Spectral Signatures Defense Report\n")
    lines.append(f"**Attack Type:** {data.get('attack', 'Unknown')}")
    lines.append(f"**Defense Method:** Spectral Signatures")
    lines.append(f"**Threshold Used:** {data['params'].get('threshold', 'N/A')}")
    lines.append(f"**Number of Removed Samples:** {data.get('num_removed', 0)}\n")

    # Accuracy section
    lines.append("## Accuracy After Defense\n")
    lines.append(f"- **Overall Accuracy:** {data.get('accuracy_after_defense', 0.0):.4f}")
    lines.append("\n### Per-Class Accuracy")
    for cls, acc in data.get("per_class_accuracy", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    # Histograms
    lines.append("\n## Spectral Histograms\n")
    lines.append("The following histograms illustrate the spectral signature magnitudes for each class.\n")

    attack_type = data.get("attack", "unknown_attack")
    hist_dir = f"results/backdoor/{attack_type}/spectral_histograms"
    if os.path.exists(hist_dir):
        for fname in sorted(os.listdir(hist_dir)):
            if fname.endswith(".png"):
                lines.append(f"### {fname.replace('_', ' ').replace('.png', '').capitalize()}")
                lines.append(f"![{fname}]({os.path.join('spectral_histograms', fname)})\n")

    # Removed samples
    lines.append("\n## Removed Examples\n")
    lines.append("The following examples were identified as suspicious and removed from the training set.\n")
    for ex in data.get("example_removed", [])[:5]:
        label = ex.get("original_label_name", ex.get("original_label", "Unknown"))
        path = ex.get("image_path", "")
        lines.append(f"**Label:** {label} — **Index:** {ex['index']}")
        lines.append(f"![Removed Example]({path})\n")

    # Write file
    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
