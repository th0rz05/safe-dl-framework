import os
import json

def generate_model_inspection_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Model Inspection Defense Report\n")
    lines.append(f"**Attack Type:** {data.get('attack', 'Unknown')}")
    lines.append("**Defense Method:** Model Inspection")
    lines.append(f"**Inspected Layers:** {', '.join(data.get('inspected_layers', []))}")
    lines.append(f"**Threshold:** {data.get('threshold', 'N/A')}")
    lines.append(f"**Number of Suspect Neurons:** {data.get('num_suspect_neurons', 0)}\n")

    lines.append("## Accuracy After Defense\n")
    lines.append(f"- **Overall Accuracy:** {data.get('accuracy_after_defense', 0.0):.4f}")

    lines.append("\n### Per-Class Accuracy")
    for cls, acc in data.get("per_class_accuracy", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    lines.append("\n## Activation Histograms")
    lines.append("The following histograms visualize the activations of inspected neurons. Neurons with extreme outlier values were flagged as suspicious.\n")

    attack_type = data.get("attack", "unknown_attack")
    hist_dir = f"results/backdoor/{attack_type}/inspection_histograms"
    if os.path.exists(hist_dir):
        for fname in sorted(os.listdir(hist_dir)):
            if fname.endswith(".png"):
                layer_name = fname.replace(".png", "")
                lines.append(f"### {layer_name}")
                lines.append(f"![{layer_name}]({os.path.join('inspection_histograms', fname)})\n")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
