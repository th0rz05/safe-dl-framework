import json
import os

def generate_model_inspection_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Model Inspection Defense Report\n")
    lines.append(f"**Attack Type:** {data.get('attack', 'Unknown')}")
    lines.append("**Defense Method:** Model Inspection")

    layers = data.get('layers_inspected', [])
    suspicious = data.get("suspicious_layers", [])

    lines.append(f"**Inspected Layers:** {', '.join(layers) if layers else 'N/A'}")
    lines.append(f"**Suspicious Layers Detected:** {len(suspicious)}")
    if suspicious:
        lines.append(f"- {', '.join(suspicious)}")
    lines.append("")

    # === Clean Evaluation ===
    lines.append("## Clean Accuracy After Defense\n")
    lines.append(f"- **Overall Accuracy:** {data.get('accuracy_clean', 0.0):.4f}")
    lines.append("\n### Per-Class Accuracy (Clean)")
    for cls, acc in data.get("per_class_accuracy_clean", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    # === Adversarial Evaluation ===
    if data.get("accuracy_adversarial") is not None:
        lines.append("\n## Adversarial Accuracy After Defense")
        lines.append(f"- **Overall Accuracy:** {data.get('accuracy_adversarial', 0.0):.4f}")
        lines.append("\n### Per-Class Accuracy (Adversarial)")
        for cls, acc in data.get("per_class_accuracy_adversarial", {}).items():
            lines.append(f"- **{cls}**: {acc:.4f}")

    # === Histograms ===
    lines.append("\n## Weight Histograms")
    lines.append("The following histograms visualize the weight distributions of the inspected layers.\n")

    attack_type = data.get("attack", "unknown_attack")
    hist_dir = f"results/backdoor/{attack_type}/inspection_histograms"

    if os.path.exists(hist_dir):
        for fname in sorted(os.listdir(hist_dir)):
            if fname.endswith(".png"):
                layer_name = fname.replace(".png", "").replace("_", " ")
                lines.append(f"### {layer_name}")
                lines.append(f"![{layer_name}]({os.path.join('inspection_histograms', fname)})\n")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Report written to {md_file}")
