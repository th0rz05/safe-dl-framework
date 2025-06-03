import json

def generate_jpeg_preprocessing_report(json_file, md_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# JPEG Preprocessing Defense Report\n")
    lines.append(f"**Attack Type:** {data.get('attack', 'Unknown')}")
    lines.append("**Defense Method:** JPEG Preprocessing\n")

    params = data.get("params", {})
    lines.append(f"**JPEG Quality:** {params.get('quality', 'N/A')}\n")

    lines.append("## Accuracy After Defense\n")
    lines.append(f"- **Accuracy on Clean Images:** {data.get('jpeg_accuracy_clean', 0.0):.4f}")
    lines.append(f"- **Accuracy on Adversarial Images:** {data.get('jpeg_accuracy_adversarial', 0.0):.4f}\n")

    lines.append("### Per-Class Accuracy (Clean)")
    for cls, acc in data.get("per_class_accuracy_clean", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    lines.append("\n### Per-Class Accuracy (Adversarial)")
    for cls, acc in data.get("per_class_accuracy_adversarial", {}).items():
        lines.append(f"- **{cls}**: {acc:.4f}")

    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[\u2714] Report written to {md_file}")
