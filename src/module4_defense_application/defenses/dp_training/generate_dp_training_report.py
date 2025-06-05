import json

def generate_report(json_path: str, md_path: str):
    """Generate a Markdown report summarising DP‑Training results."""

    with open(json_path, "r") as f:
        data = json.load(f)

    lines = []
    lines.append("# Differential‑Privacy Training Report")
    lines.append("")

    lines.append(f"**Attack type:** `{data.get('attack')}`  ")
    lines.append(f"**Target ε (epsilon):** {data.get('epsilon_target')}  ")
    lines.append(f"**Target δ (delta):** {data.get('delta_target')}  ")
    lines.append(f"**Gradient clip‑norm:** {data.get('clip_norm')}  ")
    lines.append(f"**ε spent:** {data.get('epsilon_spent'):.3f}  ")
    lines.append("")

    lines.append("## Accuracy Metrics\n")
    lines.append(f"*Overall clean accuracy:* **{data.get('accuracy_clean'):.4f}**\n")

    # Per‑class accuracy table
    per_class = data.get("per_class_accuracy_clean", {})
    if per_class:
        lines.append("### Per‑Class Accuracy")
        lines.append("| Class | Accuracy |")
        lines.append("|:------|:--------:|")
        for cls, acc in per_class.items():
            lines.append(f"| {cls} | {acc:.4f} |")
        lines.append("")

    # save
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[✔] DP‑Training report written to {md_path}")
