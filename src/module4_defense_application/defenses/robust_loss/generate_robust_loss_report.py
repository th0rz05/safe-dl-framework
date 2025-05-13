import json

def generate_report(json_path: str, md_path: str):
    """
    Load results from JSON and generate a Markdown report for the Robust Loss defense.
    """
    # Load the JSON results
    with open(json_path, 'r') as f:
        data = json.load(f)

    attack = data.get('attack', 'unknown')
    loss_type = data.get('type', 'unknown')
    clean_acc = data.get('clean_accuracy', None)
    per_class = data.get('per_class_accuracy', {})

    # Start building the report lines
    lines = []
    lines.append(f"# Robust Loss Defense Report")
    lines.append("")
    lines.append(f"**Attack type:** `{attack}`  ")
    lines.append(f"**Robust loss type:** `{loss_type}`  ")

    # List configuration parameters
    config_keys = set(data.keys()) - {'attack', 'defense', 'type', 'clean_accuracy', 'per_class_accuracy'}
    if config_keys:
        lines.append("## Defense Configuration")
        lines.append("")
        for key in sorted(config_keys):
            value = data[key]
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    # Overall metric
    lines.append("## Overall Performance")
    lines.append("")
    if clean_acc is not None:
        lines.append(f"- **Clean accuracy:** {clean_acc:.4f}")
    else:
        lines.append("- Clean accuracy: N/A")
    lines.append("")

    # Per-class accuracies
    if per_class:
        lines.append("## Per-Class Accuracy")
        lines.append("")
        lines.append("| Class | Accuracy |")
        lines.append("|:------|:--------:|")
        for cls, acc in per_class.items():
            lines.append(f"| {cls} | {acc:.4f} |")
        lines.append("")

    # Write out the Markdown file
    with open(md_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"[✔] Robust Loss report generated at {md_path}")
