import json
import os

def generate_data_poisoning_report(json_file="results/data_poisoning_metrics.json", md_file="results/data_poisoning_report.md"):
    with open(json_file, "r") as f:
        results = json.load(f)

    lines = []

    lines.append("# Data Poisoning Attack Report\n")

    lines.append("## Overview\n")
    lines.append(f"- **Attack Type:** {results.get('attack_type')}")
    lines.append(f"- **Flip Rate:** {results.get('flip_rate')}")
    target_class = results.get("target_class")
    flip_to_class = results.get("flip_to_class")
    if target_class is not None and flip_to_class is not None:
        lines.append(f"- **Targeted Attack:** Yes (from {target_class} to {flip_to_class})")
    else:
        lines.append("- **Targeted Attack:** No (untargeted)")
    lines.append(f"- **Number of Flipped Samples:** {results.get('num_flipped')}")
    lines.append("")

    lines.append("## Performance Metrics\n")
    lines.append(f"- **Accuracy After Attack:** {results.get('accuracy_after_attack')}\n")

    lines.append("## Flip Summary\n")
    lines.append("| Original -> New | Count |")
    lines.append("|-----------------|-------|")
    for key, count in results.get("flipping_map", {}).items():
        lines.append(f"| {key} | {count} |")
    lines.append("")

    lines.append("## Example Flips\n")
    lines.append("| Index | Original Label | New Label |")
    lines.append("|--------|----------------|-----------|")
    for flip in results.get("example_flips", []):
        idx = flip["index"]
        orig = flip["original_label"]
        new = flip["new_label"]
        lines.append(f"| {idx} | {orig} | {new} |")
    lines.append("")

    lines.append("## Visual Flip Examples (first 5)\n")
    for i, flip in enumerate(results.get("example_flips", [])[:5]):
        orig = flip["original_label"]
        new = flip["new_label"]
        fname = f"flipped_samples/sample_{i}_{orig}_to_{new}.png"
        lines.append(f"![{orig}->{new}]({fname})")
    lines.append("")

    os.makedirs(os.path.dirname(md_file), exist_ok=True)
    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Markdown report generated at {md_file}")

if __name__ == "__main__":
    generate_report()
