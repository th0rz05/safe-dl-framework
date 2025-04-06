import json
import os

def generate_data_poisoning_report(json_file="results/data_poisoning_metrics.json", md_file="results/data_poisoning_report.md"):
    with open(json_file, "r") as f:
        results = json.load(f)

    lines = []
    lines.append("# Data Poisoning Attack Report\n")

    lines.append("## Overview\n")
    lines.append(f"- **Attack Type:** {results.get('attack_type')}")
    lines.append(f"- **Strategy:** {results.get('flipping_strategy')}")
    lines.append(f"- **Flip Rate:** {results.get('flip_rate')}")
    if results.get("flipping_strategy") == "one_to_one":
        lines.append(f"- **Source Class:** {results.get('source_class')}")
        lines.append(f"- **Target Class:** {results.get('target_class')}")
    else:
        lines.append(f"- **Target Class:** {results.get('target_class')}")
        lines.append("- **Source Classes:** All except target")
    lines.append(f"- **Number of Flipped Samples:** {results.get('num_flipped')}")
    lines.append("")

    lines.append("## Performance Metrics\n")
    lines.append(f"- **Accuracy After Attack:** {results.get('accuracy_after_attack')}\n")

    lines.append("## Flip Summary\n")
    lines.append("| Original -> New | Count |")
    lines.append("|------------------|--------|")
    for key, count in results.get("flipping_map", {}).items():
        lines.append(f"| {key} | {count} |")
    lines.append("")

    lines.append("## Example Flips\n")
    lines.append("| Index | Original Label | New Label |")
    lines.append("|--------|----------------|-----------|")
    for flip in results.get("example_flips", []):
        idx = flip["index"]
        orig = flip.get("original_label_name", flip["original_label"])
        new = flip.get("new_label_name", flip["new_label"])
        lines.append(f"| {idx} | {orig} | {new} |")
    lines.append("")

    lines.append("## Visual Flip Examples (first 5)\n")
    lines.append('<div style="display: flex; gap: 10px;">')
    for i, flip in enumerate(results.get("example_flips", [])[:5]):
        orig = flip.get("original_label_name", flip["original_label"])
        new = flip.get("new_label_name", flip["new_label"])
        idx = flip["index"]
        fname = f"flipped_examples/flip_{idx}_{flip['original_label']}_to_{flip['new_label']}.png"
        lines.append(
            f'<div style="text-align: center;">'
            f'<small><strong>{orig} -> {new}</strong></small>'
            f'<img src="{fname}" alt="flip" style="width: 120px;"><br>'
            f'</div>'
        )
    lines.append('</div>\n')


    os.makedirs(os.path.dirname(md_file), exist_ok=True)
    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Markdown report generated at {md_file}")

if __name__ == "__main__":
    generate_data_poisoning_report()
