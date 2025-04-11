import json
import os


def generate_clean_label_report(json_file, md_file):
    # Lê o arquivo JSON com os resultados
    with open(json_file, "r") as f:
        results = json.load(f)

    lines = []
    lines.append("# Data Poisoning - Clean Label Attack Report\n")

    lines.append("## Overview\n")
    lines.append(f"- **Attack Type:** {results.get('attack_type')}")
    lines.append(f"- **Perturbation Method:** {results.get('perturbation_method')}")
    lines.append(f"- **Poison Fraction:** {results.get('fraction_poison')}")
    if results.get("target_class") is not None:
        lines.append(f"- **Target Class:** {results.get('target_class')} ({results.get('target_class_name')})")
    else:
        lines.append("- **Target Class:** None (Untargeted)")
    lines.append(f"- **Max Iterations:** {results.get('max_iterations')}")
    lines.append(f"- **Epsilon:** {results.get('epsilon')}")
    lines.append(f"- **Source Selection:** {results.get('source_selection')}")
    lines.append(f"- **Number of Poisoned Samples:** {results.get('num_poisoned_samples')}")
    lines.append("")

    lines.append("## Performance Metrics\n")
    lines.append(f"- **Accuracy After Attack:** {results.get('accuracy_after_attack')}\n")

    lines.append("### Per-Class Accuracy\n")
    lines.append("| Class | Accuracy |")
    lines.append("|--------|----------|")
    for class_name, acc in results.get("per_class_accuracy", {}).items():
        lines.append(f"| {class_name} | {acc:.4f} |")
    lines.append("")

    lines.append("## Example Poisoned Samples\n")
    lines.append("| Index | Original Label | Perturbation Norm |")
    lines.append("|--------|----------------|-------------------|")
    for sample in results.get("example_poisoned_samples", []):
        idx = sample["index"]
        orig = sample.get("original_label_name", sample["original_label"])
        lines.append(f"| {idx} | {orig} | {sample['perturbation_norm']:.4f} |")
    lines.append("")

    lines.append("## Visual Poisoned Examples (first 5)\n")
    lines.append('<div style="display: flex; gap: 10px;">')
    for i, sample in enumerate(results.get("example_poisoned_samples", [])[:5]):
        orig = sample.get("original_label_name", sample["original_label"])
        idx = sample["index"]
        fname = f"examples/poison_{idx}_{orig}.png"
        lines.append(
            f'<div style="text-align: center;">'
            f'<small><strong>{orig}</strong></small><br>'
            f'<img src="{fname}" alt="poisoned_example" style="width: 120px;">'
            f'</div>'
        )
    lines.append('</div>\n')

    # Salva o relatório gerado no arquivo Markdown
    os.makedirs(os.path.dirname(md_file), exist_ok=True)
    with open(md_file, "w") as f:
        f.write("\n".join(lines))

    print(f"[✔] Markdown report generated at {md_file}")


if __name__ == "__main__":
    generate_clean_label_report("../../../results/data_poisoning/clean_label/clean_label_metrics.json",
                                "../../../results/data_poisoning/clean_label/clean_label_report.md")
