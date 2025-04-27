import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from attacks.utils import load_model, evaluate_model
from attacks.evasion.spsa.generate_spsa_report import generate_spsa_report

def spsa_attack(model, x, y, epsilon, delta, learning_rate, num_steps, batch_size):
    device = x.device
    model.eval()

    x_adv = x.clone().detach()
    x_adv.requires_grad_(True)

    for step in range(num_steps):
        # Generate random perturbations
        perturbations = torch.sign(torch.randn((batch_size,) + x.shape[1:], device=device))
        perturbations = perturbations * delta

        x_perturbed_plus = torch.clamp(x_adv + perturbations, 0, 1)
        x_perturbed_minus = torch.clamp(x_adv - perturbations, 0, 1)

        logits_plus = model(x_perturbed_plus)
        logits_minus = model(x_perturbed_minus)

        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss_plus = loss_fn(logits_plus, y.repeat(batch_size))
        loss_minus = loss_fn(logits_minus, y.repeat(batch_size))

        est_grad = (loss_plus - loss_minus).view(-1, 1, 1, 1) * perturbations
        est_grad = est_grad.mean(dim=0)

        x_adv = x_adv + learning_rate * est_grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()
        x_adv.requires_grad_(True)

    return x_adv.detach()

def run_spsa(testset, profile, class_names):
    # === Load SPSA config ===
    cfg = profile["attack_overrides"]["evasion_attacks"]["spsa"]
    epsilon = cfg["epsilon"]
    delta = cfg["delta"]
    learning_rate = cfg["learning_rate"]
    num_steps = cfg["num_steps"]
    batch_size = cfg["batch_size"]
    max_samples = cfg.get("max_samples", 0)

    # === Load clean model ===
    trained_model = load_model("clean_model", profile)
    trained_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)

    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    total_samples = len(testloader)
    if max_samples <= 0 or max_samples > total_samples:
        max_samples = total_samples

    total = 0
    correct_clean = 0
    correct_adv = 0
    per_class_clean = [0] * len(class_names)
    per_class_total = [0] * len(class_names)
    per_class_adv = [0] * len(class_names)

    out_dir = os.path.join("results", "evasion", "spsa")
    os.makedirs(out_dir, exist_ok=True)
    examples_dir = os.path.join(out_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Delete old examples
    for file in os.listdir(examples_dir):
        file_path = os.path.join(examples_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    example_log = []

    for idx, (x, y) in enumerate(tqdm(testloader, desc="SPSA Attack", total=max_samples)):
        if idx >= max_samples:
            break

        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            pred_clean = trained_model(x).argmax(1)

        adv_x = spsa_attack(trained_model, x, y, epsilon, delta, learning_rate, num_steps, batch_size)
        pred_adv = trained_model(adv_x).argmax(1)

        total += 1
        per_class_total[y.item()] += 1

        if pred_clean.item() == y.item():
            correct_clean += 1
            per_class_clean[y.item()] += 1

        if pred_adv.item() == y.item():
            correct_adv += 1
            per_class_adv[y.item()] += 1

        # Save up to 5 examples
        if len(example_log) < 5:
            vis = adv_x[0].detach().cpu()
            if vis.shape[0] in (3, 4):
                vis = vis.permute(1, 2, 0)
            vis = torch.clamp(vis, 0.0, 1.0)
            vis = (vis * 255).to(torch.uint8).numpy()

            fname = f"spsa_{idx}_{class_names[y.item()]}_{class_names[pred_adv.item()]}.png"
            path = os.path.join(examples_dir, fname)
            plt.imsave(path, vis, format="png")

            example_log.append({
                "index": idx,
                "true_label": y.item(),
                "true_label_name": class_names[y.item()],
                "pred_clean": pred_clean.item(),
                "pred_adv": pred_adv.item(),
                "example_image_path": f"examples/{fname}"
            })

    clean_acc = correct_clean / total
    adv_acc = correct_adv / total

    per_class_clean_acc = {
        class_names[i]: round(per_class_clean[i] / per_class_total[i], 4) if per_class_total[i] > 0 else 0.0
        for i in range(len(class_names))
    }

    per_class_adv_acc = {
        class_names[i]: round(per_class_adv[i] / per_class_total[i], 4) if per_class_total[i] > 0 else 0.0
        for i in range(len(class_names))
    }

    print(f"[✔] SPSA Attack complete. Clean accuracy: {clean_acc:.4f}, Adversarial accuracy: {adv_acc:.4f}")
    print("[*] Per-class adversarial accuracy:")
    for cls, acc in per_class_adv_acc.items():
        print(f"  - {cls}: {acc:.4f}")

    result = {
        "attack_type": "spsa",
        "epsilon": epsilon,
        "delta": delta,
        "learning_rate": learning_rate,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "accuracy_clean_testset": round(clean_acc, 4),
        "accuracy_adversarial_testset": round(adv_acc, 4),
        "per_class_clean": per_class_clean_acc,
        "per_class_adversarial": per_class_adv_acc,
        "example_adversarial_samples": example_log
    }

    # Save results
    with open(os.path.join(out_dir, "spsa_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("[✔] SPSA metrics saved.")

    # Generate report
    generate_spsa_report(
        json_file=os.path.join(out_dir, "spsa_metrics.json"),
        md_file=os.path.join(out_dir, "spsa_report.md")
    )

    print("[✔] SPSA report generated.")

