import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from attacks.utils import load_model, evaluate_model
from attacks.evasion.nes.generate_nes_report import generate_nes_report

def nes_attack(model, x, y, epsilon, sigma, learning_rate, num_queries, batch_size):
    device = x.device
    model.eval()

    x_adv = x.clone().detach()
    x_adv.requires_grad = False  # NES does not use true gradients

    best_adv = x_adv.clone()

    for _ in range(num_queries // batch_size):
        noise = torch.randn((batch_size,) + x.shape[1:]).to(device)
        noise = sigma * noise

        perturbed_plus = torch.clamp(x_adv + noise, 0, 1)
        perturbed_minus = torch.clamp(x_adv - noise, 0, 1)

        logits_plus = model(perturbed_plus)
        logits_minus = model(perturbed_minus)

        loss_plus = nn.CrossEntropyLoss(reduction='none')(logits_plus, y.repeat(batch_size))
        loss_minus = nn.CrossEntropyLoss(reduction='none')(logits_minus, y.repeat(batch_size))

        est_grad = (loss_plus - loss_minus).view(-1, 1, 1, 1) * noise
        est_grad = est_grad.mean(dim=0)

        x_adv = x_adv + learning_rate * est_grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

        # Track best
        pred_adv = model(x_adv).argmax(1)
        if pred_adv.item() != y.item():
            best_adv = x_adv.clone()
            break  # early stop when attack succeeds

    return best_adv.detach()

def run_nes(testset, profile, class_names):
    # === Load NES config ===
    cfg = profile["attack_overrides"]["evasion_attacks"]["nes"]
    epsilon = cfg["epsilon"]
    sigma = cfg["sigma"]
    learning_rate = cfg["learning_rate"]
    num_queries = cfg["num_queries"]
    batch_size = cfg["batch_size"]

    # === Load clean model ===
    trained_model = load_model("clean_model", profile)
    trained_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)

    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    total = 0
    correct_clean = 0
    correct_adv = 0
    per_class_clean = [0] * len(class_names)
    per_class_total = [0] * len(class_names)
    per_class_adv = [0] * len(class_names)

    out_dir = os.path.join("results", "evasion", "nes")
    os.makedirs(out_dir, exist_ok=True)
    examples_dir = os.path.join(out_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Clear old examples
    if os.path.exists(examples_dir):
        for file in os.listdir(examples_dir):
            file_path = os.path.join(examples_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    example_log = []

    for idx, (x, y) in enumerate(tqdm(testloader, desc="NES Attack")):
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            pred_clean = trained_model(x).argmax(1)

        adv_x = nes_attack(trained_model, x, y, epsilon, sigma, learning_rate, num_queries, batch_size)
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
            vis = adv_x[0].detach().cpu().clone()
            if vis.shape[0] in (3, 4):
                vis = vis.permute(1, 2, 0)
            vis = torch.clamp(vis, 0.0, 1.0)
            vis = (vis * 255).to(torch.uint8).numpy()

            fname = f"nes_{idx}_{class_names[y.item()]}_{class_names[pred_adv.item()]}.png"
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

    print(f"[✔] NES Attack complete. Clean accuracy: {clean_acc:.4f}, Adversarial accuracy: {adv_acc:.4f}")

    result = {
        "attack_type": "nes",
        "epsilon": epsilon,
        "sigma": sigma,
        "learning_rate": learning_rate,
        "num_queries": num_queries,
        "batch_size": batch_size,
        "accuracy_clean_testset": round(clean_acc, 4),
        "accuracy_adversarial_testset": round(adv_acc, 4),
        "per_class_clean": per_class_clean_acc,
        "per_class_adversarial": per_class_adv_acc,
        "example_adversarial_samples": example_log
    }

    out_dir = "results/evasion/nes"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "nes_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    generate_nes_report(
        json_file="results/evasion/nes/nes_metrics.json",
        md_file="results/evasion/nes/nes_report.md"
    )
    print("[✔] NES report generated in results/evasion/nes/nes_report.md")
