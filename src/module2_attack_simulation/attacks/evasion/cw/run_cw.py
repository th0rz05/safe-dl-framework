import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from attacks.utils import load_model
from attacks.evasion.cw.generate_cw_report import generate_cw_report


def cw_attack(model, x, y, targeted, confidence, initial_const, learning_rate, max_iterations, binary_search_steps):
    device = x.device
    batch_size = x.size(0)
    num_classes = model(x).size(1)

    # Setup: best adversarial examples
    best_adv = x.clone()
    best_l2 = torch.full((batch_size,), float('inf'), device=device)
    best_pred = model(x).argmax(1)

    # Setup optimizer variable
    modifier = torch.zeros_like(x, requires_grad=True)

    optimizer = torch.optim.Adam([modifier], lr=learning_rate)

    # Target labels: if untargeted, use true labels; if targeted, pick random different labels
    if targeted:
        targets = torch.randint_like(y, high=num_classes)
        targets = torch.where(targets == y, (targets + 1) % num_classes, targets)
    else:
        targets = y

    for step in range(max_iterations):
        adv_x = torch.tanh(x + modifier) * 0.5 + 0.5  # ensure adv_x ∈ [0,1]
        logits = model(adv_x)

        real = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        other = logits.topk(2, dim=1)[0][:, 1]  # 2nd best logit

        if targeted:
            loss1 = torch.clamp(other - real + confidence, min=0)
        else:
            loss1 = torch.clamp(real - other + confidence, min=0)

        l2_loss = torch.sum((adv_x - x) ** 2, dim=[1, 2, 3])
        loss = l2_loss + initial_const * loss1

        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        # Update best adversarial
        with torch.no_grad():
            pred_adv = logits.argmax(1)
            if targeted:
                successful = pred_adv == targets
            else:
                successful = pred_adv != y

            improved = successful & (l2_loss < best_l2)
            best_l2 = torch.where(improved, l2_loss, best_l2)
            best_adv = torch.where(improved.unsqueeze(1).unsqueeze(2).unsqueeze(3), adv_x, best_adv)

    return best_adv.detach()


def run_cw(testset, profile, class_names):
    # === Load CW config ===
    cfg = profile["attack_overrides"]["evasion_attacks"]["cw"]
    targeted = cfg.get("targeted", False)
    confidence = cfg.get("confidence", 0.0)
    initial_const = cfg.get("initial_const", 0.001)
    learning_rate = cfg.get("learning_rate", 0.01)
    max_iterations = cfg.get("max_iterations", 1000)
    binary_search_steps = cfg.get("binary_search_steps", 9)

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

    out_dir = os.path.join("results", "evasion", "cw")
    os.makedirs(out_dir, exist_ok=True)
    examples_dir = os.path.join(out_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Clear previous examples
    if os.path.exists(examples_dir):
        for file in os.listdir(examples_dir):
            file_path = os.path.join(examples_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    example_log = []

    for idx, (x, y) in enumerate(tqdm(testloader, desc="C&W Attack")):
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            pred_clean = trained_model(x).argmax(1)

        adv_x = cw_attack(trained_model, x, y, targeted, confidence, initial_const, learning_rate, max_iterations, binary_search_steps)
        pred_adv = trained_model(adv_x).argmax(1)

        total += 1
        per_class_total[y.item()] += 1

        if pred_clean.item() == y.item():
            correct_clean += 1
            per_class_clean[y.item()] += 1

        if pred_adv.item() == y.item():
            correct_adv += 1
            per_class_adv[y.item()] += 1

        # Save up to 5 visual examples
        if len(example_log) < 5:
            vis = adv_x[0].detach().cpu().clone()
            if vis.shape[0] in (3, 4):
                vis = vis.permute(1, 2, 0)
            vis = torch.clamp(vis, 0.0, 1.0)
            vis = (vis * 255).to(torch.uint8).numpy()

            fname = f"cw_{idx}_{class_names[y.item()]}_{class_names[pred_adv.item()]}.png"
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

    print(f"[✔] C&W Attack complete. Clean accuracy: {clean_acc:.4f}, Adversarial accuracy: {adv_acc:.4f}")
    print("[*] Per-class accuracy (Adversarial):")
    for cls, acc in per_class_adv_acc.items():
        print(f"  - {cls}: {acc:.4f}")

    result = {
        "attack_type": "cw",
        "targeted": targeted,
        "confidence": confidence,
        "initial_const": initial_const,
        "learning_rate": learning_rate,
        "max_iterations": max_iterations,
        "binary_search_steps": binary_search_steps,
        "accuracy_clean_testset": round(clean_acc, 4),
        "accuracy_adversarial_testset": round(adv_acc, 4),
        "per_class_clean": per_class_clean_acc,
        "per_class_adversarial": per_class_adv_acc,
        "example_adversarial_samples": example_log
    }

    out_dir = "results/evasion/cw"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "cw_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("[✔] C&W attack complete. Metrics saved to cw_metrics.json")

    generate_cw_report(
        json_file="results/evasion/cw/cw_metrics.json",
        md_file="results/evasion/cw/cw_report.md"
    )

    print("[✔] C&W report generated in results/evasion/cw/cw_report.md")
