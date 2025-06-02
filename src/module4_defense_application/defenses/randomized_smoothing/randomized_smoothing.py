import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from evasion_utils import load_clean_model, apply_attack_to_dataset
from defenses.randomized_smoothing.generate_randomized_smoothing_report import generate_randomized_smoothing_report


def evaluate_with_smoothing(model, dataset, sigma, device, class_names, num_samples=25):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    correct = 0
    per_class = {cls: {"correct": 0, "total": 0} for cls in class_names}

    for x, y in tqdm(loader, desc=f"Evaluating with Smoothing (sigma={sigma})"):
        x, y = x.to(device), y.to(device)
        x = x.squeeze()  # Ensure x is [C, H, W]
        if x.dim() != 3:
            raise ValueError(f"Invalid input shape {x.shape} after squeeze; expected [C, H, W]")

        votes = torch.zeros(len(class_names))

        for _ in range(num_samples):
            noise = sigma * torch.randn_like(x)
            noisy_input = torch.clamp(x + noise, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(noisy_input).argmax(dim=1)
            votes[pred.item()] += 1

        final_pred = votes.argmax().item()
        correct += (final_pred == y.item())
        cls = class_names[y.item()]
        per_class[cls]["total"] += 1
        if final_pred == y.item():
            per_class[cls]["correct"] += 1

    acc = correct / len(dataset)
    per_class_acc = {cls: round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0 for cls, v in per_class.items()}
    return round(acc, 4), per_class_acc


def run_randomized_smoothing_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Randomized Smoothing defense for evasion attack: {attack_type}...")
    cfg = profile["defense_config"]["evasion_attacks"][attack_type]["randomized_smoothing"]
    sigma = cfg.get("sigma", 0.25)
    num_samples = cfg.get("num_samples", 25)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_clean_model("clean_model", profile).to(device)

    print("[*] Evaluating smoothed model on clean test set...")
    acc_clean, per_class_clean = evaluate_with_smoothing(model, testset, sigma, device, class_names, num_samples)

    print(f"[*] Generating adversarial examples with {attack_type}...")
    adv_testset = apply_attack_to_dataset(model, testset, attack_type, epsilon=0.03, device=device)

    print("[*] Evaluating smoothed model on adversarial test set...")
    acc_adv, per_class_adv = evaluate_with_smoothing(model, adv_testset, sigma, device, class_names, num_samples)

    os.makedirs(f"results/evasion/{attack_type}", exist_ok=True)
    results = {
        "defense_name": "randomized_smoothing",
        "evaluated_attack": attack_type,
        "sigma": sigma,
        "num_samples": num_samples,
        "smoothed_accuracy_clean": acc_clean,
        "smoothed_accuracy_adversarial": acc_adv,
        "per_class_smoothed_accuracy_clean": per_class_clean,
        "per_class_smoothed_accuracy_adversarial": per_class_adv
    }

    json_path = f"results/evasion/{attack_type}/randomized_smoothing_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results saved to {json_path}")

    md_path = f"results/evasion/{attack_type}/randomized_smoothing_report.md"
    generate_randomized_smoothing_report(json_file=json_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
