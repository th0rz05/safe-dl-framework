import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from defenses.adversarial_training.generate_adversarial_training_report import generate_adversarial_training_report


# Caminho para módulo 2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2_attack_simulation")))
from attacks.utils import load_model_cfg_from_profile, evaluate_model, train_model
from backdoor_utils import simulate_static_patch_attack, simulate_learned_trigger_attack


def fgsm_attack(model, x, y, epsilon):
    x.requires_grad = True
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()
    x_grad = x.grad.data.sign()
    x_adv = x + epsilon * x_grad
    return torch.clamp(x_adv, 0, 1).detach()


def pgd_attack(model, x, y, epsilon, alpha=0.01, num_iter=7):
    x_adv = x.clone().detach()
    x_adv.requires_grad = True

    for _ in range(num_iter):
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
        x_adv.requires_grad = True

    return x_adv


def run_adversarial_training_defense(profile, trainset, testset, valset, class_names, attack_type):
    print(f"[*] Running Adversarial Training defense for {attack_type}...")

    cfg = profile["defense_config"]["evasion_attacks"][attack_type]["adversarial_training"]
    epsilon = cfg.get("epsilon", 0.03)
    base_attack = cfg.get("attack_type", "fgsm")

    # Load model
    model = load_model_cfg_from_profile(profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"[*] Training model with adversarial examples (attack: {base_attack}, epsilon: {epsilon})")

    loader = DataLoader(trainset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(3):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/3")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            if base_attack == "fgsm":
                x_adv = fgsm_attack(model, x, y, epsilon)
            elif base_attack == "pgd":
                x_adv = pgd_attack(model, x, y, epsilon)
            else:
                raise ValueError(f"Unsupported attack type for adversarial training: {base_attack}")

            optimizer.zero_grad()
            out = model(x_adv)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    print("[*] Evaluating model on clean test set...")
    acc, per_class = evaluate_model(model, testset, class_names=class_names)

    os.makedirs(f"results/evasion/{attack_type}", exist_ok=True)
    results = {
        "defense": "adversarial_training",
        "attack": attack_type,
        "accuracy_after_defense": acc,
        "per_class_accuracy": per_class,
        "params": {
            "epsilon": epsilon,
            "base_attack": base_attack
        }
    }

    result_path = f"results/evasion/{attack_type}/adversarial_training_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✔] Results saved to {result_path}")

    md_path = f"results/evasion/{attack_type}/adversarial_training_report.md"
    generate_adversarial_training_report(json_file=result_path, md_file=md_path)
    print(f"[✔] Report generated at {md_path}")
