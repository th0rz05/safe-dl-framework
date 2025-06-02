import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Para carregar arquitetura correta
from attacks.utils import load_model_cfg_from_profile
from attacks.evasion.spsa.run_spsa import spsa_attack

def load_clean_model(model_name, profile):
    profile_name = profile.get("name", "default")

    # Caminho absoluto para pasta de modelos salvos no módulo 2
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "module2_attack_simulation"))
    model_path = os.path.join(base_path, "saved_models", f"{profile_name}_{model_name}.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[!] Clean model not found at {model_path}")

    model = load_model_cfg_from_profile(profile)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print(f"[✔] Loaded clean model from {model_path}")
    return model

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

def evaluate_robustness(model, testset, attack_type, epsilon, device):
    model.eval()
    loader = DataLoader(testset, batch_size=64, shuffle=False)
    correct = 0
    total = 0
    per_class = {}

    pbar = tqdm(loader, desc=f"Evaluating Robustness ({attack_type})")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        if attack_type == "fgsm":
            with torch.no_grad():
                x_adv = fgsm_attack(model, x, y, epsilon)
        elif attack_type == "pgd":
            x_adv = pgd_attack(model, x, y, epsilon)
        else:
            raise ValueError(f"[!] Unsupported evaluation attack: {attack_type}")

        preds = model(x_adv).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        for yi, pi in zip(y, preds):
            cls = yi.item()
            if cls not in per_class:
                per_class[cls] = {"correct": 0, "total": 0}
            per_class[cls]["total"] += 1
            if yi == pi:
                per_class[cls]["correct"] += 1

        pbar.set_postfix(acc=round(correct / total, 4))

    per_class_acc = {cls: round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0 for cls, v in per_class.items()}
    return round(correct / total, 4), per_class_acc

def apply_attack_to_dataset(model, testset, attack_type, epsilon, device):
    adversarial_data = []
    model.eval()
    loader = DataLoader(testset, batch_size=1, shuffle=False)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if attack_type == "fgsm":
            x_adv = fgsm_attack(model, x, y, epsilon)
        elif attack_type == "pgd":
            x_adv = pgd_attack(model, x, y, epsilon)
        else:
            raise ValueError(f"[!] Unsupported attack type: {attack_type}")
        adversarial_data.append((x_adv.cpu(), y.cpu()))

    return adversarial_data

def apply_attack_spsa_to_dataset(model, testset, profile, device):
    print("[*] Applying SPSA attack to dataset...")
    cfg = profile.get("attack_overrides", {}).get("evasion_attacks", {}).get("spsa", {})

    epsilon = cfg.get("epsilon", 0.03)
    delta = cfg.get("delta", 0.01)
    learning_rate = cfg.get("learning_rate", 0.01)
    num_steps = cfg.get("num_steps", 150)
    batch_size = cfg.get("batch_size", 32)
    max_samples = cfg.get("max_samples", 500)

    loader = DataLoader(testset, batch_size=1, shuffle=False)
    adversarial_data = []

    pbar = tqdm(total=max_samples, desc="SPSA Attack")
    count = 0

    for x, y in loader:
        if count >= max_samples:
            break
        x, y = x.to(device), y.to(device)
        x_adv = spsa_attack(model, x, y, epsilon=epsilon, delta=delta, learning_rate=learning_rate, num_steps=num_steps, batch_size=batch_size)
        adversarial_data.append((x_adv.cpu(), y.cpu()))
        count += 1
        pbar.update(1)

    pbar.close()
    return adversarial_data
