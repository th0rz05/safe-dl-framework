import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Adjust path to import utilities and simulation functions from module 2
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "module2_attack_simulation"
        )
    )
)
from attacks.utils import evaluate_model, load_model_cfg_from_profile, train_model
# import flip_labels and clean_label simulation
from attacks.data_poisoning.label_flipping.run_label_flipping import flip_labels
from attacks.data_poisoning.clean_label.run_clean_label import poison_dataset as apply_clean_label

EPS = 1e-6  # numerical stability constant

# -----------------------------------------------------------------------------
# Robust loss function definitions
# -----------------------------------------------------------------------------

class GCELoss(nn.Module):
    """Generalized Cross‑Entropy Loss (Zhang et al. 2018).
    L_q = (1 − p_i^q)/q.  Adding a clamp prevents p_i→0 from exploding."""

    def __init__(self, q: float = 0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        p = probs[torch.arange(len(targets)), targets].clamp(min=EPS)
        loss = (1 - p.pow(self.q)) / self.q
        return loss.mean()


class SCELoss(nn.Module):
    """Symmetric Cross‑Entropy: CE + Reverse‑CE, numerically stable."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        preds = torch.softmax(logits, dim=1).clamp(min=EPS)
        forward_loss = self.ce(logits, targets)
        reverse_loss = - (one_hot * torch.log(preds)).sum(dim=1).mean()
        return self.alpha * forward_loss + self.beta * reverse_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing with numerically stable cross‑entropy."""

    def __init__(self, smoothing: float = 0.1, num_classes: int = 10):
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0,1)"
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        smooth_val = self.smoothing / (self.num_classes - 1)
        true_dist = torch.full_like(log_probs, smooth_val)
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return - (true_dist * log_probs).sum(dim=1).mean()


def make_robust_loss(loss_type: str, **cfg) -> nn.Module:
    """Factory to select robust loss based on type and cfg."""
    if loss_type == "gce":
        return GCELoss(q=cfg.get("q", 0.7))
    elif loss_type == "symmetric_cross_entropy":
        return SCELoss(alpha=cfg.get("alpha", 0.5), beta=cfg.get("beta", 0.5))
    elif loss_type == "label_smoothing":
        return LabelSmoothingLoss(
            smoothing=cfg.get("smoothing", 0.1),
            num_classes=cfg.get("num_classes", 10)
        )
    else:
        raise ValueError(f"Unknown robust loss type: {loss_type}")

# -----------------------------------------------------------------------------
# Runner for Robust Loss defense
# -----------------------------------------------------------------------------

def run_robust_loss_defense(
    profile: dict,
    trainset,
    testset,
    valset,
    class_names,
    attack_type: str
):
    """Apply Robust‑Loss defense for label_flipping or clean_label data‑poisoning attacks."""

    print(f"[*] Running Robust Loss defense for attack '{attack_type}'…")

    # 1. Load defense configuration
    cfg = profile["defense_config"]["data_poisoning"][attack_type]["robust_loss"]
    loss_type = cfg.get("type")

    # 2. Initialize model from profile
    model = load_model_cfg_from_profile(profile)

    # 3. Generate poisoned training set
    if attack_type == "label_flipping":
        attack_cfg = profile.get("attack_overrides", {}).get("data_poisoning", {}).get("label_flipping", {})
        poisoned_trainset, _, _ = flip_labels(
            trainset,
            flip_rate=attack_cfg.get("flip_rate"),
            strategy=attack_cfg.get("strategy"),
            source_class=attack_cfg.get("source_class"),
            target_class=attack_cfg.get("target_class"),
            class_names=class_names
        )
    elif attack_type == "clean_label":
        attack_cfg = profile.get("attack_overrides", {}).get("data_poisoning", {}).get("clean_label", {})
        fraction = attack_cfg.get("fraction_poison", 0.05)
        target_cls = attack_cfg.get("target_class")
        method = attack_cfg.get("perturbation_method", "overlay")
        epsilon = attack_cfg.get("epsilon", 0.1)
        max_iters = attack_cfg.get("max_iterations", 100)
        selection = attack_cfg.get("source_selection", "random")
        # Train a clean helper model for feature extraction (used by clean‑label attack)
        helper = load_model_cfg_from_profile(profile)
        train_model(helper, trainset, valset,
                    epochs=profile.get("training", {}).get("epochs", 3),
                    class_names=class_names,
                    silent=True)
        poisoned_trainset, _, _ = apply_clean_label(
            trainset,
            fraction,
            target_cls,
            method,
            epsilon,
            max_iters,
            selection,
            class_names,
            model=helper
        )
    else:
        raise ValueError(f"Unsupported attack_type '{attack_type}' for Robust Loss defense.")

    # 4. Prepare training settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    lr = profile.get("training", {}).get("lr", 5e-4)  # safer default LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = make_robust_loss(loss_type, **cfg)

    batch_size = profile.get("training", {}).get("batch_size", 64)
    epochs = profile.get("training", {}).get("epochs", 3)
    loader = DataLoader(poisoned_trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size) if valset else None

    # 5. Training loop with gradient clipping and NaN guards
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, labels in tqdm(loader, desc=f"[RobustLoss] Epoch {epoch}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                print("[!] Non‑finite loss encountered; skipping batch.")
                continue
            loss.backward()
            # gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:02d}: Train loss = {total_loss:.4f}")

        if val_loader:
            val_acc, _ = evaluate_model(
                model, valset, class_names=class_names, silent=True, prefix=f"[Val RL] {epoch}/{epochs}"
            )
            print(f"Epoch {epoch:02d}: Val accuracy = {val_acc:.4f}")

    # 6. Final evaluation
    test_acc, per_class_acc = evaluate_model(
        model, testset, class_names=class_names, silent=True, prefix="[Eval RL]"
    )

    # 7. Save results
    out_dir = os.path.join("results", "data_poisoning", attack_type)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "robust_loss_results.json")
    with open(json_path, "w") as fp:
        json.dump({
            "defense": "robust_loss",
            "attack": attack_type,
            **cfg,
            "clean_accuracy": test_acc,
            "per_class_accuracy": per_class_acc
        }, fp, indent=2)
    print(f"[✔] Results JSON saved to {json_path}")

        # 8. Markdown report
    try:
        from generate_robust_loss_report import generate_report
        md_path = os.path.join(out_dir, "robust_loss_report.md")
        generate_report(json_path, md_path)
        print(f"[✔] Report generated at {md_path}")
    except ImportError:
        print("[!] generate_robust_loss_report not found; skipping report generation.")
