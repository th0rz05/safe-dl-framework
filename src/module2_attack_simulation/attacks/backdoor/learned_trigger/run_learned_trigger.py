import os
import json

from attacks.utils import train_model, evaluate_model, load_model
from attacks.backdoor.patch_utils import apply_trigger, mask_l1_loss, total_variation_loss, save_trigger_visualization

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def select_poison_indices(trainset, poison_fraction, class_names, target_class):
    """
    Given trainset (with .targets), pick a list of indices to poison.
    - poison_fraction: float in (0,1]
    - If label_mode=="clean", only pick from samples whose original label == target_class
      (but we’ll enforce label_mode outside)
    Returns: a list of indices (ints)
    """
    # Extract labels
    if hasattr(trainset, "dataset") and hasattr(trainset.dataset, "targets"):
        all_targets = trainset.dataset.targets
        all_indices = trainset.indices if hasattr(trainset, "indices") else list(range(len(all_targets)))
    elif hasattr(trainset, "targets"):
        all_targets = trainset.targets
        all_indices = list(range(len(all_targets)))
    else:
        raise ValueError("Trainset must expose .targets or .dataset.targets")

    N = len(all_indices)
    n_poison = max(1, int(poison_fraction * N))

    # Pick uniformly at random over all N
    return random.sample(all_indices, n_poison)

def optimize_trigger(trainset, model, T, M,
                     mask_weight, tv_weight, learning_rate, epochs,
                     batch_size, target_class):
    """
    Learn trigger T and mask M to force model(poisoned(x)) -> target_class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Prepare T and M as learnable
    T = T.clone().to(device).detach().requires_grad_(True)
    M = M.clone().to(device).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([T, M], lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(epochs), desc="Trigger Epochs"):
        epoch_loss = 0.0
        for x, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x = x.to(device)
            # apply current trigger+mask
            x_p = apply_trigger(x, T, M)
            # target labels
            y_t = torch.full((x.size(0),), target_class,
                             dtype=torch.long, device=device)
            # attack loss
            logits = model(x_p)
            loss_attack = criterion(logits, y_t)
            # regularizers
            loss_mask = mask_weight * mask_l1_loss(M)
            loss_tv   = tv_weight   * total_variation_loss(T)
            loss = loss_attack + loss_mask + loss_tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        tqdm.write(f"[LearnedTrigger] Epoch {epoch+1}/{epochs} loss={epoch_loss:.4f}")

    return T.detach().cpu(), M.detach().cpu()

def create_poisoned_dataset(trainset, poison_indices, T, M, label_mode, target_class):
    """
    Wrap trainset to apply trigger T with mask M on specified indices.
    label_mode: "corrupted" replaces label to target_class; "clean" keeps original.
    """
    class PoisonedDataset(Dataset):
        def __init__(self, base_dataset, poison_idx_set, T, M, label_mode, target_class):
            self.base = base_dataset
            self.poison_idx_set = set(poison_idx_set)
            # Pre-move trigger/mask to CPU; apply_trigger will cast to device of x
            self.T = T
            self.M = M
            self.label_mode = label_mode
            self.target_class = target_class

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            x, y = self.base[idx]
            if idx in self.poison_idx_set:
                # apply trigger
                x = x.unsqueeze(0)  # [1,C,H,W]
                x = apply_trigger(x, self.T, self.M)[0]
                # corrupt label?
                if self.label_mode == "corrupted":
                    y = self.target_class
            return x, y

    return PoisonedDataset(trainset, poison_indices, T, M, label_mode, target_class)


def run_learned_trigger(trainset, testset, valset, model, profile, class_names):
    # 1) Load hyperparams from profile
    cfg = profile["attack_overrides"]["backdoor"]["learned"]
    target_class     = cfg["target_class"]
    poison_fraction  = cfg["poison_fraction"]
    patch_size_ratio = cfg["patch_size_ratio"]
    label_mode       = cfg["label_mode"]
    lr_trigger       = cfg["learning_rate"]
    epochs_trigger   = cfg["epochs"]
    mask_weight      = cfg["lambda_mask"]
    tv_weight        = cfg["lambda_tv"]

    # optional retrain params
    train_epochs     = 5
    train_batch      = 64

    # 2) init small T, M
    x0, _ = trainset[0]
    C, H, W = x0.shape
    ph = max(1, int(H * patch_size_ratio))
    pw = max(1, int(W * patch_size_ratio))
    T_init = torch.zeros(C, ph, pw)
    M_init = torch.zeros(1, ph, pw)

    # 3) pick which indices to poison
    poison_idx = select_poison_indices(trainset, poison_fraction, class_names, target_class)

    trained_model = load_model("clean_model",profile)

    # 4) learn T & M
    T_opt, M_opt = optimize_trigger(
        trainset, trained_model, T_init, M_init,
        mask_weight=mask_weight,
        tv_weight=tv_weight,
        learning_rate=lr_trigger,
        epochs=epochs_trigger,
        batch_size=cfg.get("batch_size", 32),
        target_class=target_class
    )

    # 5) save artifacts + viz
    out_dir = os.path.join("results", "backdoor", "learned_trigger")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(T_opt, os.path.join(out_dir, "trigger.pt"))
    torch.save(M_opt, os.path.join(out_dir, "mask.pt"))
    save_trigger_visualization(T_opt, M_opt, out_dir)

    # 6) build poisoned training set
    poisoned_train = create_poisoned_dataset(
        trainset, poison_idx, T_opt, M_opt, label_mode, target_class
    )

    # 7) re‑init and train new model on poisoned data
    train_model(model, poisoned_train, valset=valset,
                epochs=train_epochs, batch_size=train_batch,
                class_names=class_names)

    # 8) CDA on clean test set
    cda, per_class_cda = evaluate_model(model, testset, class_names=class_names)

    # 9) ASR: force–poison all test samples
    all_idxs = list(range(len(testset)))
    poisoned_test = create_poisoned_dataset(
        testset, all_idxs, T_opt, M_opt, "corrupted", target_class
    )
    asr, _ = evaluate_model(model, poisoned_test, class_names=class_names)

    # 10) dump metrics + report
    metrics = {
        "attack": "learned",
        "target_class": target_class,
        "poison_fraction": poison_fraction,
        "patch_size_ratio": patch_size_ratio,
        "label_mode": label_mode,
        "learning_rate": lr_trigger,
        "epochs_trigger": epochs_trigger,
        "mask_weight": mask_weight,
        "tv_weight": tv_weight,
        "train_epochs": train_epochs,
        "train_batch_size": train_batch,
        "CDA": cda,
        "ASR": asr,
        "per_class_CDA": per_class_cda,
    }
    with open(os.path.join(out_dir, "learned_trigger_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


