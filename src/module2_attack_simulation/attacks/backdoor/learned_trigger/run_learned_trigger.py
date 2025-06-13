import os
import json
import random
from copy import deepcopy  # Import deepcopy for models
from tqdm.auto import tqdm  # Use tqdm.auto for better compatibility

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from attacks.utils import train_model, evaluate_model, load_model, \
    get_class_labels  # Make sure get_class_labels is imported
from attacks.backdoor.patch_utils import apply_trigger, mask_l1_loss, total_variation_loss, save_trigger_visualization

from attacks.backdoor.learned_trigger.generate_learned_trigger_report import generate_learned_trigger_report


def select_poison_indices(trainset,
                          poison_fraction,
                          class_names,
                          target_class):
    """
    Pick indices **local to `trainset`** that will be poisoned.
    Works for torch.utils.data.Dataset or torch.utils.data.Subset.
    """
    # ---------- get label list ----------
    if isinstance(trainset, torch.utils.data.Subset):
        # Subset → iterate via local indices and pull label via __getitem__
        all_indices = list(range(len(trainset)))  # 0 … N‑1  (local)

        def _label(i):  # helper to fetch label for subset index i
            return trainset[i][1]
    else:
        # Plain dataset with .targets / .dataset.targets
        if hasattr(trainset, "targets"):
            labels = trainset.targets
        elif hasattr(trainset, "dataset") and hasattr(trainset.dataset, "targets"):
            labels = trainset.dataset.targets
        else:
            # Fallback for custom datasets or subsets without .targets
            # This can be slow for large datasets, consider optimizing if needed
            labels = [trainset[i][1] for i in range(len(trainset))]

        all_indices = list(range(len(labels)))

        def _label(i):
            return labels[i]

    # ---------- select indices to poison ----------
    # we poison original_class ↦ target_class
    # so we select original_class ≠ target_class
    candidate_indices = [
        i for i in all_indices if _label(i) != target_class
    ]
    random.shuffle(candidate_indices)  # Shuffle to pick randomly

    num_poison = int(len(candidate_indices) * poison_fraction)
    poison_idx = sorted(random.sample(candidate_indices, num_poison))
    print(
        f"[*] Poisoning {len(poison_idx)} samples from {len(class_names) - 1} classes to target class '{class_names[target_class]}'")

    return poison_idx


class PoisonedDataset(Dataset):
    def __init__(self, original_dataset, poison_indices, target_class, T_opt, M_opt, label_mode, device):
        self.original_dataset = original_dataset
        self.poison_indices = poison_indices
        self.target_class = target_class
        self.T_opt = T_opt.to(device)
        self.M_opt = M_opt.to(device)
        self.label_mode = label_mode
        self.device = device

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        img = img.to(self.device)

        if idx in self.poison_indices:
            # Apply trigger
            # Unsqueeze to add batch dimension (1, C, H, W) for apply_trigger
            patched_img = apply_trigger(img.unsqueeze(0), self.T_opt, self.M_opt)[0]  # Get back (C, H, W)

            # Determine new label based on label_mode
            if self.label_mode == "corrupted":
                new_label = self.target_class
            elif self.label_mode == "clean":
                new_label = label
            else:
                raise ValueError(f"Unknown label_mode: {self.label_mode}. Must be 'corrupted' or 'clean'.")

            return patched_img, new_label
        else:
            return img, label


class BackdoorASRDataset(Dataset):
    """
    A dataset for calculating the ASR of backdoor attacks for Learned Trigger.
    Returns the poisoned image, the target_class, and the original class.
    """

    def __init__(self, original_dataset, T_opt, M_opt, target_class, poison_indices, device):
        self.original_dataset = original_dataset
        self.T_opt = T_opt.to(device)
        self.M_opt = M_opt.to(device)
        self.target_class = target_class
        self.poison_indices = poison_indices  # Indices of testset samples that will be poisoned for ASR
        self.device = device

        self.data_for_asr = []
        for original_idx in self.poison_indices:
            original_img, original_label = self.original_dataset[original_idx]
            original_img = original_img.to(self.device)  # Move to device

            # Apply the learned trigger to the original image
            poisoned_img = apply_trigger(original_img.unsqueeze(0), self.T_opt, self.M_opt)[0]

            self.data_for_asr.append(
                (poisoned_img.cpu(), self.target_class, original_label))  # Store on CPU for DataLoader

    def __len__(self):
        return len(self.data_for_asr)

    def __getitem__(self, idx):
        # Load from stored data, ensure tensors are on CPU if DataLoader moves them
        img, target_cls, original_cls = self.data_for_asr[idx]
        return img, target_cls, original_cls


def run_learned_trigger(trainset, testset, valset, model, profile, class_names):
    print("[*] Running Learned Trigger Backdoor Attack...")

    # === Load configuration ===
    attack_cfg = profile.get("attack_overrides", {}).get("backdoor", {}).get("learned_trigger", {})

    poison_fraction = attack_cfg.get("poison_fraction", 0.1)
    target_class = attack_cfg.get("target_class")
    label_mode = attack_cfg.get("label_mode", "corrupted")  # 'corrupted' or 'clean'

    # Learned Trigger specific parameters
    epochs_trigger = attack_cfg.get("epochs_trigger", 20)
    learning_rate = attack_cfg.get("learning_rate_trigger", 0.01)  # Renamed to avoid clash with model LR
    lambda_mask = attack_cfg.get("lambda_mask", 0.1)  # L1 regularization for mask sparsity
    lambda_tv = attack_cfg.get("lambda_tv", 0.0001)  # Total Variation regularization for smoothness

    input_shape = profile.get("model", {}).get("input_shape", [3, 32, 32])
    C, H, W = input_shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = profile.get("training", {}).get("batch_size", 64)

    # Validate target class
    if target_class is None:
        raise ValueError("Target class must be specified in the profile for backdoor attacks.")
    if not (0 <= target_class < len(class_names)):
        raise ValueError(f"Target class {target_class} is out of bounds for {len(class_names)} classes.")

    # === Generate initial trigger and mask ===
    # T is the trigger (patch), M is the mask (unnormalized)
    T = (torch.rand(input_shape) * 2 - 1).to(device)  # Random trigger, -1 to 1
    M = torch.randn(1, 1, H, W).to(device)  # Random mask, 1 channel

    T.requires_grad = True
    M.requires_grad = True

    optimizer_trigger = torch.optim.Adam([T, M], lr=learning_rate)

    # Freeze model parameters for trigger optimization
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)

    # === Select poison indices from the *training set* ===
    poison_train_indices = select_poison_indices(trainset, poison_fraction, class_names, target_class)

    # Create a temporary dataset for trigger optimization.
    # This dataset only provides clean samples at poison_train_indices
    # because the trigger is learned *onto* these samples.
    # The actual poisoned_trainset for model training will be created later.
    temp_trigger_opt_dataset = torch.utils.data.Subset(trainset, poison_train_indices)
    trigger_opt_loader = DataLoader(temp_trigger_opt_dataset, batch_size=batch_size, shuffle=True)

    print("[*] Learning trigger and mask...")
    for epoch in range(epochs_trigger):
        running_loss = 0.0
        for i, (img, _) in enumerate(trigger_opt_loader):
            img = img.to(device)  # Original clean images

            optimizer_trigger.zero_grad()

            # Apply trigger
            # The apply_trigger function expects (batch_size, C, H, W)
            patched_img_batch = apply_trigger(img, T, M)

            # Predict target class on patched images
            output = model(patched_img_batch)

            # Loss: maximize probability of target_class, minimize L1 of mask, minimize TV of trigger
            # For cross-entropy, target needs to be a tensor of target_class
            target_labels = torch.full((output.shape[0],), target_class, dtype=torch.long, device=device)
            classification_loss = nn.CrossEntropyLoss()(output, target_labels)

            # Regularization losses
            l1_loss = mask_l1_loss(M) * lambda_mask
            tv_loss = total_variation_loss(T) * lambda_tv

            loss = classification_loss + l1_loss + tv_loss
            loss.backward()
            optimizer_trigger.step()
            running_loss += loss.item()

        print(f"Trigger Epoch {epoch + 1}/{epochs_trigger}, Loss: {running_loss / len(trigger_opt_loader):.4f}")

    T_opt = T.detach()  # Optimized Trigger
    M_opt = M.detach()  # Optimized Mask

    # 5) save artifacts + viz
    out_dir = os.path.join("results", "backdoor", "learned_trigger")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(T_opt, os.path.join(out_dir, "trigger.pt"))
    torch.save(M_opt, os.path.join(out_dir, "mask.pt"))
    save_trigger_visualization(T_opt, M_opt, out_dir)  # This saves trigger.png, mask.png, overlay.png

    examples_dir = os.path.join(out_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Delete examples from the previous runs
    if os.path.exists(examples_dir):
        for file in os.listdir(examples_dir):
            file_path = os.path.join(examples_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    example_log = []

    # Only keep the first 5 indices
    # We use the full trainset for getting the image, as poison_train_indices are relative to the full trainset
    for idx in poison_train_indices[:5]:  # Use poison_train_indices, not just poison_idx
        # pull the clean image
        img, _ = trainset[idx]  # img: Tensor [C,H,W]
        img = img.clone().to(device)

        # apply the learned trigger
        # Unsqueeze to add batch dimension (1, C, H, W) for apply_trigger
        patched = apply_trigger(img.unsqueeze(0), T_opt.to(device), M_opt.to(device))[0]  # Get back (C,H,W)

        # ---- PREPARE A SAFE COPY FOR SAVING --------------------
        vis = patched.detach().cpu().clone()
        if vis.shape[0] in (3, 4):  # If it's a color image (C,H,W)
            vis = vis.permute(1, 2, 0)  # CHW ➜ HWC
        elif vis.shape[0] == 1:  # If it's grayscale (1,H,W)
            vis = vis.squeeze(0)  # Remove channel dimension to (H,W) for grayscale plotting

        vis = torch.clamp(vis, 0.0, 1.0)  # force into [0,1]
        vis = (vis * 255).to(torch.uint8).numpy()

        save_name = f"poison_{idx}_{class_names[target_class]}.png"
        save_path = os.path.join(examples_dir, save_name)
        plt.imsave(save_path, vis, format="png")
        # ---------------------------------------------------------

        # compute perturbation norm
        perturb_norm = torch.norm((patched - img).view(-1), p=2).item()

        example_log.append({
            "index": idx,
            "target_class": target_class,
            "target_class_name": class_names[target_class],
            "perturbation_norm": perturb_norm,
            "example_image_path": os.path.join("examples", save_name)  # Store relative path for report
        })

    # === Train poisoned model ===
    print("[*] Training model on poisoned data...")
    # Make a deepcopy of the *original* model, as the 'model' object was put into eval mode and its grads frozen
    poisoned_model = deepcopy(model).to(device)

    # Unfreeze model parameters for training
    for param in poisoned_model.parameters():
        param.requires_grad = True

    # Create the poisoned training dataset using the learned trigger and mask
    final_poisoned_trainset = PoisonedDataset(trainset, poison_train_indices, target_class, T_opt, M_opt, label_mode,
                                              device)

    # Extract training parameters from profile to pass to train_model
    training_cfg = profile.get("training", {})
    epochs = training_cfg.get("epochs", 3)
    # Use the model's learning rate from profile for training, not the trigger's LR
    model_lr = training_cfg.get("learning_rate", 1e-3)

    # Call train_model (it trains in-place, does not return the model)
    train_model(poisoned_model, final_poisoned_trainset, valset,
                epochs=epochs,
                batch_size=batch_size,
                class_names=class_names,
                lr=model_lr)

    # The 'poisoned_model' object now holds the trained weights.
    trained_poisoned_model = poisoned_model
    trained_poisoned_model.eval()  # Ensure it's in evaluation mode for metrics calculation

    # === Evaluate clean accuracy of poisoned model ===
    print("[*] Evaluating clean accuracy of poisoned model...")
    clean_acc, clean_per_class = evaluate_model(trained_poisoned_model, testset, class_names, batch_size, device)
    print(f"[+] Clean Accuracy: {clean_acc:.4f}")

    # === Evaluate Attack Success Rate (ASR) ===
    print("[*] Evaluating Attack Success Rate (ASR)...")

    # Select samples from the *testset* for ASR calculation.
    asr_test_indices = []
    if hasattr(testset, 'targets'):
        all_test_targets = testset.targets
    elif hasattr(testset.dataset, 'targets'):  # For Subset
        all_test_targets = testset.dataset.targets
    else:
        all_test_targets = [testset[i][1] for i in range(len(testset))]

    for i, label in enumerate(all_test_targets):
        if label != target_class:
            asr_test_indices.append(i)

    num_asr_samples = min(1000, len(asr_test_indices))  # Limit to avoid excessive computation
    selected_asr_indices = random.sample(asr_test_indices, num_asr_samples)

    # Create the DataLoader for ASR evaluation using the custom dataset
    testset_poisoned_asr = BackdoorASRDataset(testset, T_opt, M_opt, target_class, selected_asr_indices, device)
    test_loader_poisoned_asr = DataLoader(testset_poisoned_asr, batch_size=batch_size, shuffle=False)

    asr_total = 0
    asr_success = 0
    # Initialize dictionaries for ASR per original class
    asr_success_per_class = {cls_name: 0 for cls_name in class_names}
    asr_total_per_class = {cls_name: 0 for cls_name in class_names}

    with torch.no_grad():
        for i, (img, target_cls_batch, original_cls_batch) in enumerate(
                tqdm(test_loader_poisoned_asr, desc="Calculating ASR")):
            img = img.to(device)
            output = trained_poisoned_model(img)
            pred = output.argmax(1)  # Model's prediction

            for j in range(img.shape[0]):  # Iterate per sample in the batch
                current_original_class_name = class_names[original_cls_batch[j].item()]

                asr_total_per_class[current_original_class_name] += 1
                asr_total += 1

                # If the prediction matches the target_class, it's an ASR success
                if pred[j].item() == target_cls_batch[j].item():
                    asr_success_per_class[current_original_class_name] += 1
                    asr_success += 1

    asr = asr_success / asr_total if asr_total > 0 else 0.0

    # Calculate ASR per class
    calculated_asr_per_class = {}
    for cls_name in class_names:
        if asr_total_per_class[cls_name] > 0:
            calculated_asr_per_class[cls_name] = asr_success_per_class[cls_name] / asr_total_per_class[cls_name]
        else:
            calculated_asr_per_class[cls_name] = 0.0

    print(f"[+] Overall ASR: {asr:.4f} ({asr_success}/{asr_total})")
    print("[+] ASR by Original Class:")
    for cls_name, asr_val in calculated_asr_per_class.items():
        print(f"    - {cls_name}: {asr_val:.4f} ({asr_success_per_class[cls_name]}/{asr_total_per_class[cls_name]})")

    # 10) Dump metrics + report
    results_dir = "results/backdoor/learned_trigger"
    os.makedirs(results_dir, exist_ok=True)  # Ensure the directory exists

    result = {
        "attack_type": "learned_trigger",
        "poison_fraction": poison_fraction,
        "label_mode": label_mode,
        "target_class": target_class,
        "target_class_name": class_names[target_class],
        "epochs_trigger": epochs_trigger,
        "learning_rate_trigger": learning_rate,
        "lambda_mask": lambda_mask,
        "lambda_tv": lambda_tv,

        "accuracy_clean_testset": clean_acc,
        "per_class_clean": clean_per_class,  # This is clean accuracy per class

        "attack_success_rate": asr,
        "attack_success_numerator": asr_success,
        "attack_success_denominator": asr_total,
        "per_class_attack_success_rate": calculated_asr_per_class,  # NEW! ASR per class
        "per_class_asr_numerator": asr_success_per_class,  # NEW!
        "per_class_asr_denominator": asr_total_per_class,  # NEW!

        "example_poisoned_samples": example_log
    }

    with open(os.path.join(results_dir, "learned_trigger_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("[✔] Metrics saved to results/backdoor/learned_trigger/learned_trigger_metrics.json")

    # Generate learned trigger report
    generate_learned_trigger_report(
        os.path.join(results_dir, "learned_trigger_metrics.json"),
        os.path.join(results_dir, "learned_trigger_report.md"),
        class_names  # Pass class_names to the report generator
    )