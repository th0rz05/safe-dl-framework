import os
import json
import random
from copy import deepcopy
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset  # Import DataLoader and Dataset

import matplotlib.pyplot as plt

# We MUST use the existing functions as they are defined in your patch_utils.py
from attacks.backdoor.patch_utils import generate_patch, apply_static_patch, update_poisoned_sample
# Ensure save_model is imported if it's used elsewhere, but for this specific request,
# we are removing the call to save_model, so it might not be strictly needed here anymore
from attacks.utils import train_model, evaluate_model, get_class_labels, \
    save_model  # Keep save_model import just in case, but it won't be called

from attacks.backdoor.static_patch.generate_static_patch_report import generate_static_patch_report


class BackdoorASRDataset(Dataset):
    """
    A dataset for calculating the ASR of backdoor attacks.
    Returns the poisoned image, the target_class, and the original class.
    """

    def __init__(self, original_dataset, patch, patch_position, target_class, blend_alpha, poison_indices):
        self.original_dataset = original_dataset
        self.patch = patch
        self.patch_position = patch_position
        self.target_class = target_class
        self.blend_alpha = blend_alpha
        self.poison_indices = poison_indices  # Indices of testset samples that will be poisoned for ASR

        # Create a list of tuples: (poisoned_image, target_class, original_label)
        self.data_for_asr = []
        for original_idx in self.poison_indices:
            original_img, original_label = self.original_dataset[original_idx]

            # Apply the patch to the original image using the provided apply_static_patch
            # Note: apply_static_patch expects (image_tensor, patch, patch_position, blend_alpha)
            # original_img should be a tensor
            poisoned_img = apply_static_patch(original_img, self.patch, self.patch_position, self.blend_alpha)

            self.data_for_asr.append((poisoned_img, self.target_class, original_label))

    def __len__(self):
        return len(self.data_for_asr)

    def __getitem__(self, idx):
        return self.data_for_asr[idx]


def run_static_patch(trainset, testset, valset, model, profile, class_names):
    print("[*] Running Static Patch Backdoor Attack...")

    # === Load configuration ===
    attack_cfg = profile.get("attack_overrides", {}).get("backdoor", {}).get("static_patch", {})
    patch_type = attack_cfg.get("patch_type", "white_square")
    patch_size_ratio = attack_cfg.get("patch_size_ratio", 0.15)
    poison_fraction = attack_cfg.get("poison_fraction", 0.1)
    target_class = attack_cfg.get("target_class")
    patch_position = attack_cfg.get("patch_position", "bottom_right")
    label_mode = attack_cfg.get("label_mode", "corrupted")
    blend_alpha = attack_cfg.get("blend_alpha", 1.0)

    input_shape = profile.get("model", {}).get("input_shape", [3, 32, 32])
    _, H, W = input_shape
    patch_size = int(W * patch_size_ratio), int(H * patch_size_ratio)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = profile.get("training", {}).get("batch_size", 64)

    # Validate target class
    if target_class is None:
        raise ValueError("Target class must be specified in the profile for backdoor attacks.")
    if not (0 <= target_class < len(class_names)):
        raise ValueError(f"Target class {target_class} is out of bounds for {len(class_names)} classes.")

    # === Generate patch ===
    dummy_image = testset[0][0]  # Get a dummy image to infer C
    patch = generate_patch(patch_type, patch_size, image_tensor=dummy_image, position=patch_position).to(device)

    # === Select poison indices from the *training set* ===
    # These are the samples from the training set that will be modified
    all_train_indices = list(range(len(trainset)))
    num_poison_samples = int(len(trainset) * poison_fraction)
    poison_train_indices = random.sample(all_train_indices, num_poison_samples)

    # Create a poisoned training dataset
    poisoned_trainset = deepcopy(trainset)
    print(
        f"[*] Applying static patch to {len(poison_train_indices)} training samples (poison_fraction={poison_fraction})...")

    for idx in tqdm(poison_train_indices, desc="Poisoning training set"):
        original_img, original_label = trainset[idx]  # Get original image and label

        # 1. Apply the patch to the original image directly here
        poisoned_img_for_train = apply_static_patch(original_img, patch, patch_position, blend_alpha)

        # 2. Determine the new label based on label_mode
        if label_mode == "corrupted":
            new_label_for_train = target_class
        elif label_mode == "clean":
            new_label_for_train = original_label  # Keep original label
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}. Must be 'corrupted' or 'clean'.")

        # 3. Call your existing update_poisoned_sample with the prepared image and label
        update_poisoned_sample(poisoned_trainset, idx, poisoned_img_for_train, new_label_for_train)

    # === Train poisoned model ===
    print("[*] Training model on poisoned data...")
    poisoned_model = deepcopy(model).to(device)

    # Extract training parameters from profile to pass to train_model
    training_cfg = profile.get("training", {})
    epochs = training_cfg.get("epochs", 3)
    lr = training_cfg.get("learning_rate", 1e-3)

    # Call train_model without 'save_path'
    train_model(poisoned_model, poisoned_trainset, valset,
                epochs=epochs,
                batch_size=batch_size,
                class_names=class_names,
                lr=lr)

    # NO CALL TO save_model HERE, as per your request.

    # The 'poisoned_model' object now holds the trained weights.
    # Assign it to 'trained_poisoned_model' for consistency with downstream code.
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

    testset_poisoned_asr = BackdoorASRDataset(testset, patch, patch_position, target_class, blend_alpha,
                                              selected_asr_indices)
    test_loader_poisoned_asr = DataLoader(testset_poisoned_asr, batch_size=batch_size, shuffle=False)

    asr_total = 0
    asr_success = 0
    asr_success_per_class = {cls_name: 0 for cls_name in class_names}
    asr_total_per_class = {cls_name: 0 for cls_name in class_names}

    example_log = []  # To store examples of poisoned images
    num_examples_saved = 0
    max_examples = 5

    # Delete examples from the previous runs
    path = "results/backdoor/static_patch/examples/"
    if os.path.exists(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(path)

    with torch.no_grad():
        for i, (img, target_cls, original_cls) in enumerate(tqdm(test_loader_poisoned_asr, desc="Calculating ASR")):
            img = img.to(device)
            output = trained_poisoned_model(img)
            pred = output.argmax(1)  # Model's prediction

            for j in range(img.shape[0]):  # Iterate per sample in the batch
                current_original_class_name = class_names[original_cls[j].item()]

                asr_total_per_class[current_original_class_name] += 1
                asr_total += 1

                if pred[j].item() == target_cls[j].item():
                    asr_success_per_class[current_original_class_name] += 1
                    asr_success += 1

                    if num_examples_saved < max_examples:
                        img_to_save = img[j].cpu().squeeze().permute(1, 2, 0).numpy() if img[j].dim() == 3 else img[
                            j].cpu().squeeze().numpy()
                        img_to_save = (img_to_save * 255).astype("uint8")
                        save_path = f"results/backdoor/static_patch/examples/poison_{selected_asr_indices[i * batch_size + j]}_{current_original_class_name}.png"
                        plt.imsave(save_path, img_to_save)

                        example_log.append({
                            "original_idx": selected_asr_indices[i * batch_size + j] if (i * batch_size + j) < len(
                                selected_asr_indices) else -1,
                            "original_class": original_cls[j].item(),
                            "original_class_name": current_original_class_name,
                            "predicted_class": pred[j].item(),
                            "predicted_class_name": class_names[pred[j].item()],
                            "target_class": target_class,
                            "filename": f"poison_{selected_asr_indices[i * batch_size + j]}_{current_original_class_name}.png"
                        })
                        num_examples_saved += 1

    asr = asr_success / asr_total if asr_total > 0 else 0.0

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
    results_dir = "results/backdoor/static_patch"
    os.makedirs(results_dir, exist_ok=True)  # Ensure the directory exists

    result = {
        "attack_type": "static_patch",
        "patch_type": patch_type,
        "patch_size_ratio": patch_size_ratio,
        "patch_position": patch_position,
        "poison_fraction": poison_fraction,
        "label_mode": label_mode,
        "blend_alpha": blend_alpha,
        "target_class": target_class,
        "target_class_name": class_names[target_class],
        "accuracy_clean_testset": clean_acc,
        "per_class_clean": clean_per_class,
        "attack_success_rate": asr,
        "attack_success_numerator": asr_success,
        "attack_success_denominator": asr_total,
        "per_class_attack_success_rate": calculated_asr_per_class,
        "per_class_asr_numerator": asr_success_per_class,
        "per_class_asr_denominator": asr_total_per_class,
        "example_poisoned_samples": example_log
    }

    with open(os.path.join(results_dir, "static_patch_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("[✔] Metrics saved to results/backdoor/static_patch/static_patch_metrics.json")

    # Generate static patch report
    generate_static_patch_report(
        os.path.join(results_dir, "static_patch_metrics.json"),
        os.path.join(results_dir, "static_patch_report.md"),
        class_names
    )