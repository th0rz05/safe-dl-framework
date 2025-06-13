import os
import sys
import random
from collections import defaultdict

import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "module2_attack_simulation")))

from attacks.backdoor.patch_utils import generate_patch, apply_static_patch, apply_trigger


def simulate_static_patch_attack(profile, trainset, testset, class_names):
    """
    Reproduce the static patch attack using profile config.
    Returns poisoned training set, patched test set, and trigger info.
    """




    cfg = profile.get("attack_overrides", {}).get("backdoor", {}).get("static_patch", {})
    patch_type = cfg.get("patch_type", "white_square")
    patch_size_ratio = cfg.get("patch_size_ratio", 0.15)
    poison_fraction = cfg.get("poison_fraction", 0.1)
    target_class = cfg.get("target_class")
    patch_position = cfg.get("patch_position", "bottom_right")
    label_mode = cfg.get("label_mode", "corrupted")
    blend_alpha = cfg.get("blend_alpha", 1.0)

    input_shape = profile.get("model", {}).get("input_shape", [3, 32, 32])
    _, H, W = input_shape
    patch_size = int(W * patch_size_ratio), int(H * patch_size_ratio)



    patch = generate_patch(patch_type, patch_size)

    poisoned_trainset = deepcopy(trainset)
    if label_mode == "clean":
        eligible_indices = [i for i in range(len(poisoned_trainset)) if poisoned_trainset[i][1] == target_class]
    else:
        eligible_indices = list(range(len(poisoned_trainset)))

    total_poison = int(len(poisoned_trainset) * poison_fraction)
    poison_indices = random.sample(eligible_indices, min(total_poison, len(eligible_indices)))

    print("[*] Simulating Static Patch Backdoor Attack...")
    print(f"    → Patch type         : {patch_type}")
    print(f"    → Patch size ratio   : {patch_size_ratio}")
    print(f"    → Poison fraction    : {poison_fraction}")
    print(f"    → Target class       : {target_class} ({class_names[target_class]})")
    print(f"    → Label mode         : {label_mode}")
    print(f"    → Blending alpha     : {blend_alpha}")
    print(f"    → Patch position     : {patch_position}")
    print(f"    → Number of poisoned samples: {len(poison_indices)}")

    for local_idx in poison_indices:
        image, label = poisoned_trainset[local_idx]
        current_patch = patch
        if patch_type == "random_noise":
            current_patch = generate_patch(patch_type, patch_size, image, patch_position)
        patched_img = apply_static_patch(image.clone(), current_patch.clone(), position=patch_position,
                                         blend_alpha=blend_alpha)
        new_label = label if label_mode == "clean" else target_class

        if isinstance(poisoned_trainset, torch.utils.data.Subset):
            base_dataset = poisoned_trainset.dataset
            real_idx = poisoned_trainset.indices[local_idx]
        else:
            base_dataset = poisoned_trainset
            real_idx = local_idx

        image_np = patched_img.detach().cpu()
        if image_np.shape[0] in [1, 3]:
            image_np = image_np.permute(1, 2, 0)
        if image_np.max() <= 1.0:
            image_np = image_np * 255
        image_np = image_np.byte().numpy()

        if hasattr(base_dataset, "data") and isinstance(base_dataset.data, np.ndarray):
            base_dataset.data[real_idx] = image_np
        elif hasattr(base_dataset, "data") and isinstance(base_dataset.data, torch.Tensor):
            base_dataset.data[real_idx] = torch.from_numpy(image_np)
        elif hasattr(base_dataset, "tensors"):
            tensors = list(base_dataset.tensors)
            tensors[0][real_idx] = patched_img
            tensors[1][real_idx] = new_label
            base_dataset.tensors = tuple(tensors)
        else:
            raise TypeError("Unsupported dataset type for poison update.")

        if hasattr(base_dataset, "targets"):
            if isinstance(base_dataset.targets, list):
                base_dataset.targets[real_idx] = new_label
            elif isinstance(base_dataset.targets, torch.Tensor):
                base_dataset.targets[real_idx] = new_label

    # Patch test set → new dataset
    patched_images = []
    patched_labels = []

    for idx in range(len(testset)):
        image, label = testset[idx]
        current_patch = patch
        if patch_type == "random_noise":
            current_patch = generate_patch(patch_type, patch_size, image, patch_position)
        patched_img = apply_static_patch(image.clone(), current_patch.clone(), position=patch_position,
                                         blend_alpha=blend_alpha)
        patched_images.append(patched_img.unsqueeze(0))
        patched_labels.append(torch.tensor(label))

    print(f"[✔] Patched training set with static patch applied to {len(poison_indices)} samples.")

    patched_testset = TensorDataset(torch.cat(patched_images), torch.stack(patched_labels))

    trigger_info = {
        "patch": patch,
        "patch_type": patch_type,
        "position": patch_position,
        "blend_alpha": blend_alpha,
        "target_class": target_class,
        "label_mode": label_mode
    }

    return poisoned_trainset, patched_testset, trigger_info


def simulate_learned_trigger_attack(profile, trainset, testset, class_names):
    """
    Apply learned trigger and mask saved from Module 2.
    Returns poisoned training set, patched test set, and trigger info.
    """

    cfg = profile.get("attack_overrides", {}).get("backdoor", {}).get("learned_trigger", {})
    target_class = cfg.get("target_class")
    poison_fraction = cfg.get("poison_fraction", 0.1)
    label_mode = cfg.get("label_mode", "corrupted")


    trigger = torch.load(os.path.join(
        os.path.dirname(__file__), "..", "module2_attack_simulation",
        "results/backdoor/learned_trigger/trigger.pt"))
    mask = torch.load(os.path.join(
        os.path.dirname(__file__), "..", "module2_attack_simulation",
        "results/backdoor/learned_trigger/mask.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trigger, mask = trigger.to(device), mask.to(device)

    def apply_learned_trigger(x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return apply_trigger(x.to(device), trigger, mask)[0].cpu()

    poisoned_trainset = deepcopy(trainset)
    eligible_indices = [
        i for i in range(len(poisoned_trainset))
        if label_mode == "clean" and poisoned_trainset[i][1] == target_class or label_mode == "corrupted"
    ]
    total_poison = int(len(poisoned_trainset) * poison_fraction)
    poison_indices = random.sample(eligible_indices, min(total_poison, len(eligible_indices)))

    print("[*] Simulating Learned Trigger Backdoor Attack...")
    print(f"    → Poison fraction    : {poison_fraction}")
    print(f"    → Target class       : {target_class} ({class_names[target_class]})")
    print(f"    → Label mode         : {label_mode}")
    print(f"    → Trigger loaded from: trigger.pt")
    print(f"    → Mask loaded from   : mask.pt")
    print(f"    → Number of poisoned samples: {len(poison_indices)}")

    for local_idx in poison_indices:
        image, label = poisoned_trainset[local_idx]
        patched = apply_learned_trigger(image)
        new_label = label if label_mode == "clean" else target_class

        if isinstance(poisoned_trainset, torch.utils.data.Subset):
            base_dataset = poisoned_trainset.dataset
            real_idx = poisoned_trainset.indices[local_idx]
        else:
            base_dataset = poisoned_trainset
            real_idx = local_idx

        image_np = patched.detach().cpu()
        if image_np.shape[0] in [1, 3]:
            image_np = image_np.permute(1, 2, 0)
        if image_np.max() <= 1.0:
            image_np = image_np * 255
        image_np = image_np.byte().numpy()

        if hasattr(base_dataset, "data") and isinstance(base_dataset.data, np.ndarray):
            base_dataset.data[real_idx] = image_np
        elif hasattr(base_dataset, "data") and isinstance(base_dataset.data, torch.Tensor):
            base_dataset.data[real_idx] = torch.from_numpy(image_np)
        elif hasattr(base_dataset, "tensors"):
            tensors = list(base_dataset.tensors)
            tensors[0][real_idx] = patched
            tensors[1][real_idx] = new_label
            base_dataset.tensors = tuple(tensors)
        else:
            raise TypeError("Unsupported dataset type for poison update.")

        if hasattr(base_dataset, "targets"):
            if isinstance(base_dataset.targets, list):
                base_dataset.targets[real_idx] = new_label
            elif isinstance(base_dataset.targets, torch.Tensor):
                base_dataset.targets[real_idx] = new_label

    # Patch test set → new dataset
    patched_images = []
    patched_labels = []

    for idx in range(len(testset)):
        image, label = testset[idx]
        patched = apply_learned_trigger(image)
        patched_images.append(patched.unsqueeze(0))
        patched_labels.append(torch.tensor(label))

    print(f"[✔] Patched training set with learned trigger.")

    patched_testset = TensorDataset(torch.cat(patched_images), torch.stack(patched_labels))

    trigger_info = {
        "trigger": trigger.cpu(),
        "mask": mask.cpu(),
        "target_class": target_class,
        "label_mode": label_mode
    }

    return poisoned_trainset, patched_testset, trigger_info


def evaluate_backdoor_asr(model, patched_testset, target_class, class_names=None, batch_size=64, silent=False, prefix="[Eval ASR]"):
    """
    Evaluate the Attack Success Rate (ASR) for backdoor attacks.
    This measures how many samples with a trigger are misclassified to the target_class,
    considering only samples whose original label was NOT the target_class.

    Args:
        model (torch.nn.Module): The model to evaluate.
        patched_testset (torch.utils.data.Dataset): The test set with backdoor triggers applied.
        target_class (int): The target class of the backdoor attack.
        class_names (list, optional): List of class names for logging. Defaults to None.
        batch_size (int, optional): Batch size for evaluation. Defaults to 64.
        silent (bool, optional): If True, suppresses tqdm progress bar. Defaults to False.
        prefix (str, optional): Prefix for the progress bar and print statements. Defaults to "[Eval ASR]".

    Returns:
        tuple: (overall_asr (float), per_class_asr (dict))
               overall_asr: The overall attack success rate (0-1).
               per_class_asr: A dictionary with ASR breakdown by original class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    loader = DataLoader(patched_testset, batch_size=batch_size, shuffle=False)

    asr_success = 0
    asr_total = 0 # Only counts samples not originally of target_class
    per_original_class_asr_success = defaultdict(int)
    per_original_class_asr_total = defaultdict(int)

    iterator = loader if silent else tqdm(loader, desc=prefix, leave=False)

    with torch.no_grad():
        for x, original_y in iterator:
            x = x.to(device) # Image with trigger
            # original_y contains the true labels of the images *before* patching/triggering

            preds = model(x).argmax(1) # Model's prediction

            for i in range(len(original_y)):
                original_label = original_y[i].item()
                predicted_label = preds[i].item()

                # Only consider samples whose original label is NOT the target_class
                # This ensures we measure the *deviation* caused by the backdoor.
                if original_label != target_class:
                    asr_total += 1
                    per_original_class_asr_total[original_label] += 1

                    if predicted_label == target_class:
                        asr_success += 1
                        per_original_class_asr_success[original_label] += 1

    overall_asr = asr_success / asr_total if asr_total > 0 else 0.0
    print(f"{prefix} Overall ASR: {overall_asr:.4f} ({asr_success}/{asr_total})")

    per_class_asr = {}
    for cls_idx in sorted(per_original_class_asr_total.keys()):
        total_for_class = per_original_class_asr_total[cls_idx]
        success_for_class = per_original_class_asr_success.get(cls_idx, 0)
        acc_for_class = success_for_class / total_for_class if total_for_class > 0 else 0.0
        name = class_names[cls_idx] if class_names else str(cls_idx)
        per_class_asr[name] = round(acc_for_class, 4)
        print(f"  - Original Class {name:>10} ASR: {acc_for_class:.4f} ({success_for_class}/{total_for_class})")

    return overall_asr, per_class_asr