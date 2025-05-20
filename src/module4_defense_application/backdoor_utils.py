import os
import sys
import random
import torch
from copy import deepcopy

# Add module2 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "module2_attack_simulation")))

from attacks.backdoor.patch_utils import generate_patch, apply_static_patch
from attacks.backdoor.patch_utils import apply_trigger
from torchvision.transforms.functional import to_tensor

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

    # Generate patch
    patch = generate_patch(patch_type, patch_size)

    # Poison training set
    poisoned_trainset = deepcopy(trainset)
    if label_mode == "clean":
        eligible_indices = [i for i in range(len(poisoned_trainset)) if poisoned_trainset[i][1] == target_class]
    else:
        eligible_indices = list(range(len(poisoned_trainset)))

    total_poison = int(len(poisoned_trainset) * poison_fraction)
    poison_indices = random.sample(eligible_indices, min(total_poison, len(eligible_indices)))

    for idx in poison_indices:
        image, label = poisoned_trainset[idx]
        current_patch = patch
        if patch_type == "random_noise":
            current_patch = generate_patch(patch_type, patch_size, image, patch_position)
        patched_img = apply_static_patch(image.clone(), current_patch.clone(), position=patch_position, blend_alpha=blend_alpha)
        new_label = label if label_mode == "clean" else target_class
        poisoned_trainset[idx] = (patched_img, new_label)

    # Patch test set for ASR evaluation
    patched_testset = deepcopy(testset)
    for idx in range(len(patched_testset)):
        image, label = patched_testset[idx]
        current_patch = patch
        if patch_type == "random_noise":
            current_patch = generate_patch(patch_type, patch_size, image, patch_position)
        patched_img = apply_static_patch(image.clone(), current_patch.clone(), position=patch_position, blend_alpha=blend_alpha)
        patched_testset[idx] = (patched_img, label)

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

    # Load trigger and mask
    trigger = torch.load(os.path.join(
        os.path.dirname(__file__),
        "..","module2_attack_simulation/results/backdoor/learned_trigger/trigger.pt"))
    mask = torch.load(os.path.join(
        os.path.dirname(__file__),
        "..","module2_attack_simulation/results/backdoor/learned_trigger/mask.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trigger, mask = trigger.to(device), mask.to(device)

    def apply_learned_trigger(x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return apply_trigger(x.to(device), trigger, mask)[0].cpu()

    # Poison training set
    poisoned_trainset = deepcopy(trainset)
    eligible_indices = [i for i in range(len(poisoned_trainset)) if label_mode == "clean" and poisoned_trainset[i][1] == target_class or label_mode == "corrupted"]
    total_poison = int(len(poisoned_trainset) * poison_fraction)
    poison_indices = random.sample(eligible_indices, min(total_poison, len(eligible_indices)))

    for idx in poison_indices:
        image, label = poisoned_trainset[idx]
        patched = apply_learned_trigger(image)
        new_label = label if label_mode == "clean" else target_class
        poisoned_trainset[idx] = (patched, new_label)

    # Patch test set
    patched_testset = deepcopy(testset)
    for idx in range(len(patched_testset)):
        image, label = patched_testset[idx]
        patched = apply_learned_trigger(image)
        patched_testset[idx] = (patched, label)

    trigger_info = {
        "trigger": trigger.cpu(),
        "mask": mask.cpu(),
        "target_class": target_class,
        "label_mode": label_mode
    }

    return poisoned_trainset, patched_testset, trigger_info
