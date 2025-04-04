import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
import random
import os
import json
from attacks.utils import train_model, evaluate_model, get_class_labels


def flip_labels(dataset, flip_rate=0.1, target_class=None, flip_to_class=None):
    poisoned_dataset = deepcopy(dataset)
    targets = poisoned_dataset.dataset.targets  # .dataset needed due to random_split
    indices = poisoned_dataset.indices

    indices_to_flip = []

    if target_class is not None:
        indices_to_flip = [i for i in indices if targets[i] == target_class]
    else:
        indices_to_flip = list(indices)

    num_to_flip = int(len(indices_to_flip) * flip_rate)
    indices_to_flip = random.sample(indices_to_flip, num_to_flip)

    for idx in indices_to_flip:
        original = targets[idx]
        if target_class is not None and flip_to_class is not None:
            targets[idx] = flip_to_class
        else:
            targets[idx] = random.choice([i for i in range(10) if i != original])

    return poisoned_dataset

def run(trainset, testset, valset, model, profile):
    cfg = profile["threat_model"]
    goal = cfg.get("attack_goal", "untargeted")
    data_source = cfg.get("training_data_source", "internal_clean")
    classes = get_class_labels(trainset)

    if data_source == "user_generated":
        if goal == "targeted":
            target_class = classes[0]
            flip_to_class = classes[-1]
            flip_rate = 0.05
        else:  # untargeted
            target_class = None
            flip_to_class = None
            flip_rate = 0.08

    elif data_source == "mixed":
        if goal == "targeted":
            target_class = classes[0]
            flip_to_class = classes[-1]
            flip_rate = 0.04
        else:
            target_class = None
            flip_to_class = None
            flip_rate = 0.08

    elif data_source == "external_public":
        if goal == "targeted":
            target_class = classes[0]
            flip_to_class = classes[-1]
            flip_rate = 0.08
        else:
            target_class = None
            flip_to_class = None
            flip_rate = 0.1


    else:  # internal_clean or unknown
        target_class = None
        flip_to_class = None
        flip_rate = 0.05


    print(f"[*] Applying label flipping attack: rate={flip_rate}, target={target_class}→{flip_to_class}")
    poisoned_trainset = flip_labels(trainset, flip_rate, target_class, flip_to_class)

    print("[*] Training model on poisoned dataset...")
    train_model(model, poisoned_trainset, valset, epochs=3)

    acc = evaluate_model(model, testset)
    print(f"[+] Accuracy after poisoning: {acc:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/data_poisoning_metrics.json", "w") as f:
        json.dump({
            "attack_type": "label_flipping",
            "accuracy_after_attack": acc,
            "flip_rate": flip_rate,
            "target_class": target_class,
            "flip_to_class": flip_to_class
        }, f, indent=2)
