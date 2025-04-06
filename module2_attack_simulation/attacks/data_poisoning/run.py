import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from collections import defaultdict
import random
import os
import json

from attacks.utils import train_model, evaluate_model, get_class_labels, save_flip_examples
from generate_data_poisoning_report import generate_data_poisoning_report


def flip_labels(dataset, flip_rate=0.1, strategy="one_to_one", source_class=None, target_class=None):
    poisoned_dataset = deepcopy(dataset)
    targets = poisoned_dataset.dataset.targets
    indices = poisoned_dataset.indices

    flip_log = []
    flip_map = defaultdict(int)

    if strategy == "one_to_one":
        if source_class is None or target_class is None:
            raise ValueError("one_to_one flipping requires both source_class and target_class")
        indices_to_flip = [i for i in indices if targets[i] == source_class]
        selected_indices = random.sample(indices_to_flip, int(len(indices_to_flip) * flip_rate))

        for idx in selected_indices:
            original = int(targets[idx])
            targets[idx] = target_class
            flip_log.append({"index": idx, "original_label": original, "new_label": target_class})
            flip_map[f"{original}->{target_class}"] += 1

    elif strategy == "many_to_one":
        if target_class is None:
            raise ValueError("many_to_one flipping requires a target_class")
        indices_to_flip = [i for i in indices if targets[i] != target_class]
        selected_indices = random.sample(indices_to_flip, int(len(indices_to_flip) * flip_rate))

        for idx in selected_indices:
            original = int(targets[idx])
            targets[idx] = target_class
            flip_log.append({"index": idx, "original_label": original, "new_label": target_class})
            flip_map[f"{original}->{target_class}"] += 1

    elif strategy == "fully_random":
        selected_indices = random.sample(indices, int(len(indices) * flip_rate))
        num_classes = len(set(int(targets[i]) for i in indices))
        for idx in selected_indices:
            original = int(targets[idx])
            new_label = random.choice([c for c in range(num_classes) if c != original])
            targets[idx] = new_label
            flip_log.append({"index": idx, "original_label": original, "new_label": new_label})
            flip_map[f"{original}->{new_label}"] += 1

    else:
        raise ValueError(f"Unknown flipping strategy: {strategy}")

    return poisoned_dataset, flip_log, dict(flip_map)


def run(trainset, testset, valset, model, profile):
    print("[*] Running label flipping attack from profile configuration...")

    attack_cfg = profile.get("attack_overrides", {}).get("data_poisoning", {})
    strategy = attack_cfg.get("strategy")
    flip_rate = attack_cfg.get("flip_rate")
    source_class = attack_cfg.get("source_class")
    target_class = attack_cfg.get("target_class")

    if strategy not in ["one_to_one", "many_to_one", "fully_random"]:
        raise ValueError("Invalid or missing strategy in attack_overrides.data_poisoning")

    if flip_rate is None:
        raise ValueError("Missing flip_rate in attack_overrides.data_poisoning")

    classes = get_class_labels(trainset)

    if strategy == "many_to_one" and target_class is None:
        target_class = random.choice(classes)

    if strategy == "one_to_one":
        if source_class is None:
            source_class = random.choice(classes)
        if target_class is None:
            target_class = random.choice([c for c in classes if c != source_class])

    print(f"[*] Applying label flipping attack")
    print(f"    Strategy: {strategy}, Flip rate: {flip_rate}")
    if strategy == "one_to_one":
        print(f"    Flipping: {source_class} -> {target_class}")
    elif strategy == "many_to_one":
        print(f"    Flipping: * -> {target_class}")
    else:
        print("    Flipping: random class → random class")

    poisoned_trainset, flip_log, flip_map = flip_labels(
        trainset,
        flip_rate=flip_rate,
        strategy=strategy,
        source_class=source_class,
        target_class=target_class
    )

    print("[*] Training model on poisoned dataset...")
    train_model(model, poisoned_trainset, valset, epochs=3)

    acc = evaluate_model(model, testset)
    print(f"[+] Accuracy after poisoning: {acc:.4f}")

    os.makedirs("results", exist_ok=True)
    save_flip_examples(trainset.dataset, flip_log, num_examples=5)

    result = {
        "attack_type": "label_flipping",
        "flipping_strategy": strategy,
        "accuracy_after_attack": acc,
        "flip_rate": flip_rate,
        "source_class": source_class,
        "target_class": target_class,
        "num_flipped": len(flip_log),
        "flipping_map": flip_map,
        "example_flips": flip_log[:10]
    }

    with open("results/data_poisoning_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("[✔] Data poisoning metrics saved to results/data_poisoning_metrics.json")

    generate_data_poisoning_report(
        json_file="results/data_poisoning_metrics.json",
        md_file="results/data_poisoning_report.md"
    )
    print("[✔] Data poisoning report generated at results/data_poisoning_report.md")
