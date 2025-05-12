import os
import sys
import json
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","module2_attack_simulation")))
from attacks.utils import train_model, evaluate_model, get_class_labels,load_model_cfg_from_profile
from attacks.data_poisoning.label_flipping.run_label_flipping import flip_labels


def apply_data_cleaning(trainset, params):
    """
    Placeholder for data cleaning logic.
    Currently removes no samples – must be updated in final version.
    """
    print(f"[*] Applying data cleaning with method={params['method']} and threshold={params['threshold']}")
    # TODO: implement actual data cleaning logic
    return trainset


def run_data_cleaning_defense(profile, trainset, testset, valset, class_names):
    print("[*] Running data_cleaning defense for label_flipping...")

    attack_cfg = profile["attack_overrides"]["data_poisoning"]["label_flipping"]
    defense_cfg = profile["defense_config"]["data_poisoning"]["label_flipping"]["data_cleaning"]

    model = load_model_cfg_from_profile(profile)
    class_ids = get_class_labels(trainset)

    # 1. Apply label flipping
    poisoned_trainset, flip_log, flip_map = flip_labels(
        trainset,
        flip_rate=attack_cfg["flip_rate"],
        strategy=attack_cfg["strategy"],
        source_class=attack_cfg.get("source_class"),
        target_class=attack_cfg.get("target_class"),
        class_names=class_names
    )

    # 2. Apply data cleaning
    cleaned_dataset = apply_data_cleaning(poisoned_trainset, defense_cfg)

    # 3. Train model on cleaned dataset
    train_model(model, cleaned_dataset, valset, epochs=3, class_names=class_names)

    # 4. Evaluate on clean test set
    acc, per_class = evaluate_model(model, testset, class_names=class_names)

    # 5. Save metrics
    os.makedirs("results/module4_defenses", exist_ok=True)
    results = {
        "defense": "data_cleaning",
        "attack": "label_flipping",
        "accuracy_after_defense": acc,
        "per_class_accuracy": per_class,
        "flip_map": flip_map,
        "num_flipped": len(flip_log),
        "cleaning_params": defense_cfg
    }
    with open("results/module4_defenses/data_cleaning_label_flipping.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[✔] Defense results saved.")
