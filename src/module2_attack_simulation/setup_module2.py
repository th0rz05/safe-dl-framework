import questionary
from questionary import Choice
import yaml
import os
from dataset_loader import list_builtin_datasets, load_builtin_dataset, load_user_dataset
from model_loader import get_builtin_model
from glob import glob
import torch
import random


def select_dataset():
    dataset_shapes = {
        "mnist": [1, 28, 28],
        "fashionmnist": [1, 28, 28],
        "cifar10": [3, 32, 32],
        "cifar100": [3, 32, 32],
        "svhn": [3, 32, 32],
        "kmnist": [1, 28, 28],
        "emnist": [1, 28, 28]
    }

    choices = [f"{name} (built-in)" for name in list_builtin_datasets()] + ["user_dataset.py"]
    selected = questionary.select("Select a dataset:", choices=choices).ask()

    if "user" in selected:
        dataset_info = {"type": "custom", "name": "user_dataset.py"}
        try:
            _, _, _,class_names, num_classes = load_user_dataset()
        except Exception as e:
            print(f"[!] Could not load custom dataset: {e}")
            num_classes = None
        input_shape = [1, 28, 28]  # fallback default
    else:
        dataset_name = selected.split(" ")[0].lower()
        dataset_info = {"type": "builtin", "name": dataset_name}
        try:
            _, _, _, class_names, num_classes  = load_builtin_dataset(dataset_name)
        except Exception as e:
            print(f"[!] Could not load built-in dataset '{dataset_name}': {e}")
            num_classes = None
        input_shape = dataset_shapes.get(dataset_name, [1, 28, 28])

    if num_classes is None:
        num_classes = int(questionary.text("How many output classes does your dataset have?", default="10").ask())
        
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
        

    return dataset_info, num_classes, input_shape, class_names


def select_model(num_classes, input_shape):
    choices = ["cnn", "mlp", "resnet18", "resnet50", "vit", "user_model.py"]
    selected = questionary.select("Select a model:", choices=choices).ask()

    model_info = {
        "type": "custom" if "user" in selected else "builtin",
        "name": selected,
        "num_classes": num_classes
    }

    # Handle input shape for built-in models
    if model_info["type"] == "builtin":
        if questionary.confirm("Do you want to customize the input shape?", default=False).ask():
            channels = int(questionary.text("Number of input channels:", default=str(input_shape[0])).ask())
            height = int(questionary.text("Image height:", default=str(input_shape[1])).ask())
            width = int(questionary.text("Image width:", default=str(input_shape[2])).ask())
            model_info["input_shape"] = [channels, height, width]
        else:
            model_info["input_shape"] = input_shape

        # Optional parameters for CNN and MLP
        if selected == "cnn":
            filters = questionary.text("Number of conv filters (default: 32):", default="32").ask()
            hidden = questionary.text("Hidden layer size (default: 128):", default="128").ask()
            model_info["params"] = {
                "conv_filters": int(filters),
                "hidden_size": int(hidden)
            }

        elif selected == "mlp":
            hidden = questionary.text("Hidden layer size (default: 128):", default="128").ask()
            input_size = questionary.text("Input size (default: 784):", default="784").ask()
            model_info["params"] = {
                "hidden_size": int(hidden),
                "input_size": int(input_size)
            }

    return model_info



def select_profile():
    profiles = glob("../profiles/*.yaml")
    if not profiles:
        print("No profiles found in ../profiles/")
        return None

    selected = questionary.select("Select a threat profile to use:", choices=profiles).ask()
    return selected

def configure_label_flipping(profile_data, class_names):
    print("\n=== Configuring Label Flipping ===\n")
    cfg = profile_data.get("threat_model", {})
    goal = cfg.get("attack_goal", "untargeted")
    data_source = cfg.get("training_data_source", "internal_clean")
    num_classes = profile_data.get("model", {}).get("num_classes", 10)
    classes = list(range(num_classes))

    # Default suggestion
    if goal == "targeted":
        if data_source == "user_generated":
            strategy = "one_to_one"
            flip_rate = 0.05
        else:
            strategy = "many_to_one"
            flip_rate = 0.08
    else:
        strategy = "fully_random"
        flip_rate = 0.1 if data_source == "external_public" else 0.08

    source_class = None
    target_class = None

    if strategy == "many_to_one":
        target_class = random.choice(classes)
    elif strategy == "one_to_one":
        source_class = random.choice(classes)
        target_class = random.choice([c for c in classes if c != source_class])

    print(f"Suggested strategy: {strategy}")
    print(f"Suggested flip_rate: {flip_rate}")
    if source_class is not None:
        print(f"Suggested source class: {source_class} – {class_names[source_class]}")
    if target_class is not None:
        print(f"Suggested target class: {target_class} – {class_names[target_class]}")

    if not questionary.confirm("Do you want to accept these suggestions?").ask():
        strategy = questionary.select(
            "Choose flipping strategy:",
            choices=[
                Choice("Fully random (random→random)", value="fully_random"),
                Choice("Random to fixed (many→one)", value="many_to_one"),
                Choice("Fixed to fixed (one→one)", value="one_to_one")
            ]
        ).ask()

        flip_rate = float(questionary.text("Flip rate (e.g., 0.08):", default=str(flip_rate)).ask())

        class_options = [f"{i} – {name}" for i, name in enumerate(class_names)]

        if strategy == "many_to_one":
            if questionary.confirm("Pick target class randomly?").ask():
                target_class = random.choice(classes)
            else:
                target_class_str = questionary.select("Select target class (flip TO):", choices=class_options).ask()
                target_class = int(target_class_str.split(" ")[0])
            source_class = None

        elif strategy == "one_to_one":
            if questionary.confirm("Pick source class randomly?").ask():
                source_class = random.choice(classes)
            else:
                source_class_str = questionary.select("Select source class (flip FROM):", choices=class_options).ask()
                source_class = int(source_class_str.split(" ")[0])

            if questionary.confirm("Pick target class randomly?").ask():
                target_class = random.choice([c for c in classes if c != source_class])
            else:
                target_class_str = questionary.select("Select target class (flip TO):", choices=class_options).ask()
                target_class = int(target_class_str.split(" ")[0])
        else:
            source_class = None
            target_class = None

    cfg = {
                "strategy": strategy,
                "flip_rate": flip_rate,
                "source_class": source_class,
                "target_class": target_class
            }

    return cfg

def configure_clean_label(profile_data,class_names):
    print("\n=== Configuring Clean Label Attack ===\n")

    cfg = profile_data.get("threat_model", {})
    goal = cfg.get("attack_goal", "untargeted")
    data_source = cfg.get("training_data_source", "internal_clean")
    num_classes = profile_data.get("model", {}).get("num_classes", len(class_names))

    # ======== SUGGESTION LOGIC ========
    if goal == "targeted":
        target_class = random.randint(0, num_classes - 1)
        fraction_poison = 0.05
    else:
        target_class = None
        fraction_poison = 0.08

    if data_source == "user_generated":
        perturbation_method = "overlay"
    else:
        perturbation_method = "feature_collision"

    max_iterations = 100
    epsilon = 0.1
    source_selection = "random"

    # ======== SHOW SUGGESTION TO USER ========
    print("Suggested configuration:")
    print(f"  - Poison fraction: {fraction_poison}")
    if target_class is not None:
        print(f"  - Target class: {target_class} – {class_names[target_class]}")
    else:
        print("  - Target class: None (untargeted)")
    print(f"  - Perturbation method: {perturbation_method}")
    print(f"  - Max iterations: {max_iterations}")
    print(f"  - Epsilon: {epsilon}")
    print(f"  - Source selection: {source_selection}")

    if not questionary.confirm("Do you want to accept these suggestions?").ask():
        fraction_poison = float(questionary.text("Poisoning fraction (e.g., 0.05):", default=str(fraction_poison)).ask())

        class_options = [f"{i} – {name}" for i, name in enumerate(class_names)]
        if questionary.confirm("Is this a targeted attack?", default=(target_class is not None)).ask():
            target_class_str = questionary.select("Select target class:", choices=class_options).ask()
            target_class = int(target_class_str.split(" ")[0])
        else:
            target_class = None

        perturbation_method = questionary.select(
            "Select perturbation method:",
            choices=["overlay", "feature_collision", "noise"]
        ).ask()

        max_iterations = int(questionary.text("Max optimization iterations:", default=str(max_iterations)).ask())
        epsilon = float(questionary.text("Perturbation epsilon:", default=str(epsilon)).ask())

        source_selection = questionary.select(
            "Source image selection strategy:",
            choices=["random", "most_confident", "least_confident"]
        ).ask()

    return {
        "fraction_poison": fraction_poison,
        "target_class": target_class,
        "perturbation_method": perturbation_method,
        "max_iterations": max_iterations,
        "epsilon": epsilon,
        "source_selection": source_selection
    }

def configure_static_patch(profile_data, class_names):
    print("\n=== Configuring Static Patch Backdoor Attack ===\n")

    cfg = profile_data.get("threat_model", {})
    goal = cfg.get("attack_goal", "targeted")
    data_source = cfg.get("training_data_source", "internal_clean")
    num_classes = profile_data.get("model", {}).get("num_classes", len(class_names))
    classes = list(range(num_classes))


    # === Default Suggestions ===
    if goal == "targeted":
        target_class = random.randint(0, num_classes - 1)
    else:
        target_class = random.randint(0, num_classes - 1)
        print("[!] Warning: Static patch attack is designed for targeted scenarios. Running untargeted for testing purposes.")

    patch_size_ratio = 0.2 if data_source == "user_generated" else 0.15
    patch_type = "white_square"
    poison_fraction = 0.1 if data_source == "external_public" else 0.05
    patch_position = "bottom_right"
    label_mode = "corrupted"
    blend_alpha = 1.0  # full visibility

    # === Show Suggestions ===
    print("Suggested configuration:")
    print(f"  - Target class: {target_class} – {class_names[target_class]}")
    print(f"  - Patch type: {patch_type}")
    print(f"  - Patch size (relative): {patch_size_ratio}")
    print(f"  - Poison fraction: {poison_fraction}")
    print(f"  - Patch position: {patch_position}")
    print(f"  - Label mode: {label_mode}")
    print(f"  - Blend alpha: {blend_alpha}")

    if not questionary.confirm("Do you want to accept these suggestions?").ask():
        class_options = [f"{i} – {name}" for i, name in enumerate(class_names)]

        if questionary.confirm("Pick target class randomly?", default=True).ask():
            target_class = random.choice(classes)
        else:
            target_class_str = questionary.select("Select target class (to misclassify as):",
                                                  choices=class_options).ask()
            target_class = int(target_class_str.split(" ")[0])

        patch_type = questionary.select(
            "Select the type of patch to apply:",
            choices=["white_square", "checkerboard", "random_noise"]
        ).ask()

        patch_position = questionary.select(
            "Select where to place the patch:",
            choices=["bottom_right", "bottom_left", "top_right", "top_left", "center"]
        ).ask()

        patch_size_ratio = float(questionary.text("Patch size (as ratio of image width, e.g., 0.2):", default=str(patch_size_ratio)).ask())
        poison_fraction = float(questionary.text("Poisoning fraction (e.g., 0.1):", default=str(poison_fraction)).ask())

        label_mode = questionary.select(
            "Select label mode:",
            choices=["corrupted", "clean"]
        ).ask()

        blend_alpha = float(questionary.text("Blending alpha (0.0 to 1.0):",
                                             default=str(blend_alpha)).ask())

    return {
        "target_class": target_class,
        "patch_type": patch_type,
        "patch_position": patch_position,
        "patch_size_ratio": patch_size_ratio,
        "poison_fraction": poison_fraction,
        "label_mode": label_mode,
        "blend_alpha": blend_alpha
    }

import random
import questionary

def configure_learned_trigger(profile_data, class_names):
    print("\n=== Configuring Adversarially Learned Trigger ===\n")

    # Load threat-model context
    cfg = profile_data.get("threat_model", {})
    data_source = cfg.get("training_data_source", "internal_clean")

    # Determine number of classes
    num_classes = profile_data.get("model", {}).get("num_classes", len(class_names))

    # === Default Suggestions ===
    target_class     = random.randint(0, num_classes - 1)
    patch_size_ratio = 0.1 if data_source == "user_generated" else 0.05
    poison_fraction  = 0.05
    learning_rate    = 0.1
    epochs           = 30
    lambda_mask      = 0.001
    lambda_tv        = 0.01
    label_mode       = "corrupted"  # forced

    # === Show Defaults ===
    print("Suggested configuration:")
    print(f"  • Target class       : {target_class} – {class_names[target_class]}")
    print(f"  • Patch size ratio   : {patch_size_ratio}")
    print(f"  • Poison fraction    : {poison_fraction}")
    print(f"  • Label mode         : {label_mode} (fixed)")
    print(f"  • Learning rate      : {learning_rate}")
    print(f"  • Optimization epochs: {epochs}")
    print(f"  • Mask L1 weight     : {lambda_mask}")
    print(f"  • TV regularization  : {lambda_tv}")

    # Ask for confirmation
    if not questionary.confirm("Do you want to accept these defaults?").ask():
        # Target class selection
        target_class = int(
            questionary.select(
                "Select target class (to misclassify as):",
                choices=[f"{i} – {name}" for i, name in enumerate(class_names)],
                default=f"{target_class} – {class_names[target_class]}"
            ).ask().split(" ")[0]
        )

        # Patch size ratio
        patch_size_ratio = float(
            questionary.text(
                "Patch size ratio (fraction of image, e.g. 0.05):",
                default=str(patch_size_ratio)
            ).ask()
        )

        # Poison fraction
        poison_fraction = float(
            questionary.text(
                "Poisoning fraction (e.g. 0.05):",
                default=str(poison_fraction)
            ).ask()
        )

        # Learning rate
        learning_rate = float(
            questionary.text(
                "Learning rate for trigger optimization:",
                default=str(learning_rate)
            ).ask()
        )

        # Epochs
        epochs = int(
            questionary.text(
                "Number of optimization epochs:",
                default=str(epochs)
            ).ask()
        )

        # Mask regularization weight
        lambda_mask = float(
            questionary.text(
                "L1 weight for mask regularization:",
                default=str(lambda_mask)
            ).ask()
        )

        # TV regularization weight
        lambda_tv = float(
            questionary.text(
                "Weight for total variation regularization:",
                default=str(lambda_tv)
            ).ask()
        )

    return {
        "target_class"      : target_class,
        "patch_size_ratio"  : patch_size_ratio,
        "poison_fraction"   : poison_fraction,
        "label_mode"        : label_mode,
        "learning_rate"     : learning_rate,
        "epochs"            : epochs,
        "lambda_mask"       : lambda_mask,
        "lambda_tv"         : lambda_tv
    }

def configure_fgsm(profile_data):
    print("\n=== Configuring FGSM Attack ===\n")
    epsilon = 0.03  # default suggestion

    print(f"Suggested epsilon: {epsilon}")
    if not questionary.confirm("Do you want to accept this suggestion?").ask():
        epsilon = float(questionary.text("Epsilon (perturbation strength, e.g. 0.03):", default=str(epsilon)).ask())

    return {
        "epsilon": epsilon
    }

def configure_pgd(profile_data):
    print("\n=== Configuring PGD Attack ===\n")

    cfg = profile_data.get("threat_model", {})
    goal = cfg.get("attack_goal", "untargeted")
    data_source = cfg.get("training_data_source", "internal_clean")

    # ======= DEFAULT SUGGESTION =======
    if data_source == "user_generated":
        epsilon = 0.02
        alpha = 0.005
        num_iter = 30
    else:
        epsilon = 0.03
        alpha = 0.01
        num_iter = 40

    if goal == "targeted":
        num_iter += 10  # increase iterations for targeted attacks

    random_start = True  # always beneficial in PGD

    # ======= SHOW SUGGESTION =======
    print("Suggested configuration:")
    print(f"- Epsilon      : {epsilon}")
    print(f"- Alpha        : {alpha}")
    print(f"- Num Iterations: {num_iter}")
    print(f"- Random Start : {random_start}")

    if not questionary.confirm("Do you want to accept these suggestions?").ask():
        epsilon = float(questionary.text("Epsilon (maximum perturbation, e.g., 0.03):", default=str(epsilon)).ask())
        alpha = float(questionary.text("Alpha (step size per iteration, e.g., 0.01):", default=str(alpha)).ask())
        num_iter = int(questionary.text("Number of PGD iterations (e.g., 40):", default=str(num_iter)).ask())
        random_start = questionary.confirm("Use random start?", default=random_start).ask()

    return {
        "epsilon": epsilon,
        "alpha": alpha,
        "num_iter": num_iter,
        "random_start": random_start
    }


def configure_cw(profile_data):
    print("\n=== Configuring Carlini & Wagner (C&W) Attack ===\n")

    cfg = profile_data.get("threat_model", {})
    data_sensitivity = cfg.get("data_sensitivity", "medium")
    deployment = cfg.get("deployment_scenario", "cloud")

    # Default suggestions
    confidence = 0.1 if data_sensitivity == "high" else 0.5
    binary_search_steps = 9 if deployment == "cloud" else 5
    max_iterations = 1000 if deployment == "cloud" else 500
    learning_rate = 0.01
    initial_const = 0.001

    print("Suggested configuration:")
    print(f"  • Confidence (kappa)        : {confidence}")
    print(f"  • Binary search steps       : {binary_search_steps}")
    print(f"  • Max optimization iterations: {max_iterations}")
    print(f"  • Learning rate              : {learning_rate}")
    print(f"  • Initial constant (c)       : {initial_const}")

    if not questionary.confirm("Do you want to accept these defaults?").ask():
        confidence = float(questionary.text(
            "Confidence (kappa) - e.g., 0.0 to 1.0 (higher = stronger attack):",
            default=str(confidence)
        ).ask())

        binary_search_steps = int(questionary.text(
            "Binary search steps (higher = better L2/performance tradeoff):",
            default=str(binary_search_steps)
        ).ask())

        max_iterations = int(questionary.text(
            "Maximum optimization iterations:",
            default=str(max_iterations)
        ).ask())

        learning_rate = float(questionary.text(
            "Learning rate for Adam optimizer:",
            default=str(learning_rate)
        ).ask())

        initial_const = float(questionary.text(
            "Initial constant 'c' for balancing loss terms:",
            default=str(initial_const)
        ).ask())

    return {
        "confidence": confidence,
        "binary_search_steps": binary_search_steps,
        "max_iterations": max_iterations,
        "learning_rate": learning_rate,
        "initial_const": initial_const
    }

def configure_deepfool(profile_data):
    print("\n=== Configuring DeepFool Attack ===\n")

    cfg = profile_data.get("threat_model", {})
    model_type = cfg.get("model_type", "cnn")
    deployment = cfg.get("deployment_scenario", "cloud")

    # === Default suggestions based on context ===
    if model_type == "cnn":
        max_iter = 50
    else:
        max_iter = 30

    overshoot = 0.02 if deployment == "cloud" else 0.03

    # === Show suggested parameters ===
    print("Suggested configuration:")
    print(f"  • Max Iterations : {max_iter}")
    print(f"  • Overshoot      : {overshoot}")

    if not questionary.confirm("Do you want to accept these suggestions?").ask():
        max_iter = int(questionary.text(
            "Maximum number of iterations (e.g., 30–50):",
            default=str(max_iter)
        ).ask())

        overshoot = float(questionary.text(
            "Overshoot factor (e.g., 0.02 or 0.03):",
            default=str(overshoot)
        ).ask())

    return {
        "max_iter": max_iter,
        "overshoot": overshoot
    }

def configure_nes(profile_data):
    print("\n=== Configuring NES (Natural Evolution Strategies) Attack ===\n")

    cfg = profile_data.get("threat_model", {})
    model_access = cfg.get("model_access", "black-box")
    data_sensitivity = cfg.get("data_sensitivity", "medium")

    # ======== DEFAULT SUGGESTIONS BASED ON PROFILE ========
    if model_access == "black-box":
        sigma = 0.001  # smaller noise for subtle queries
        learning_rate = 0.01
        epsilon = 0.03
        num_queries = 1000 if data_sensitivity == "high" else 500
        batch_size = 50
    else:
        # If somehow NES is chosen with more access, more aggressive parameters
        sigma = 0.01
        learning_rate = 0.02
        epsilon = 0.05
        num_queries = 500
        batch_size = 30

    # ======== SHOW SUGGESTIONS ========
    print("Suggested configuration:")
    print(f"  • Sigma (noise stddev)  : {sigma}")
    print(f"  • Learning rate         : {learning_rate}")
    print(f"  • Epsilon (max perturb.) : {epsilon}")
    print(f"  • Max queries           : {num_queries}")
    print(f"  • Batch size (per query) : {batch_size}")

    if not questionary.confirm("Do you want to accept these defaults?").ask():
        epsilon = float(questionary.text(
            "Epsilon (maximum allowed perturbation, e.g., 0.03):",
            default=str(epsilon)
        ).ask())

        sigma = float(questionary.text(
            "Sigma (noise standard deviation for NES gradient estimation, e.g., 0.001):",
            default=str(sigma)
        ).ask())

        learning_rate = float(questionary.text(
            "Learning rate for updates (e.g., 0.01):",
            default=str(learning_rate)
        ).ask())

        num_queries = int(questionary.text(
            "Maximum number of queries allowed (e.g., 1000):",
            default=str(num_queries)
        ).ask())

        batch_size = int(questionary.text(
            "Batch size for NES gradient estimation (e.g., 50):",
            default=str(batch_size)
        ).ask())

    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "learning_rate": learning_rate,
        "num_queries": num_queries,
        "batch_size": batch_size
    }

def configure_spsa(profile_data):
    print("\n=== Configuring SPSA Attack ===\n")

    # === Get Threat Model Context ===
    threat_cfg = profile_data.get("threat_model", {})
    attack_goal = threat_cfg.get("attack_goal", "untargeted")
    model_access = threat_cfg.get("model_access", "black-box")
    deployment_scenario = threat_cfg.get("deployment_scenario", "cloud")
    data_sensitivity = threat_cfg.get("data_sensitivity", "medium")

    # === Default suggestions based on threat model ===
    if model_access == "black-box":
        batch_size = 64  # Higher batch size for smoother gradient estimates
    else:
        batch_size = 32

    if deployment_scenario in ["cloud", "api_public"]:
        num_steps = 150  # Allow more optimization steps
    else:
        num_steps = 100

    epsilon = 0.05  # General for CIFAR, normalized inputs
    delta = 0.01  # Perturbation magnitude for SPSA estimate
    learning_rate = 0.01

    # Tighten epsilon if data is very sensitive
    if data_sensitivity == "high":
        epsilon = 0.03

    suggested_max_samples = 500

    # === Show suggested values ===
    print("Suggested configuration based on threat model:")
    print(f"  • Epsilon (max perturbation)    : {epsilon}")
    print(f"  • Delta (perturbation size)      : {delta}")
    print(f"  • Learning rate                  : {learning_rate}")
    print(f"  • Number of optimization steps   : {num_steps}")
    print(f"  • Batch size for gradient estimate: {batch_size}")
    print(f"  • Max samples to attack          : {suggested_max_samples}")

    # === Allow override by user ===
    if not questionary.confirm("Do you want to accept these suggested values?").ask():
        epsilon = float(questionary.text("Epsilon (e.g., 0.05):", default=str(epsilon)).ask())
        delta = float(questionary.text("Delta (perturbation size for SPSA, e.g., 0.01):", default=str(delta)).ask())
        learning_rate = float(questionary.text("Learning rate (e.g., 0.01):", default=str(learning_rate)).ask())
        num_steps = int(questionary.text("Number of steps (e.g., 100):", default=str(num_steps)).ask())
        batch_size = int(questionary.text("Batch size (e.g., 32):", default=str(batch_size)).ask())
        suggested_max_samples = int(questionary.text("Max samples to attack (0 = all):", default=str(suggested_max_samples)).ask())

    return {
        "epsilon": epsilon,
        "delta": delta,
        "learning_rate": learning_rate,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "max_samples": suggested_max_samples
    }

def configure_transfer_attack(profile_data):
    print("\n=== Configuring Transfer-Based Attack ===\n")

    # === Read target model information ===
    model_info = profile_data.get("model", {})
    model_name = model_info.get("name", "")
    model_type = model_info.get("type", "builtin")
    model_params = model_info.get("params", {})

    substitute_suggestion = {}
    warning = None

    # === Suggest substitute based on target model ===
    if model_type == "custom" or "user_model" in model_name:
        warning = "[!] Cannot infer smaller model automatically from user_model.py. Suggesting simple CNN substitute."
        substitute_suggestion = {
            "name": "cnn",
            "params": {
                "conv_filters": 16,
                "hidden_size": 64
            }
        }
    elif model_name == "cnn":
        original_filters = model_params.get("conv_filters", 32)
        original_hidden = model_params.get("hidden_size", 128)
        substitute_suggestion = {
            "name": "cnn",
            "params": {
                "conv_filters": max(8, original_filters // 2),
                "hidden_size": max(32, original_hidden // 2)
            }
        }
    elif model_name in ["resnet18", "resnet50"]:
        substitute_suggestion = {
            "name": "cnn",
            "params": {
                "conv_filters": 16,
                "hidden_size": 64
            }
        }
    elif model_name == "vit":
        substitute_suggestion = {
            "name": "mlp",
            "params": {
                "hidden_size": 64,
                "input_size": 784
            }
        }
    elif model_name == "mlp":
        original_hidden = model_params.get("hidden_size", 128)
        substitute_suggestion = {
            "name": "mlp",
            "params": {
                "hidden_size": max(32, original_hidden // 2),
                "input_size": model_params.get("input_size", 784)
            }
        }
    else:
        warning = "[!] Unknown model type. Using simple CNN substitute."
        substitute_suggestion = {
            "name": "cnn",
            "params": {
                "conv_filters": 16,
                "hidden_size": 64
            }
        }

    # === Suggest attack method ===
    suggested_attack = "fgsm"

    # === Show full suggestion ===
    print("\n[*] Suggested Substitute Model:")
    print(f"  • Model: {substitute_suggestion['name']}")
    for k, v in substitute_suggestion.get("params", {}).items():
        print(f"    - {k}: {v}")

    print(f"\n[*] Suggested attack to generate adversarial examples: {suggested_attack.upper()}")

    if warning:
        print(f"\n{warning}")

    # === Accept suggestion or not ===
    if questionary.confirm("\nDo you want to accept this suggested configuration?").ask():
        substitute = substitute_suggestion
        attack_method = suggested_attack
    else:
        # Manual selection
        print("\n[*] Manual Substitute Model Selection")
        model_choices = ["cnn", "mlp", "resnet18", "user_model.py"]
        selected = questionary.select("Select substitute model:", choices=model_choices).ask()

        substitute = {"name": selected, "params": {}}

        if selected == "cnn":
            conv_filters = int(questionary.text("Number of conv filters (e.g., 16):", default="16").ask())
            hidden_size = int(questionary.text("Hidden layer size (e.g., 64):", default="64").ask())
            substitute["params"] = {
                "conv_filters": conv_filters,
                "hidden_size": hidden_size
            }
        elif selected == "mlp":
            hidden_size = int(questionary.text("Hidden size (e.g., 64):", default="64").ask())
            input_size = int(questionary.text("Input size (e.g., 784):", default="784").ask())
            substitute["params"] = {
                "hidden_size": hidden_size,
                "input_size": input_size
            }
        elif selected == "resnet18":
            substitute["params"] = {}
        elif selected == "user_model.py":
            print("[!] Reminder: You must ensure substitute is compatible manually.")

        # Choose attack manually too
        attack_method = questionary.select(
            "Select attack to use for generating adversarial examples:",
            choices=["fgsm", "pgd"]
        ).ask()

    # === Return final configuration ===
    return {
        "substitute_model": substitute,
        "attack_method": attack_method
    }


def run_setup():
    print("\n=== Safe-DL Framework — Module 2 Setup Wizard ===\n")

    dataset_info, num_classes, input_shape, class_names = select_dataset()
    model_info = select_model(num_classes, input_shape)
    profile_path = select_profile()

    if profile_path is None:
        print("[!] No profile selected. Exiting.")
        return

    with open(profile_path, "r") as f:
        profile_data = yaml.safe_load(f)

    profile_data["dataset"] = dataset_info
    profile_data["model"] = model_info

    threat_categories = profile_data.get("threat_model", {}).get("threat_categories", [])

    if "data_poisoning" in threat_categories:
        print("[*] Data poisoning enabled in threat model.")

        selected_attacks = questionary.checkbox(
            "Select the data poisoning attacks to simulate:",
            choices=[
                Choice("Label Flipping", value="label_flipping"),
                Choice("Clean Label ", value="clean_label"),
            ]
        ).ask()

        data_poisoning_cfg = {}

        if "label_flipping" in selected_attacks:
            data_poisoning_cfg["label_flipping"] = configure_label_flipping(profile_data,class_names)
        if "clean_label" in selected_attacks:
            data_poisoning_cfg["clean_label"] = configure_clean_label(profile_data, class_names)

        # if none of the attacks are selected
        if not data_poisoning_cfg:
            print("[!] No data poisoning attacks selected.")
        else:
            print(f"[*] Data poisoning configuration: {data_poisoning_cfg}")
            profile_data["attack_overrides"] = profile_data.get("attack_overrides", {})
            profile_data["attack_overrides"]["data_poisoning"] = data_poisoning_cfg

    if "backdoor_attacks" in threat_categories:
        print("[*] Backdoor attacks enabled in threat model.")

        selected_backdoors = questionary.checkbox(
            "Select the backdoor attacks to simulate:",
            choices=[
                Choice("Static Patch Trigger", value="static_patch"),
                Choice("Adversarially Learned Trigger", value="learned")
            ]
        ).ask()

        backdoor_cfg = {}

        if "static_patch" in selected_backdoors:
            backdoor_cfg["static_patch"] = configure_static_patch(profile_data, class_names)

        if "learned" in selected_backdoors:
            backdoor_cfg["learned"] = configure_learned_trigger(profile_data, class_names)

        if not backdoor_cfg:
            print("[!] No backdoor attacks selected.")
        else:
            print(f"[*] Backdoor configuration: {backdoor_cfg}")
            profile_data["attack_overrides"] = profile_data.get("attack_overrides", {})
            profile_data["attack_overrides"]["backdoor"] = backdoor_cfg

    if "evasion_attacks" in threat_categories:
        print("[*] Evasion attacks enabled in threat model.")

        selected_evasions = questionary.checkbox(
            "Select the evasion attacks to simulate:",
            choices=[
                Choice("Fast Gradient Sign Method (FGSM)", value="fgsm"),
                Choice("Projected Gradient Descent (PGD)", value="pgd"),
                Choice("Carlini & Wagner (C&W)", value="cw"),
                Choice("DeepFool", value="deepfool"),
                Choice("Natural Evolution Strategies (NES)", value="nes"),
                Choice("Simultaneous Perturbation Stochastic Approximation (SPSA)", value="spsa"),
                Choice("Transfer-based Attack", value="transfer")
            ]
        ).ask()

        evasion_cfg = {}
        if "fgsm" in selected_evasions:
            evasion_cfg["fgsm"] = configure_fgsm(profile_data)

        if "pgd" in selected_evasions:
            evasion_cfg["pgd"] = configure_pgd(profile_data)

        if "cw" in selected_evasions:
            evasion_cfg["cw"] = configure_cw(profile_data)

        if "deepfool" in selected_evasions:
            evasion_cfg["deepfool"] = configure_deepfool(profile_data)

        if "nes" in selected_evasions:
            evasion_cfg["nes"] = configure_nes(profile_data)

        if "spsa" in selected_evasions:
            evasion_cfg["spsa"] = configure_spsa(profile_data)

        if "transfer" in selected_evasions:
            evasion_cfg["transfer"] = configure_transfer_attack(profile_data)

        if evasion_cfg:
            profile_data["attack_overrides"] = profile_data.get("attack_overrides", {})
            profile_data["attack_overrides"]["evasion_attacks"] = evasion_cfg


    with open(profile_path, "w") as f:
        yaml.dump(profile_data, f)

    print(f"\n[✔] Profile updated and saved at: {profile_path}")


if __name__ == "__main__":
    run_setup()
