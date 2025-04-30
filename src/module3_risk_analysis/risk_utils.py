def analyze_label_flipping(data, baseline):
    acc_clean = baseline["overall_accuracy"]
    acc_poisoned = data["accuracy_after_attack"]
    acc_drop = acc_clean - acc_poisoned

    # Adjust severity: how much accuracy dropped (normalized by a threshold)
    severity = min(acc_drop / 0.3, 1.0)

    # Consider attack size
    flip_rate = data.get("flip_rate", None)
    if flip_rate is not None and flip_rate < 0.05:
        severity *= 0.8  # lower weight for very small attacks

    # Assume full model access for now
    probability = 1.0

    # Visibility proxy: more flipped samples = more visible
    num_flipped = data.get("num_flipped", 0)
    visibility = min(0.3 + (num_flipped / 10000), 1.0)  # cap visibility at 1.0

    risk_score = round(severity * probability * (1 + (1 - visibility)), 3)

    return {
        "type": "data_poisoning",
        "accuracy_clean": round(acc_clean, 4),
        "accuracy_attack": round(acc_poisoned, 4),
        "severity": round(severity, 3),
        "probability": probability,
        "visibility": round(visibility, 3),
        "risk_score": risk_score,
        "flip_rate": flip_rate,
        "num_flipped": num_flipped
    }

def analyze_clean_label(data, baseline):
    acc_clean = baseline["overall_accuracy"]
    acc_poisoned = data["accuracy_after_attack"]
    acc_drop = acc_clean - acc_poisoned

    severity = min(acc_drop / 0.3, 1.0)

    # Lower severity if poison fraction is small
    poison_fraction = data.get("fraction_poison", 1.0)
    if poison_fraction < 0.05:
        severity *= 0.8

    probability = 0.9  # more subtle to execute

    # Visibility: use average perturbation norm if available
    visibility = 0.3
    example_poisoned = data.get("example_poisoned_samples", [])
    if example_poisoned:
        norms = [ex.get("perturbation_norm", 0.0) for ex in example_poisoned]
        avg_norm = sum(norms) / len(norms) if norms else 0.0
        if avg_norm > 0.2:
            visibility = 0.6  # more visible poisoning
        elif avg_norm < 0.05:
            visibility = 0.2  # very stealthy

    risk_score = round(severity * probability * (1 + (1 - visibility)), 3)

    return {
        "type": "data_poisoning",
        "accuracy_clean": round(acc_clean, 4),
        "accuracy_attack": round(acc_poisoned, 4),
        "severity": round(severity, 3),
        "probability": probability,
        "visibility": round(visibility, 3),
        "risk_score": risk_score,
        "fraction_poison": poison_fraction,
        "avg_perturbation_norm": round(avg_norm, 4) if example_poisoned else None
    }

def analyze_static_patch(data):
    acc_clean = data["accuracy_clean_testset"]
    asr = data.get("attack_success_rate", 0.0)

    # Severity: ASR * drop in clean accuracy
    severity = min(asr + (1 - acc_clean), 1.0)

    # Probability: corrupted-label + white-box (default = 1.0)
    probability = 1.0

    # Visibility: higher patch size = easier to spot
    patch_ratio = data.get("patch_size_ratio", 0.15)
    visibility = min(0.3 + 2 * patch_ratio, 1.0)

    risk_score = round(severity * probability * (1 + (1 - visibility)), 3)

    return {
        "type": "backdoor",
        "accuracy_clean": round(acc_clean, 4),
        "asr": round(asr, 4),
        "severity": round(severity, 3),
        "probability": probability,
        "visibility": round(visibility, 3),
        "risk_score": risk_score,
        "patch_size_ratio": patch_ratio,
        "poison_fraction": data.get("poison_fraction"),
        "blend_alpha": data.get("blend_alpha")
    }


def analyze_learned_trigger(data):
    acc_clean = data["accuracy_clean_testset"]
    asr = data.get("attack_success_rate", 0.0)

    # Severity = ASR + degradation on clean test set
    severity = min(asr + (1 - acc_clean), 1.0)

    # Probability: learned trigger = a bit harder to execute
    probability = 0.9

    # Visibility: inferred from ASR + blending
    visibility = 0.2
    blend_alpha = data.get("blend_alpha", 1.0)
    if blend_alpha < 0.5:
        visibility = 0.4  # semi-stealth
    if blend_alpha < 0.2:
        visibility = 0.6  # more visible

    risk_score = round(severity * probability * (1 + (1 - visibility)), 3)

    return {
        "type": "backdoor",
        "accuracy_clean": round(acc_clean, 4),
        "asr": round(asr, 4),
        "severity": round(severity, 3),
        "probability": probability,
        "visibility": round(visibility, 3),
        "risk_score": risk_score,
        "poison_fraction": data.get("poison_fraction"),
        "blend_alpha": blend_alpha
    }

def analyze_fgsm(data, baseline):
    acc_clean = data["accuracy_clean_testset"]
    acc_adv = data["accuracy_adversarial_testset"]
    acc_drop = acc_clean - acc_adv

    # Severity: how much accuracy dropped, normalized
    severity = min(acc_drop / 0.3, 1.0)

    # Probability: high (white-box, simple gradient method)
    probability = 1.0

    # Visibility: use epsilon if present, fallback to perturbation norms
    epsilon = data.get("epsilon", None)
    visibility = 0.3  # default base

    if epsilon is not None:
        if epsilon >= 0.05:
            visibility = 0.6
        elif epsilon <= 0.01:
            visibility = 0.2
    else:
        norms = [ex.get("perturbation_norm", 0.0) for ex in data.get("example_adversarial_samples", [])]
        avg_norm = sum(norms) / len(norms) if norms else 0.0
        if avg_norm > 0.2:
            visibility = 0.6
        elif avg_norm < 0.05:
            visibility = 0.2

    # Final risk score
    risk_score = round(severity * probability * (1 + (1 - visibility)), 3)

    return {
        "type": "evasion",
        "attack_name": "fgsm",
        "accuracy_clean": round(acc_clean, 4),
        "accuracy_attack": round(acc_adv, 4),
        "severity": round(severity, 3),
        "probability": probability,
        "visibility": round(visibility, 3),
        "risk_score": risk_score,
        "epsilon": epsilon
    }

def analyze_pgd(data, baseline):
    acc_clean = data["accuracy_clean_testset"]
    acc_adv = data["accuracy_adversarial_testset"]
    acc_drop = acc_clean - acc_adv

    # Severity: drop in accuracy normalized
    severity = min(acc_drop / 0.3, 1.0)

    # Probability: high (white-box, iterative method)
    probability = 1.0

    # Visibility: depends on epsilon + number of steps
    epsilon = data.get("epsilon", None)
    steps = data.get("pgd_steps", 10)  # default fallback
    visibility = 0.3  # base

    if epsilon is not None:
        if epsilon >= 0.05:
            visibility = 0.5
        elif epsilon <= 0.01:
            visibility = 0.2

    if steps >= 20:
        visibility += 0.1
    visibility = min(visibility, 1.0)

    # Risk score calculation
    risk_score = round(severity * probability * (1 + (1 - visibility)), 3)

    return {
        "type": "evasion",
        "attack_name": "pgd",
        "accuracy_clean": round(acc_clean, 4),
        "accuracy_attack": round(acc_adv, 4),
        "severity": round(severity, 3),
        "probability": probability,
        "visibility": round(visibility, 3),
        "risk_score": risk_score,
        "epsilon": epsilon,
        "pgd_steps": steps
    }


def analyze_cw(data, baseline):
    acc_clean = data["accuracy_clean_testset"]
    acc_adv = data["accuracy_adversarial_testset"]
    acc_drop = acc_clean - acc_adv

    # Severity: as usual, normalized accuracy drop
    severity = min(acc_drop / 0.3, 1.0)

    # Probability: moderate — requires optimization and access
    probability = 0.9

    # Visibility: based on perturbation norm and confidence parameter
    visibility = 0.3
    norm_type = data.get("norm_type", "l2")
    confidence = data.get("confidence", 0.0)

    norms = [ex.get("perturbation_norm", 0.0) for ex in data.get("example_adversarial_samples", [])]
    avg_norm = sum(norms) / len(norms) if norms else 0.0

    if avg_norm > 0.2 or confidence > 10:
        visibility = 0.6
    elif avg_norm < 0.05 and confidence <= 0:
        visibility = 0.2

    # Final score
    risk_score = round(severity * probability * (1 + (1 - visibility)), 3)

    return {
        "type": "evasion",
        "attack_name": "cw",
        "accuracy_clean": round(acc_clean, 4),
        "accuracy_attack": round(acc_adv, 4),
        "severity": round(severity, 3),
        "probability": probability,
        "visibility": round(visibility, 3),
        "risk_score": risk_score,
        "norm_type": norm_type,
        "confidence": confidence,
        "avg_perturbation_norm": round(avg_norm, 4) if norms else None
    }


def analyze_deepfool(data, baseline):
    acc_clean = data["accuracy_clean_testset"]
    acc_adv = data["accuracy_adversarial_testset"]
    acc_drop = acc_clean - acc_adv

    # Severity: normalized accuracy drop
    severity = min(acc_drop / 0.3, 1.0)

    # Probability: high — white-box, fast linearized approximation
    probability = 1.0

    # Visibility: DeepFool é projetado para ser minimal — usa perturbação média
    norms = [ex.get("perturbation_norm", 0.0) for ex in data.get("example_adversarial_samples", [])]
    avg_norm = sum(norms) / len(norms) if norms else 0.0

    if avg_norm < 0.02:
        visibility = 0.1  # quase impercetível
    elif avg_norm < 0.05:
        visibility = 0.2
    else:
        visibility = 0.4

    # Final risk score
    risk_score = round(severity * probability * (1 + (1 - visibility)), 3)

    return {
        "type": "evasion",
        "attack_name": "deepfool",
        "accuracy_clean": round(acc_clean, 4),
        "accuracy_attack": round(acc_adv, 4),
        "severity": round(severity, 3),
        "probability": probability,
        "visibility": round(visibility, 3),
        "risk_score": risk_score,
        "avg_perturbation_norm": round(avg_norm, 4) if norms else None
    }

def analyze_nes(data, baseline):
    acc_clean = data["accuracy_clean_testset"]
    acc_adv = data["accuracy_adversarial_testset"]
    acc_drop = acc_clean - acc_adv

    # Severity: normalized drop
    severity = min(acc_drop / 0.3, 1.0)

    # Probability: lower than white-box due to query complexity
    probability = 0.8

    # Visibility: assume higher due to approximate gradients — use perturbation norm
    norms = [ex.get("perturbation_norm", 0.0) for ex in data.get("example_adversarial_samples", [])]
    avg_norm = sum(norms) / len(norms) if norms else 0.0

    if avg_norm >= 0.2:
        visibility = 0.6
    elif avg_norm <= 0.05:
        visibility = 0.2
    else:
        visibility = 0.4

    # Final score
    risk_score = round(severity * probability * (1 + (1 - visibility)), 3)

    return {
        "type": "evasion",
        "attack_name": "nes",
        "accuracy_clean": round(acc_clean, 4),
        "accuracy_attack": round(acc_adv, 4),
        "severity": round(severity, 3),
        "probability": probability,
        "visibility": round(visibility, 3),
        "risk_score": risk_score,
        "avg_perturbation_norm": round(avg_norm, 4) if norms else None
    }


def analyze_spsa(data, baseline):
    acc_clean = data["accuracy_clean_testset"]
    acc_adv = data["accuracy_adversarial_testset"]
    acc_drop = acc_clean - acc_adv

    # Severity: normalized drop
    severity = min(acc_drop / 0.3, 1.0)

    # Probability: moderate — black-box and requires many queries
    probability = 0.8

    # Visibility: based on average perturbation norm
    norms = [ex.get("perturbation_norm", 0.0) for ex in data.get("example_adversarial_samples", [])]
    avg_norm = sum(norms) / len(norms) if norms else 0.0

    if avg_norm >= 0.2:
        visibility = 0.6
    elif avg_norm <= 0.05:
        visibility = 0.2
    else:
        visibility = 0.4

    # Final score
    risk_score = round(severity * probability * (1 + (1 - visibility)), 3)

    return {
        "type": "evasion",
        "attack_name": "spsa",
        "accuracy_clean": round(acc_clean, 4),
        "accuracy_attack": round(acc_adv, 4),
        "severity": round(severity, 3),
        "probability": probability,
        "visibility": round(visibility, 3),
        "risk_score": risk_score,
        "avg_perturbation_norm": round(avg_norm, 4) if norms else None
    }


def analyze_transfer(data, baseline):
    acc_clean = data["accuracy_clean_testset"]
    acc_adv = data["accuracy_adversarial_testset"]
    acc_drop = acc_clean - acc_adv

    # Severity: normalized drop in performance
    severity = min(acc_drop / 0.3, 1.0)

    # Probability: moderate-high — requires access to surrogate model
    probability = 0.85

    # Visibility: assume perturbation was tuned on other model, use norm
    norms = [ex.get("perturbation_norm", 0.0) for ex in data.get("example_adversarial_samples", [])]
    avg_norm = sum(norms) / len(norms) if norms else 0.0

    if avg_norm >= 0.2:
        visibility = 0.6
    elif avg_norm <= 0.05:
        visibility = 0.2
    else:
        visibility = 0.4

    # Final score
    risk_score = round(severity * probability * (1 + (1 - visibility)), 3)

    return {
        "type": "evasion",
        "attack_name": "transfer",
        "accuracy_clean": round(acc_clean, 4),
        "accuracy_attack": round(acc_adv, 4),
        "severity": round(severity, 3),
        "probability": probability,
        "visibility": round(visibility, 3),
        "risk_score": risk_score,
        "avg_perturbation_norm": round(avg_norm, 4) if norms else None
    }


