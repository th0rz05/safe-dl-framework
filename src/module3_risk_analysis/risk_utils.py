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