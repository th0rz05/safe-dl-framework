def compute_mitigation_score(acc_baseline, acc_attack, acc_defense):
    """
    Computes how effectively the defense recovers the performance drop caused by the attack.

    Args:
        acc_baseline (float): Accuracy on clean data before the attack.
        acc_attack (float): Accuracy after the attack (no defense applied).
        acc_defense (float): Accuracy after the defense has been applied.

    Returns:
        float: Mitigation score (can be < 0 if defense made it worse).
    """
    if acc_baseline == acc_attack:
        return 0.0  # No degradation to mitigate
    return (acc_defense - acc_attack) / (acc_baseline - acc_attack)


def compute_cad_score(acc_baseline, acc_defense, max_allowed_drop=0.1):
    """
    Computes a normalized score for clean accuracy drop caused by the defense.

    Args:
        acc_baseline (float): Accuracy on clean data before any attack or defense.
        acc_defense (float): Accuracy on clean data after defense is applied.
        max_allowed_drop (float): Maximum acceptable drop in accuracy (default: 10%).

    Returns:
        float: Clean Accuracy Drop score in [0, 1], where 1 means no drop.
    """
    drop = acc_baseline - acc_defense
    return max(0.0, 1.0 - drop / max_allowed_drop)

def estimate_defense_cost(defense_name):
    """
    Estimates the computational or implementation cost of a defense using a fixed internal mapping.

    Args:
        defense_name (str): The name of the defense technique.

    Returns:
        float: Estimated cost in [0, 1], where higher means more computationally intensive.
    """
    COST_MAP = {
        "data_cleaning": 0.2,
        "per_class_monitoring": 0.2,
        "dp_training": 0.7,
        "robust_loss": 0.5,
        "influence_function": 0.6,
        "activation_clustering": 0.3,
        "spectral_signature": 0.4,
        "anomaly_detection": 0.3,
        "pruning": 0.3,
        "fine_pruning": 0.4,
        "model_inspection": 0.2,
        "adversarial_training": 0.8,
        "jpeg_preprocessing": 0.1,
        "randomized_smoothing": 0.5,
        "certified_defense": 0.9,
        "gradient_masking": 0.4,
    }
    return COST_MAP.get(defense_name, 0.5)

def compute_defense_score(ms, cad_score, pcr=1.0, cs=1.0, dcs=0.0):
    """
    Combines all individual scores into a final defense score.

    Args:
        ms (float): Mitigation Score (effectiveness in restoring accuracy).
        cad_score (float): Clean Accuracy Drop score.
        pcr (float, optional): Per-Class Recovery score (default: 1.0 if not available).
        cs (float, optional): Coverage Score — how broadly the defense was applied (default: 1.0).
        dcs (float, optional): Defense Cost Score — computational overhead estimate (default: 0.0).

    Returns:
        float: Final defense score, normalized such that higher means more effective and efficient.
    """
    return (ms * cad_score) * (0.5 + 0.5 * pcr) * (0.5 + 0.5 * cs) / (1.0 + dcs)

