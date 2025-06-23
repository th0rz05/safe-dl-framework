def compute_mitigation_score(acc_baseline, acc_attack, acc_defense):
    """
    Computes how effectively the defense recovers the performance drop caused by the attack.
    """
    if abs(acc_baseline - acc_attack) < 1e-5:
        return 0.0
    return (acc_defense - acc_attack) / (acc_baseline - acc_attack)

def compute_mitigation_score_evasion(acc_attack, acc_defense_adv, max_possible=1.0):
    """
    Computes the mitigation score specifically for evasion defenses,
    based on the improvement in adversarial accuracy.

    Args:
        acc_attack (float): Accuracy on adversarial samples before defense.
        acc_defense_adv (float): Accuracy on adversarial samples after defense.
        max_possible (float): Max possible accuracy (usually 1.0)

    Returns:
        float: Score in [0, 1], measuring how much adversarial accuracy improved.
    """
    improvement = acc_defense_adv - acc_attack
    range_possible = max_possible - acc_attack
    if range_possible == 0:
        return 0.0
    return max(0.0, improvement / range_possible)


def compute_asr_mitigation_score(asr_before, asr_after):
    """
    Computes how much the defense reduced the attack success rate (ASR).

    Args:
        asr_before (float): ASR before applying the defense (higher = worse).
        asr_after (float): ASR after applying the defense (lower = better).

    Returns:
        float: Mitigation score in [0, 1], where 1.0 means complete ASR elimination.
               Can be < 0 if defense increased the ASR.
    """
    if abs(asr_before) < 1e-5:
        return 0.0  # No ASR to mitigate
    return max(0.0, (asr_before - asr_after) / asr_before)


def compute_cad_score(acc_clean_attack, acc_clean_defense):
    """
    Computes the Clean Accuracy Drop score as a percentage of retained performance,
    using accuracy after the attack as a baseline instead of a fixed threshold.

    Returns:
        float in [0, 1], where 1 means no drop compared to clean-after-attack.
    """
    if acc_clean_attack == 0:
        return 0.0  # avoid division by zero
    drop = acc_clean_attack - acc_clean_defense
    return max(0.0, 1.0 - drop / acc_clean_attack)



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

def compute_defense_score(ms, cad_score, dcs=0.0):
    """
    Final defense score based only on:
    - Mitigation (ms)
    - Clean Accuracy Drop score (cad_score)
    - Defense Cost (dcs)
    """
    W_MS = 0.8   # Prioridade principal: recuperar performance
    W_CAD = 0.2  # Penalizar quedas na clean
    W_COST = 0.1 # Penalizar custos de defesa

    effectiveness = W_MS * ms + W_CAD * cad_score
    penalty = 1 + W_COST * dcs

    return round(effectiveness / penalty, 3)