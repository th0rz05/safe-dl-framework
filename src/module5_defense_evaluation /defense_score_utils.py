from defense_utils import (
    compute_mitigation_score,
    compute_cad_score,
    estimate_defense_cost,
    compute_defense_score
)

def evaluate_activation_clustering(defense_data, attack_data):
    """
    Evaluates the effectiveness of the 'activation_clustering' defense
    using standard metrics and scoring formulas.

    Args:
        defense_data (dict): JSON-loaded results from the defense.
        attack_data (dict): JSON-loaded results from the attack (without defense).

    Returns:
        dict: Dictionary containing mitigation score, CAD score, cost score, and final score.
    """
    acc_clean_base = attack_data["accuracy_clean_testset"]
    acc_attack = attack_data["attack_success_rate"]
    acc_clean_def = defense_data["accuracy_clean"]
    acc_adv_def = defense_data["accuracy_adversarial"]

    # Invert ASR to get adversarial accuracy during attack
    acc_adv_base = 1.0 - acc_attack

    # --- Scores ---
    mitigation = compute_mitigation_score(acc_clean_base, acc_adv_base, acc_adv_def)
    cad = compute_cad_score(acc_clean_base, acc_clean_def)
    cost = estimate_defense_cost("activation_clustering")

    final = compute_defense_score(mitigation, cad, dcs=cost)

    return {
        "mitigation_score": round(mitigation, 3),
        "cad_score": round(cad, 3),
        "defense_cost_score": round(cost, 3),
        "final_score": round(final, 3)
    }


def evaluate_spectral_signatures(defense_data: dict, attack_data: dict) -> dict:
    """
    Evaluates the performance of the spectral signature defense against a backdoor attack.

    Args:
        defense_data (dict): The results.json of the defense (e.g., spectral_signature).
        attack_data (dict): The attack_metrics.json of the original backdoor attack.

    Returns:
        dict: Dictionary with all evaluation metrics and final score.
    """
    # Extract relevant fields
    acc_baseline = attack_data.get("accuracy_clean_testset", 0.0)
    acc_attack = attack_data.get("attack_success_rate", 0.0)
    acc_defense_adv = defense_data.get("accuracy_adversarial", 0.0)
    acc_defense_clean = defense_data.get("accuracy_clean", 0.0)

    # Compute scores
    mitigation_score = compute_mitigation_score(acc_baseline, acc_attack, acc_defense_adv)
    cad_score = compute_cad_score(acc_baseline, acc_defense_clean)
    defense_cost_score = estimate_defense_cost("spectral_signature")

    # Final score
    final_score = compute_defense_score(
        mitigation_score, cad_score,
        pcr=1.0, cs=1.0, dcs=defense_cost_score
    )

    return {
        "mitigation_score": round(mitigation_score, 3),
        "cad_score": round(cad_score, 3),
        "defense_cost_score": defense_cost_score,
        "final_score": round(final_score, 3)
    }


def evaluate_anomaly_detection(defense_data: dict, attack_data: dict) -> dict:
    """
    Evaluates the performance of the anomaly detection defense against a backdoor attack.

    Args:
        defense_data (dict): The results.json from the anomaly detection defense.
        attack_data (dict): The original attack's attack_metrics.json.

    Returns:
        dict: Dictionary with mitigation score, CAD score, cost, and final score.
    """
    # Extract accuracy values
    acc_baseline = attack_data.get("accuracy_clean_testset", 0.0)
    acc_attack = attack_data.get("attack_success_rate", 0.0)
    acc_defense_adv = defense_data.get("accuracy_adversarial", 0.0)
    acc_defense_clean = defense_data.get("accuracy_clean", 0.0)

    # Compute all scores
    mitigation_score = compute_mitigation_score(acc_baseline, acc_attack, acc_defense_adv)
    cad_score = compute_cad_score(acc_baseline, acc_defense_clean)
    defense_cost_score = estimate_defense_cost("anomaly_detection")

    final_score = compute_defense_score(
        mitigation_score, cad_score,
        pcr=1.0, cs=1.0, dcs=defense_cost_score
    )

    return {
        "mitigation_score": round(mitigation_score, 3),
        "cad_score": round(cad_score, 3),
        "defense_cost_score": defense_cost_score,
        "final_score": round(final_score, 3)
    }


def evaluate_pruning(defense_data: dict, attack_data: dict) -> dict:
    """
    Evaluates the performance of the pruning defense against a backdoor attack.

    Args:
        defense_data (dict): The results.json from the pruning defense.
        attack_data (dict): The original attack's attack_metrics.json.

    Returns:
        dict: Dictionary with mitigation score, CAD score, cost, and final score.
    """
    # Extract accuracy values
    acc_baseline = attack_data.get("accuracy_clean_testset", 0.0)
    acc_attack = attack_data.get("attack_success_rate", 0.0)
    acc_defense_adv = defense_data.get("accuracy_adversarial", 0.0)
    acc_defense_clean = defense_data.get("accuracy_clean", 0.0)

    # Compute all scores
    mitigation_score = compute_mitigation_score(acc_baseline, acc_attack, acc_defense_adv)
    cad_score = compute_cad_score(acc_baseline, acc_defense_clean)
    defense_cost_score = estimate_defense_cost("pruning")

    final_score = compute_defense_score(
        mitigation_score, cad_score,
        pcr=1.0, cs=1.0, dcs=defense_cost_score
    )

    return {
        "mitigation_score": round(mitigation_score, 3),
        "cad_score": round(cad_score, 3),
        "defense_cost_score": defense_cost_score,
        "final_score": round(final_score, 3)
    }

def evaluate_fine_pruning(defense_data: dict, attack_data: dict) -> dict:
    acc_baseline = attack_data.get("accuracy_clean_testset", 0.0)
    acc_attack = attack_data.get("attack_success_rate", 0.0)
    acc_defense_adv = defense_data.get("accuracy_adversarial", 0.0)
    acc_defense_clean = defense_data.get("accuracy_clean", 0.0)

    mitigation_score = compute_mitigation_score(acc_baseline, acc_attack, acc_defense_adv)
    cad_score = compute_cad_score(acc_baseline, acc_defense_clean)
    defense_cost_score = estimate_defense_cost("fine_pruning")

    final_score = compute_defense_score(
        mitigation_score, cad_score,
        pcr=1.0, cs=1.0, dcs=defense_cost_score
    )

    return {
        "mitigation_score": round(mitigation_score, 3),
        "cad_score": round(cad_score, 3),
        "defense_cost_score": defense_cost_score,
        "final_score": round(final_score, 3)
    }


def evaluate_model_inspection(defense_data: dict, attack_data: dict) -> dict:
    acc_baseline = attack_data.get("accuracy_clean_testset", 0.0)
    acc_attack = attack_data.get("attack_success_rate", 0.0)
    acc_defense_adv = defense_data.get("accuracy_adversarial", 0.0)
    acc_defense_clean = defense_data.get("accuracy_clean", 0.0)

    mitigation_score = compute_mitigation_score(acc_baseline, acc_attack, acc_defense_adv)
    cad_score = compute_cad_score(acc_baseline, acc_defense_clean)
    defense_cost_score = estimate_defense_cost("model_inspection")

    final_score = compute_defense_score(
        mitigation_score, cad_score,
        pcr=1.0, cs=1.0, dcs=defense_cost_score
    )

    return {
        "mitigation_score": round(mitigation_score, 3),
        "cad_score": round(cad_score, 3),
        "defense_cost_score": defense_cost_score,
        "final_score": round(final_score, 3)
    }

