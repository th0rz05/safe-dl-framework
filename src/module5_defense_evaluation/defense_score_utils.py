from defense_utils import (
    compute_mitigation_score,
    compute_cad_score,
    estimate_defense_cost,
    compute_defense_score,
    compute_asr_mitigation_score,
    compute_mitigation_score_evasion
)

def evaluate_backdoor_defense(defense_name: str, defense_data: dict, attack_data: dict) -> dict:
    """
    Evaluates the effectiveness of a backdoor defense using standard metrics and scoring formulas.
    This is a generic function that can be used for any backdoor defense.

    Args:
        defense_name (str): The name of the defense (e.g., "activation_clustering", "pruning").
                            Used to estimate the defense cost.
        defense_data (dict): JSON-loaded results from the defense.
        attack_data (dict): JSON-loaded results from the attack (without defense).

    Returns:
        dict: Dictionary containing mitigation score, CAD score, cost score, and final score.
    """
    acc_clean_base = attack_data["accuracy_clean_testset"]
    acc_clean_def = defense_data["accuracy_clean"]

    asr_before = attack_data["attack_success_rate"]
    asr_after = defense_data["asr_after_defense"]

    mitigation = compute_asr_mitigation_score(asr_before, asr_after)
    cad = compute_cad_score(acc_clean_base, acc_clean_def)
    cost = estimate_defense_cost(defense_name) # Use the passed defense_name to get the cost

    final = compute_defense_score(mitigation, cad, dcs=cost)

    return {
        "mitigation_score": round(mitigation, 3),
        "cad_score": round(cad, 3),
        "defense_cost_score": round(cost, 3),
        "final_score": round(final, 3)
    }


def evaluate_data_poisoning_defense(defense_name: str, defense_data: dict, attack_data: dict, baseline_data: dict) -> dict:
    """
    Evaluates the effectiveness of a data poisoning defense using standard metrics and scoring formulas.
    This is a generic function that can be used for any data poisoning defense.

    Args:
        defense_name (str): The name of the defense (e.g., "data_cleaning", "robust_loss").
                            Used to estimate the defense cost.
        defense_data (dict): JSON-loaded results from the defense.
        attack_data (dict): JSON-loaded results from the attack (without defense).
        baseline_data (dict): JSON with 'overall_accuracy' from the clean baseline model.

    Returns:
        dict: Dictionary containing mitigation score, CAD score, cost score, and final score.
    """
    acc_baseline = baseline_data["overall_accuracy"]
    acc_attack = attack_data["accuracy_after_attack"] # Accuracy after poisoning attack (lower is worse)
    acc_defense = defense_data["accuracy_clean"]      # Accuracy after defense (hopefully higher)

    mitigation = compute_mitigation_score(acc_baseline, acc_attack, acc_defense)
    cad_score = compute_cad_score(acc_attack, acc_defense)
    cost = estimate_defense_cost(defense_name) # Use the passed defense_name to get the cost

    final = compute_defense_score(mitigation, cad_score, dcs=cost)


    return {
        "mitigation_score": round(mitigation, 3),
        "cad_score": round(cad_score, 3),
        "defense_cost_score": round(cost, 3),
        "final_score": round(final, 3)
    }


def evaluate_evasion_defense(defense_name: str, defense_data: dict, attack_data: dict, baseline_data: dict) -> dict:
    acc_clean_attack = baseline_data["overall_accuracy"]
    acc_attack = attack_data["accuracy_adversarial_testset"]
    acc_defense_clean = defense_data["accuracy_clean"]
    acc_defense_adv = defense_data["accuracy_adversarial"]

    mitigation = compute_mitigation_score_evasion(acc_attack, acc_defense_adv)
    cad = compute_cad_score(acc_clean_attack, acc_defense_clean)
    cost = estimate_defense_cost(defense_name)

    final = compute_defense_score(mitigation, cad, dcs=cost)

    return {
        "mitigation_score": round(mitigation, 3),
        "cad_score": round(cad, 3),
        "defense_cost_score": round(cost, 3),
        "final_score": round(final, 3)
    }