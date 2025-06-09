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
