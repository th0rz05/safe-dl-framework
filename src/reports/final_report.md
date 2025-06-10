# Safe-DL Framework - Final Security Report
**Profile Selected**: `test.yaml`
**Report Generated On**: 2025-06-10 22:33:42

---

## 1. Introduction and Overview
This comprehensive report aggregates the findings from the Safe-DL framework's security assessment, covering threat modeling, attack simulation, risk analysis, defense application, and defense evaluation. Its purpose is to provide a unified overview of the deep learning model's security posture against adversarial threats, document the mitigation strategies applied, and quantify their effectiveness. This dossier serves as a critical resource for decision-making regarding model deployment and continuous security improvement.

## 2. System Under Evaluation Details
This section details the core components of the system analyzed in this report, as defined in the selected threat profile.

### 2.1 Model Details
- **Name**: `cnn`
- **Type**: `builtin`
- **Input Shape**: `[3, 32, 32]`
- **Number of Classes**: `10`
- **Parameters**: `{'conv_filters': 32, 'hidden_size': 128}`


### 2.2 Dataset Details
- **Name**: `cifar10`
- **Type**: `builtin`

## 3. Threat Profile (Module 1)
This section outlines the specific characteristics of the system's environment and the anticipated adversary, as defined during the threat modeling phase (Module 1). These parameters guide the subsequent attack simulations, risk analysis, and defense applications.

- **Model Access**: `white-box`
  *Describes the level of access the adversary has to the model internals (e.g., weights, architecture).*

- **Attack Goal**: `targeted`
  *Defines the adversary's objective (e.g., targeted misclassification, untargeted denial of service).*

- **Deployment Scenario**: `cloud`
  *Indicates where the model is deployed (e.g., cloud, edge device, mobile).*

- **Interface Exposed**: `api`
  *How the model interacts with external entities (e.g., API, direct access, web application).*

- **Model Type**: `cnn`
  *The architectural type of the deep learning model.*

- **Data Sensitivity**: `high`
  *The sensitivity level of the data used by the model, impacting privacy concerns.*

- **Training Data Source**: `internal_clean`
  *Origin and cleanliness of the data used for training the model.*

- **Threat Categories**:
    - `data_poisoning`
    - `backdoor_attacks`
    - `evasion_attacks`
    - `model_stealing`
    - `membership_inference`
    - `model_inversion`

  *A list of attack types considered relevant for this threat profile.*
  
*Note: While listed in the threat profile, 'model_stealing', 'membership_inference', and 'model_inversion' attack simulations and their corresponding defenses are currently considered future work and are not yet fully implemented in subsequent Modules.*

## 4. Attack Simulation (Module 2)
This section summarizes the outcomes of the adversarial attack simulations performed against the model based on the defined threat profile. These simulations quantify the model's vulnerability to various attack types before any defenses are applied.

### 4.1 Overview of Simulated Attacks

| Attack Category | Attack Method | Clean Acc. (Pre-Attack) | Impact on Clean Acc. | Attack Metric | Key Parameters | Full Results |
|:----------------|:--------------|:------------------------|:---------------------|:--------------|:---------------|:-------------|
| Data Poisoning | Clean Label | 67.54% | 62.76% | 62.76% (Degraded Acc.) | Poison Fraction: 0.05, Target Class: 5 | [Details](../module2_attack_simulation/results/data_poisoning/clean_label/clean_label_report.md) |
| Data Poisoning | Label Flipping | 67.54% | 54.88% | 54.88% (Degraded Acc.) | Flip Rate: 0.08, Target Class: 1 | [Details](../module2_attack_simulation/results/data_poisoning/label_flipping/label_flipping_report.md) |
| Backdoor | Static Patch | 67.54% | 60.58% | 77.99% (ASR) | Poison Frac.: 0.05, Target Class: 7, Patch Type: white_square | [Details](../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md) |
| Evasion | Pgd | 67.54% | 67.54% | 0.00% (Adv. Acc.) | Epsilon: 0.03, Num Iter: 50 | [Details](../module2_attack_simulation/results/evasion/pgd/pgd_report.md) |
| Evasion | Spsa | 67.54% | 68.60% | 2.20% (Adv. Acc.) | Epsilon: 0.03, Num Steps: 150, Delta: 0.01 | [Details](../module2_attack_simulation/results/evasion/spsa/spsa_report.md) |

**Note**: 'Clean Acc. (Pre-Attack)' represents the model's accuracy on clean data before any attack preparations. 'Impact on Clean Acc.' shows the model's accuracy on clean data *after* being subjected to the attack (e.g., trained with poisoned data, or backdoor injected). For Data Poisoning attacks, 'Attack Metric' displays the degraded accuracy of the model on clean inputs after poisoning. For Backdoor attacks, 'Attack Metric' displays the Attack Success Rate (ASR), indicating the percentage of adversarial samples (with trigger) successfully misclassified to the target class. For Evasion attacks, 'Attack Metric' displays the Adversarial Accuracy (Adv. Acc.) on perturbed inputs, where a lower value indicates a more successful attack.



## 5. Risk Analysis (Module 3)

This section summarizes the risk assessment performed on the simulated attacks. Each attack is evaluated based on its severity, probability, and visibility. A final risk score is computed to help prioritize mitigation strategies, followed by specific defense recommendations.


### 5.1 Risk Summary Table

| Attack | Type | Severity | Probability | Visibility | Risk Score | Report |
|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| Pgd | Evasion | 1.00 | 1.00 | 0.30 | 1.70 | [Details](../module2_attack_simulation/results/evasion/pgd/pgd_report.md) |
| Spsa | Evasion | 1.00 | 0.80 | 0.20 | 1.44 | [Details](../module2_attack_simulation/results/evasion/spsa/spsa_report.md) |
| Static Patch | Backdoor | 1.00 | 1.00 | 0.60 | 1.40 | [Details](../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md) |
| Label Flipping | Data Poisoning | 0.42 | 1.00 | 0.70 | 0.55 | [Details](../module2_attack_simulation/results/data_poisoning/label_flipping/label_flipping_report.md) |
| Clean Label | Data Poisoning | 0.16 | 0.90 | 0.30 | 0.24 | [Details](../module2_attack_simulation/results/data_poisoning/clean_label/clean_label_report.md) |


### 5.2 Risk Matrix (Qualitative)

This matrix categorizes attacks based on their qualitative Severity and Probability levels.

| Severity \ Probability | Low | Medium | High |
|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| High | - | - | Pgd, Spsa, Static Patch |
| Medium | - | - | Label Flipping |
| Low | - | - | Clean Label |


### 5.3 Risk Ranking

Attacks ranked by their calculated Risk Score, from highest to lowest.

1. **Pgd** — Risk Score: 1.70 → [Details](../module2_attack_simulation/results/evasion/pgd/pgd_report.md)
2. **Spsa** — Risk Score: 1.44 → [Details](../module2_attack_simulation/results/evasion/spsa/spsa_report.md)
3. **Static Patch** — Risk Score: 1.40 → [Details](../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md)
4. **Label Flipping** — Risk Score: 0.55 → [Details](../module2_attack_simulation/results/data_poisoning/label_flipping/label_flipping_report.md)
5. **Clean Label** — Risk Score: 0.24 → [Details](../module2_attack_simulation/results/data_poisoning/clean_label/clean_label_report.md)



### 5.4 Defense Recommendations

Based on the identified risks and threat profile, the following defense recommendations are provided:

- **Clean Label**:
  - provenance_tracking
  - influence_functions
- **Label Flipping**:
  - data_cleaning
  - per_class_monitoring
- **Pgd**:
  - adversarial_training
  - randomized_smoothing
- **Spsa**:
  - gradient_masking
  - jpeg_preprocessing



### 5.5 Paths to Details

For more in-depth information about individual attacks, including raw metrics, attack visualizations, and specific parameters, please refer to the detailed reports linked in the 'Risk Summary Table' and 'Risk Ranking' sections above.
