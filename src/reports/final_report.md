# Safe-DL Framework - Final Security Report
**Profile Selected**: `test.yaml`
**Report Generated On**: 2025-06-10 21:00:16

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

## 4. Attack Simulation (Module 2)
This section summarizes the outcomes of the adversarial attack simulations performed against the model based on the defined threat profile. These simulations quantify the model's vulnerability to various attack types before any defenses are applied.

### 4.1 Overview of Simulated Attacks

| Attack Category | Attack Method | Clean Acc. (Pre-Attack) | Impact on Clean Acc. | Key Parameters | Full Results |
|:----------------|:--------------|:------------------------|:---------------------|:---------------|:-------------|
| Data Poisoning | Clean Label | 67.54% | 62.76% | Poison Fraction: 0.05, Target Class: 5 | [Details](../module2_attack_simulation/results/data_poisoning/clean_label/clean_label_metrics.json) |
| Data Poisoning | Label Flipping | 67.54% | 54.88% | Flip Rate: 0.08, Target Class: 1 | [Details](../module2_attack_simulation/results/data_poisoning/label_flipping/label_flipping_metrics.json) |

**Note**: 'Clean Acc. (Pre-Attack)' represents the model's accuracy on clean data before any attack preparations. 'Impact on Clean Acc.' shows the model's accuracy on clean data *after* being trained with poisoned data, reflecting the attack's effectiveness in degrading performance.


