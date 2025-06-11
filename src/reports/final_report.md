# Safe-DL Framework - Final Security Report
**Profile Selected**: `test.yaml`
**Report Generated On**: 2025-06-11 15:55:10

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

---
## 6. Defense Application (Module 4)
This section details the performance of the implemented defenses against the simulated attacks identified in the risk analysis. For each attack, the table shows the model's accuracy on clean data *before* and *after* defense, and its accuracy on adversarial data *after* defense. Key defense parameters are also provided, along with a link to a detailed report.

| Attack Category   | Attack Method   | Defense Applied       | Clean Acc. (Pre-Defense)   | Clean Acc. (Post-Defense)   | Adv. Acc. (Post-Defense)   | Key Parameters                                                       | Link to Details                                                                                                |
|:------------------|:----------------|:----------------------|:---------------------------|:----------------------------|:---------------------------|:---------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|
| Backdoor          | Static Patch    | Activation Clustering | 60.58%                     | 43.56%                      | 12.32%                     | num_clusters: 2                                                      | [Details](../module4_defense_application/results/backdoor/static_patch/activation_clustering_report.md)        |
| Backdoor          | Static Patch    | Spectral Signatures   | 60.58%                     | 55.65%                      | 17.04%                     | threshold: 0.9                                                       | [Details](../module4_defense_application/results/backdoor/static_patch/spectral_signatures_report.md)          |
| Backdoor          | Static Patch    | Anomaly Detection     | 60.58%                     | 66.18%                      | 19.76%                     | type: isolation_forest                                               | [Details](../module4_defense_application/results/backdoor/static_patch/anomaly_detection_report.md)            |
| Backdoor          | Static Patch    | Pruning               | 60.58%                     | 33.49%                      | 11.43%                     | pruning_ratio: 0.2, scope: all_layers                                | [Details](../module4_defense_application/results/backdoor/static_patch/pruning_report.md)                      |
| Backdoor          | Static Patch    | Fine Pruning          | 60.58%                     | 60.00%                      | 35.33%                     | pruning_ratio: 0.2                                                   | [Details](../module4_defense_application/results/backdoor/static_patch/fine_pruning_report.md)                 |
| Backdoor          | Static Patch    | Model Inspection      | 60.58%                     | 62.75%                      | 14.40%                     | layers: ['conv.0.weight', 'conv.0.bias', 'fc.1.weight', 'fc.1.bias'] | [Details](../module4_defense_application/results/backdoor/static_patch/model_inspection_report.md)             |
| Data Poisoning    | Clean Label     | Provenance Tracking   | 62.76%                     | 65.44%                      | N/A                        | granularity: sample                                                  | [Details](../module4_defense_application/results/data_poisoning/clean_label/provenance_tracking_report.md)     |
| Data Poisoning    | Clean Label     | Influence Functions   | 62.76%                     | 65.50%                      | N/A                        | method: grad_influence, sample_size: 500                             | [Details](../module4_defense_application/results/data_poisoning/clean_label/influence_functions_report.md)     |
| Data Poisoning    | Label Flipping  | Data Cleaning         | 54.88%                     | 62.51%                      | N/A                        | method: loss_filtering, threshold: 0.9                               | [Details](../module4_defense_application/results/data_poisoning/label_flipping/data_cleaning_report.md)        |
| Data Poisoning    | Label Flipping  | Per Class Monitoring  | 54.88%                     | 54.88%                      | N/A                        | std_threshold: 2.0                                                   | [Details](../module4_defense_application/results/data_poisoning/label_flipping/per_class_monitoring_report.md) |
| Data Poisoning    | Label Flipping  | Robust Loss           | 54.88%                     | 63.42%                      | N/A                        | type: gce                                                            | [Details](../module4_defense_application/results/data_poisoning/label_flipping/robust_loss_report.md)          |
| Data Poisoning    | Label Flipping  | Dp Training           | 54.88%                     | 36.53%                      | N/A                        | clip_norm: 1.0, delta: 1e-05, epsilon: 2.0                           | [Details](../module4_defense_application/results/data_poisoning/label_flipping/dp_training_report.md)          |
| Evasion           | Pgd             | Adversarial Training  | 67.54%                     | 44.39%                      | 19.42%                     | attack_type: fgsm, epsilon: 0.03                                     | [Details](../module4_defense_application/results/evasion/pgd/adversarial_training_report.md)                   |
| Evasion           | Pgd             | Randomized Smoothing  | 67.54%                     | 24.72%                      | 4.76%                      | sigma: 0.25                                                          | [Details](../module4_defense_application/results/evasion/pgd/randomized_smoothing_report.md)                   |
| Evasion           | Spsa            | Gradient Masking      | 68.60%                     | 67.08%                      | 3.50%                      | strength: 0.5                                                        | [Details](../module4_defense_application/results/evasion/spsa/gradient_masking_report.md)                      |
| Evasion           | Spsa            | Jpeg Preprocessing    | 68.60%                     | 64.45%                      | 36.50%                     | quality: 75                                                          | [Details](../module4_defense_application/results/evasion/spsa/jpeg_preprocessing_report.md)                    |

**Note**: 'Clean Acc. (Pre-Defense)' for data poisoning and backdoor attacks refers to the model's accuracy on clean data *after* the initial attack training/injection (the compromised model's clean accuracy). For evasion attacks, it's the original model's clean accuracy. 'Adv. Acc. (Post-Defense)' for data poisoning attacks is 'N/A' as the primary defense objective is to restore clean accuracy. For backdoor attacks, 'Adv. Acc. (Post-Defense)' indicates the model's accuracy on backdoored inputs *after* defense, where a higher value signifies better defense against the backdoor's malicious effect. For evasion attacks, it represents the model's accuracy on adversarial examples *after* defense, aiming for a higher value.

### 6.1 Applied Defenses and their Purposes
The following defenses were applied and evaluated to mitigate the identified risks:

* **Activation Clustering**: A defense aimed at detecting and neutralizing backdoors in models. It works by clustering intermediate layer activations of the model to identify and isolate training samples containing the malicious trigger, allowing for their removal.
* **Adversarial Training**: One of the most common and effective defenses against evasion attacks. It involves augmenting the training dataset with adversarial examples (generated by the attack itself) and retraining the model. This improves the model's robustness, making it more resistant to future adversarial perturbations.
* **Anomaly Detection**: Applies algorithms to identify data points or model behaviors that deviate significantly from normal patterns, potentially indicating the presence of an attack (e.g., poisoned samples or triggered backdoor inputs).
* **Data Cleaning**: A general approach to remove corrupted, mislabeled, or outlier samples from the training dataset. It aims to improve the overall quality and integrity of the data, thereby making the model more robust to various forms of data-based attacks, including poisoning.
* **Dp Training**: Differentially Private Training adds noise to the training process (e.g., to gradients) to protect the privacy of individual training data points. While primarily for privacy, it can also offer some robustness benefits against certain data poisoning attacks by limiting the influence of individual samples.
* **Fine Pruning**: A defense method primarily against backdoor attacks. It involves pruning specific neurons or connections in the neural network that are highly activated by the backdoor trigger but are less critical for clean accuracy, effectively disrupting the backdoor's functionality.
* **Gradient Masking**: This defense aims to obscure or modify the gradients seen by an attacker, making it harder for gradient-based adversarial attacks to succeed. It can involve various techniques like non-differentiable transformations or adding noise to gradients.
* **Influence Functions**: This technique is used to identify and remove training samples that have a disproportionate or negative influence on the model. It is particularly effective against data poisoning attacks, such as 'Clean Label', by helping to purify the training dataset.
* **Jpeg Preprocessing**: A simple defense that applies JPEG compression to inputs before feeding them to the model. The compression process can flatten out small adversarial perturbations, making the adversarial examples less effective against the model.
* **Model Inspection**: Involves analyzing the internal states and behaviors of the model (e.g., activations, weights) to identify anomalies or patterns indicative of malicious injections like backdoors. This is a diagnostic defense often used in conjunction with other mitigation techniques.
* **Per Class Monitoring**: This defense involves monitoring the model's performance or internal states on a per-class basis. Anomalies in specific class predictions or feature distributions can indicate a targeted attack, such as label flipping, allowing for timely intervention.
* **Provenance Tracking**: This defense focuses on tracing the origin and modifications of data throughout the pipeline. By maintaining a verifiable history of data, it helps detect and prevent data poisoning by identifying unauthorized or malicious alterations to the training set.
* **Pruning**: Reduces the size of the neural network by removing less important connections or neurons. While often used for model compression, it can also help remove redundant capacity that might be exploited by certain attacks, including backdoors.
* **Randomized Smoothing**: A certified defense that provides provable robustness guarantees against adversarial attacks. It works by adding random noise to inputs during inference and then classifying based on the aggregated predictions, making it difficult for an attacker to craft effective adversarial examples.
* **Robust Loss**: Utilizing loss functions that are less sensitive to noisy or adversarial labels during training. This can help the model learn more robust features and reduce the impact of poisoned data.
* **Spectral Signatures**: A backdoor detection technique that analyzes the spectral properties of the hidden layer activations. It identifies anomalous patterns indicative of a backdoor trigger embedded in the training data, allowing for the isolation and mitigation of poisoned samples.
