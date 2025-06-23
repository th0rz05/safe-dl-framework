# Safe-DL Framework - Final Security Report
**Profile Selected**: `visitech.yaml`
**Report Generated On**: 2025-06-23 12:56:36

---

## 1. Introduction and Overview
This comprehensive report aggregates the findings from the Safe-DL framework's security assessment, covering threat modeling, attack simulation, risk analysis, defense application, and defense evaluation. Its purpose is to provide a unified overview of the deep learning model's security posture against adversarial threats, document the mitigation strategies applied, and quantify their effectiveness. This dossier serves as a critical resource for decision-making regarding model deployment and continuous security improvement.

## 2. System Under Evaluation Details
This section details the core components of the system analyzed in this report, as defined in the selected threat profile.

### 2.1 Model Details
- **Name**: `resnet18`
- **Type**: `builtin`
- **Input Shape**: `[3, 32, 32]`
- **Number of Classes**: `10`


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
| Data Poisoning | Label Flipping | 82.93% | 65.78% | 65.78% (Degraded Acc.) | Flip Rate: 0.08, Target Class: 9 | [Details](../module2_attack_simulation/results/data_poisoning/label_flipping/label_flipping_report.md) |
| Backdoor | Static Patch | 82.93% | 69.40% | 3.00% (ASR) | Poison Frac.: 0.15, Target Class: 7, Patch Type: white_square | [Details](../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md) |
| Evasion | Pgd | 82.93% | 83.70% | 19.10% (Adv. Acc.) | Epsilon: 0.03, Num Iter: 50 | [Details](../module2_attack_simulation/results/evasion/pgd/pgd_report.md) |
| Evasion | Spsa | 82.93% | 82.40% | 29.20% (Adv. Acc.) | Epsilon: 0.03, Num Steps: 150, Delta: 0.01 | [Details](../module2_attack_simulation/results/evasion/spsa/spsa_report.md) |

**Note**: 'Clean Acc. (Pre-Attack)' represents the model's accuracy on clean data before any attack preparations. 'Impact on Clean Acc.' shows the model's accuracy on clean data *after* being subjected to the attack (e.g., trained with poisoned data, or backdoor injected). For Data Poisoning attacks, 'Attack Metric' displays the degraded accuracy of the model on clean inputs after poisoning. For Backdoor attacks, 'Attack Metric' displays the Attack Success Rate (ASR), indicating the percentage of adversarial samples (with trigger) successfully misclassified to the target class. For Evasion attacks, 'Attack Metric' displays the Adversarial Accuracy (Adv. Acc.) on perturbed inputs, where a lower value indicates a more successful attack.



## 5. Risk Analysis (Module 3)

This section summarizes the risk assessment performed on the simulated attacks. Each attack is evaluated based on its severity, probability, and visibility. A final risk score is computed to help prioritize mitigation strategies, followed by specific defense recommendations.


### 5.1 Risk Summary Table

| Attack | Type | Severity | Probability | Visibility | Risk Score | Report |
|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| Pgd | Evasion | 1.00 | 1.00 | 0.30 | 1.70 | [Details](../module2_attack_simulation/results/evasion/pgd/pgd_report.md) |
| Spsa | Evasion | 1.00 | 0.80 | 0.20 | 1.44 | [Details](../module2_attack_simulation/results/evasion/spsa/spsa_report.md) |
| Label Flipping | Data Poisoning | 0.57 | 1.00 | 0.62 | 0.79 | [Details](../module2_attack_simulation/results/data_poisoning/label_flipping/label_flipping_report.md) |
| Static Patch | Backdoor | 0.34 | 1.00 | 0.60 | 0.47 | [Details](../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md) |


### 5.2 Risk Matrix (Qualitative)

This matrix categorizes attacks based on their qualitative Severity and Probability levels.

| Severity \ Probability | Low | Medium | High |
|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| Low | - | - | - |
| Medium | - | - | Label Flipping, Static Patch |
| High | - | - | Pgd, Spsa |


### 5.3 Risk Ranking

Attacks ranked by their calculated Risk Score, from highest to lowest.

1. **Pgd** — Risk Score: 1.70 → [Details](../module2_attack_simulation/results/evasion/pgd/pgd_report.md)
2. **Spsa** — Risk Score: 1.44 → [Details](../module2_attack_simulation/results/evasion/spsa/spsa_report.md)
3. **Label Flipping** — Risk Score: 0.79 → [Details](../module2_attack_simulation/results/data_poisoning/label_flipping/label_flipping_report.md)
4. **Static Patch** — Risk Score: 0.47 → [Details](../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md)



### 5.4 Defense Recommendations

Based on the identified risks and threat profile, the following defense recommendations are provided:

- **Label Flipping**:
  - data_cleaning
  - per_class_monitoring
- **Pgd**:
  - adversarial_training
  - randomized_smoothing
- **Spsa**:
  - gradient_masking
  - jpeg_preprocessing
- **Static Patch**:
  - activation_clustering
  - spectral_signatures



### 5.5 Paths to Details

For more in-depth information about individual attacks, including raw metrics, attack visualizations, and specific parameters, please refer to the detailed reports linked in the 'Risk Summary Table' and 'Risk Ranking' sections above.

---
## 6. Defense Application (Module 4)
This section details the performance of the implemented defenses against the simulated attacks identified in the risk analysis. For each attack, the table shows the model's accuracy on clean data *before* and *after* defense, and the metric on malicious inputs *before* and *after* defense (ASR for backdoor, adversarial accuracy for evasion). Key defense parameters are also provided, along with a link to a detailed report.

| Attack Category   | Attack Method   | Defense Applied       | Clean Acc. (Pre-Defense)   | Metric on Malicious Inputs (Pre-Defense)   | Clean Acc. (Post-Defense)   | Metric on Malicious Inputs (Post-Defense)   | Key Parameters                         | Link to Details                                                                                         |
|:------------------|:----------------|:----------------------|:---------------------------|:-------------------------------------------|:----------------------------|:--------------------------------------------|:---------------------------------------|:--------------------------------------------------------------------------------------------------------|
| Backdoor          | Static Patch    | Activation Clustering | 69.40%                     | ASR: 3.00%                                 | 61.54%                      | ASR: 0.09%                                  | num_clusters: 2                        | [Details](../module4_defense_application/results/backdoor/static_patch/activation_clustering_report.md) |
| Backdoor          | Static Patch    | Spectral Signatures   | 69.40%                     | ASR: 3.00%                                 | 51.58%                      | ASR: 3.76%                                  | threshold: 0.9                         | [Details](../module4_defense_application/results/backdoor/static_patch/spectral_signatures_report.md)   |
| Data Poisoning    | Label Flipping  | Data Cleaning         | 65.78%                     | N/A                                        | 68.37%                      | N/A                                         | method: loss_filtering, threshold: 0.9 | [Details](../module4_defense_application/results/data_poisoning/label_flipping/data_cleaning_report.md) |
| Evasion           | Pgd             | Adversarial Training  | 83.70%                     | Adv. Acc.: 19.10%                          | 76.55%                      | Adv. Acc.: 41.88%                           | attack_type: fgsm, epsilon: 0.03       | [Details](../module4_defense_application/results/evasion/pgd/adversarial_training_report.md)            |
| Evasion           | Pgd             | Randomized Smoothing  | 83.70%                     | Adv. Acc.: 19.10%                          | 37.20%                      | Adv. Acc.: 19.88%                           | sigma: 0.25                            | [Details](../module4_defense_application/results/evasion/pgd/randomized_smoothing_report.md)            |
| Evasion           | Spsa            | Gradient Masking      | 82.40%                     | Adv. Acc.: 29.20%                          | 82.80%                      | Adv. Acc.: 29.60%                           | strength: 0.5                          | [Details](../module4_defense_application/results/evasion/spsa/gradient_masking_report.md)               |
| Evasion           | Spsa            | Jpeg Preprocessing    | 82.40%                     | Adv. Acc.: 29.20%                          | 11.94%                      | Adv. Acc.: 29.40%                           | quality: 75                            | [Details](../module4_defense_application/results/evasion/spsa/jpeg_preprocessing_report.md)             |

**Note**:
- **Clean Acc. (Pre-Defense)**: Accuracy of the attacked or original model on clean data before applying defense. For data poisoning and backdoor, this is the compromised model’s clean accuracy after poisoning/injection; for evasion, the original model’s clean accuracy.
- **Metric on Malicious Inputs (Pre-Defense)**: For evasion, “Adv. Acc.” on adversarial examples before defense (lower means a stronger attack). For backdoor, “ASR” (Attack Success Rate) on triggered inputs before defense (higher means a more successful backdoor). Marked N/A for data poisoning.
- **Clean Acc. (Post-Defense)**: Accuracy on clean data after defense is applied; indicates how well clean performance is maintained or restored.
- **Metric on Malicious Inputs (Post-Defense)**: For evasion, “Adv. Acc.” on adversarial examples after defense (higher is better). For backdoor, “ASR” on triggered inputs after defense (lower is better). Marked N/A for data poisoning.
- A lower ASR after defense indicates stronger mitigation of the backdoor; a higher Adv. Acc. after defense indicates stronger robustness against evasion.

### 6.1 Applied Defenses and their Purposes
The following defenses were applied and evaluated to mitigate the identified risks:

* **Activation Clustering**: A defense aimed at detecting and neutralizing backdoors in models. It works by clustering intermediate layer activations of the model to identify and isolate training samples containing the malicious trigger, allowing for their removal.
* **Adversarial Training**: One of the most common and effective defenses against evasion attacks. It involves augmenting the training dataset with adversarial examples (generated by the attack itself) and retraining the model. This improves the model's robustness, making it more resistant to future adversarial perturbations.
* **Data Cleaning**: A general approach to remove corrupted, mislabeled, or outlier samples from the training dataset. It aims to improve the overall quality and integrity of the data, thereby making the model more robust to various forms of data-based attacks, including poisoning.
* **Gradient Masking**: This defense aims to obscure or modify the gradients seen by an attacker, making it harder for gradient-based adversarial attacks to succeed. It can involve various techniques like non-differentiable transformations or adding noise to gradients.
* **Jpeg Preprocessing**: A simple defense that applies JPEG compression to inputs before feeding them to the model. The compression process can flatten out small adversarial perturbations, making the adversarial examples less effective against the model.
* **Randomized Smoothing**: A certified defense that provides provable robustness guarantees against adversarial attacks. It works by adding random noise to inputs during inference and then classifying based on the aggregated predictions, making it difficult for an attacker to craft effective adversarial examples.
* **Spectral Signatures**: A backdoor detection technique that analyzes the spectral properties of the hidden layer activations. It identifies anomalous patterns indicative of a backdoor trigger embedded in the training data, allowing for the isolation and mitigation of poisoned samples.

---
## 7. Defense Evaluation (Module 5)
This section presents the evaluation of applied defenses, summarizing their mitigation effectiveness, impact on clean accuracy, computational/resource cost, and overall final score. We also highlight the top-performing defenses for each attack method and discuss key observations.

### 7.1 Summary Table of Defense Evaluation Scores

| Attack Category   | Attack Method   | Defense               |   Mitigation |   CAD |   Cost |   Final Score |
|:------------------|:----------------|:----------------------|-------------:|------:|-------:|--------------:|
| Backdoor          | Static Patch    | Activation Clustering |       -0.105 | 0.214 |    0.3 |        -0.017 |
| Backdoor          | Static Patch    | Spectral Signatures   |        0.027 | 0     |    0.5 |         0     |
| Data Poisoning    | Label Flipping  | Data Cleaning         |        0.151 | 0     |    0.2 |         0     |
| Evasion           | Pgd             | Adversarial Training  |        0.357 | 0.362 |    0.8 |         0.072 |
| Evasion           | Pgd             | Randomized Smoothing  |        0.012 | 0     |    0.5 |         0     |
| Evasion           | Spsa            | Gradient Masking      |        0.007 | 0.987 |    0.4 |         0.005 |
| Evasion           | Spsa            | Jpeg Preprocessing    |        0.004 | 0     |    0.1 |         0     |

### 7.2 Top-Performing Defenses

- **Backdoor / Static Patch**: Top defense is **Spectral Signatures** (Mitigation: 0.027, CAD: 0.000, Cost: 0.500, Final Score: 0.000).
- **Data Poisoning / Label Flipping**: Top defense is **Data Cleaning** (Mitigation: 0.151, CAD: 0.000, Cost: 0.200, Final Score: 0.000).
- **Evasion / Pgd**: Top defense is **Adversarial Training** (Mitigation: 0.357, CAD: 0.362, Cost: 0.800, Final Score: 0.072).
- **Evasion / Spsa**: Top defense is **Gradient Masking** (Mitigation: 0.007, CAD: 0.987, Cost: 0.400, Final Score: 0.005).

### 7.3 Observations and Recommendations

Based on the evaluation scores above, consider the following:

**Detailed per-attack-method rankings:**
**Backdoor / Static Patch**:
- Spectral Signatures: Final Score 0.000 (Mitigation 0.027, CAD 0.000, Cost 0.500) — net zero or negative (no effective balance: mitigation too small relative to cost or clean-accuracy impact).
- Activation Clustering: Final Score -0.017 (Mitigation -0.105, CAD 0.214, Cost 0.300) — net zero or negative (no effective balance: no effective mitigation).

**Data Poisoning / Label Flipping**:
- Data Cleaning: Final Score 0.000 (Mitigation 0.151, CAD 0.000, Cost 0.200) — net zero or negative (no effective balance: mitigation too small relative to cost or clean-accuracy impact).

**Evasion / Pgd**:
- Adversarial Training: Final Score 0.072 (Mitigation 0.357, CAD 0.362, Cost 0.800) 
- Randomized Smoothing: Final Score 0.000 (Mitigation 0.012, CAD 0.000, Cost 0.500) — net zero or negative (no effective balance: mitigation too small relative to cost or clean-accuracy impact).

**Evasion / Spsa**:
- Gradient Masking: Final Score 0.005 (Mitigation 0.007, CAD 0.987, Cost 0.400) — marginal improvement; likely not worth deploying alone.
- Jpeg Preprocessing: Final Score 0.000 (Mitigation 0.004, CAD 0.000, Cost 0.100) — net zero or negative (no effective balance: mitigation too small relative to cost or clean-accuracy impact).

**Overall Recommendation:**
- **Backdoor / Static Patch**: Top defense is **Spectral Signatures** (Final Score 0.000) — No defense shows clear positive net benefit; consider revisiting defense configurations or exploring alternate methods.
- **Data Poisoning / Label Flipping**: Top defense is **Data Cleaning** (Final Score 0.000) — No defense shows clear positive net benefit; consider revisiting defense configurations or exploring alternate methods.
- **Evasion / Pgd**: Top defense is **Adversarial Training** (Final Score 0.072) — Recommended.
- **Evasion / Spsa**: Top defense is **Gradient Masking** (Final Score 0.005) — Only marginal benefit; consider combined approaches or reevaluate cost versus gain.

For more details, refer to the full defense evaluation report: [Details](../module5_defense_evaluation/results/defense_evaluation_report.md).

---
## 8. Conclusions and Executive Summary

**Highest-Risk Attack:** Pgd (Risk Score: 1.700).
- Severity: 1.000, Probability: 1.000, Visibility: 0.300.
**Also high risk:** Spsa (1.440), Label Flipping (0.787).

**Most Effective Defenses Identified:**
- Against **Static Patch**, top defense: **Spectral Signatures** (Final Score: 0.000).
- Against **Label Flipping**, top defense: **Data Cleaning** (Final Score: 0.000).
- Against **Pgd**, top defense: **Adversarial Training** (Final Score: 0.072).
- Against **Spsa**, top defense: **Gradient Masking** (Final Score: 0.005).

**Notable Gaps:**
- The following attack methods showed no defense with positive net benefit at current settings: Label Flipping, Static Patch.

**Overall Security Posture:**
- Pgd identified as highest risk. Effective defenses identified for most attacks, except some evasion methods.

**Practical Recommendations:**
- Prioritize deploying **Adversarial Training** against Pgd.
- Prioritize deploying **Gradient Masking** against Spsa.
- For Label Flipping, revisit defense parameters or explore alternative defenses, as none yielded positive net benefit.
- For Static Patch, revisit defense parameters or explore alternative defenses, as none yielded positive net benefit.

---
## 9. Recommendations for Continuous Monitoring and Post-Deployment

### 9.1 Monitoring Metrics
- **Input Distribution Monitoring**: Continuously track statistics of incoming data (e.g., feature distributions, class frequencies). Unexpected shifts may signal data drift or adversarial attempts.
- **Model Performance Metrics**: Monitor live accuracy or proxy metrics on clean-like validation streams if available. Sudden drops could indicate emerging attacks or data issues.
- **Confidence and Uncertainty**: Log model confidence scores and uncertainty metrics (e.g., softmax entropy). A rise in low-confidence predictions or abnormal confidence patterns can hint at adversarial inputs.
- **Error Rates per Class**: Track per-class error rates over time. Spikes in errors for specific classes may indicate targeted data poisoning or evolving distribution shifts.
- **Resource Usage and Latency**: Monitor inference latency and resource consumption, especially if defenses (e.g., input preprocessing) are in place. Degradation may affect user experience and could be exploited.

### 9.2 Periodic Re-assessment
- **Scheduled Security Audits**: Automate rerunning Modules 2–5 on updated data or model versions at regular intervals (e.g., quarterly or upon major model updates).
- **Retraining with Fresh Data**: If new data is collected over time, include it in retraining pipelines with relevant attack/defense simulations to ensure up-to-date robustness.
- **Threat Landscape Updates**: Stay informed about new attack methods; incorporate new simulations and defenses into the framework as they emerge.
- **Regression Testing**: After model updates or defense adjustments, re-evaluate known attack scenarios to ensure no regressions in vulnerability.

### 9.3 Alerting and Thresholds
- **Define Alert Conditions**: Establish thresholds for monitored metrics (e.g., sudden shift in input distribution, drop in accuracy beyond X%, unusual rise in low-confidence predictions).
- **Automated Alerts**: Connect monitoring to alerting systems (e.g., email, Slack) to notify stakeholders when thresholds are crossed.
- **Anomaly Detection on Incoming Requests**: Deploy runtime anomaly detection to flag suspicious inputs (e.g., out-of-distribution or adversarial patterns).
- **Logging and Auditing**: Maintain detailed logs of input features, predictions, confidence, and any preprocessing steps, to facilitate forensic analysis after an alert.

### 9.4 Data Drift and Concept Drift
- **Drift Detection Techniques**: Use statistical tests or drift-detection algorithms (e.g., population stability index, KS test) to identify significant changes in input feature distributions.
- **Model Retraining Triggers**: Define criteria for retraining when drift is detected (e.g., sustained drift over time or performance degradation beyond threshold).
- **Adversarial Drift Monitoring**: Pay attention to shifts that may be induced by adversarial behavior; correlate drift alerts with security incidents.

### 9.5 Model Versioning and Rollback Plans
- **Version Control for Models**: Tag and store model artifacts (weights, configurations) with version identifiers. Ensure easy retrieval of previous safe versions.
- **Canary Deployments**: Roll out new model versions to a subset of traffic first; monitor metrics closely before full deployment.
- **Rollback Procedures**: Define automated or manual rollback processes if new vulnerabilities or performance regressions are detected.
- **A/B Testing Isolation**: When testing different model versions, isolate traffic to prevent cross-contamination of potential attacks.

### 9.6 Security Incident Response
- **Incident Response Plan**: Document steps to take when an attack or anomaly is detected (e.g., isolate affected services, trigger deeper forensic analysis).
- **Containment Strategies**: If an ongoing attack is detected (e.g., data poisoning detected in training pipeline), halt training or deployment until issue is resolved.
- **Remediation Actions**: Procedures for retraining or patching the model, applying additional defenses, or updating preprocessing pipelines.
- **Post-Incident Review**: After an incident, analyze root causes, update threat model assumptions, and refine monitoring and defense strategies accordingly.
- **Stakeholder Communication**: Establish clear communication channels and responsibilities for notifying relevant teams (e.g., security, ML engineering, product) during incidents.

### 9.7 Integration into CI/CD Pipelines
- **Automated Security Checks**: Incorporate automated runs of attack/defense simulations (Modules 2–5) into CI pipelines triggered by code or data changes.
- **Gatekeeping Deployments**: Block deployments if security evaluation metrics fall below predefined thresholds (e.g., risk score above threshold, defense evaluation final score below threshold).
- **Continuous Reporting**: Generate and archive periodic security reports; notify stakeholders of changes in security posture.
