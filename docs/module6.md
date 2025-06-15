
# Module 6 — Final Report Aggregation

This document details Module 6 of the Safe-DL framework, focusing on the crucial final step of aggregating all security assessment results into a comprehensive, audit-ready report. This module brings together the insights from threat modeling, attack simulations, risk analysis, and defense evaluations to provide a holistic view of the deep learning system's security posture.

## Table of Contents

- [1. Introduction](#1-introduction)
  * [1.1 Goal of the Module](#11-goal-of-the-module)
  * [1.2 The Importance of Aggregated Reporting](#12-the-importance-of-aggregated-reporting)
- [2. Module Objectives](#2-module-objectives)
  * [2.1 Primary Objectives](#21-primary-objectives)
  * [2.2 Secondary Objectives](#22-secondary-objectives)
- [3. Inputs and Outputs](#3-inputs-and-outputs)
  * [3.1 Inputs (Data Sources)](#31-inputs--data-sources-)
  * [3.2 Outputs (Generated Artifacts)](#32-outputs--generated-artifacts-)
- [4. Implementation and Execution Flow](#4-implementation-and-execution-flow)
  * [4.1 Key Scripts](#41-key-scripts)
  * [4.2 Execution Steps](#42-execution-steps)
- [5. Structure of the Final Report](#5-structure-of-the-final-report)
  * [5.1 Report Header and Overview](#51-report-header-and-overview)
  * [5.2 System Under Evaluation Details](#52-system-under-evaluation-details)
  * [5.3 Threat Profile Summary](#53-threat-profile-summary)
  * [5.4 Attack Simulation Results](#54-attack-simulation-results)
  * [5.5 Risk Analysis and Matrix](#55-risk-analysis-and-matrix)
  * [5.6 Defense Application Summary](#56-defense-application-summary)
  * [5.7 Defense Evaluation and Scoring](#57-defense-evaluation-and-scoring)
  * [5.8 Conclusions and Executive Summary](#58-conclusions-and-executive-summary)
  * [5.9 Recommendations for Continuous Monitoring and Post-Deployment](#59-recommendations-for-continuous-monitoring-and-post-deployment)
- [6. Example Output](#6-example-output)
  * [6.1 Snippet of `final_report.md`](#61-snippet-of--final-reportmd-)
- [7. Integration with the Framework](#7-integration-with-the-framework)
  * [7.1 Execution Flow](#71-execution-flow)
  * [7.2 Dependency on Previous Modules](#72-dependency-on-previous-modules)
  * [7.3 `generate_report_utils.py`](#73--generate-report-utilspy-)


## 1. Introduction

This section provides an overview of Module 6, outlining its purpose within the Safe-DL framework and emphasizing the significance of producing a consolidated security report.

### 1.1 Goal of the Module

Module 6, "Final Report Aggregation," serves as the culmination of the entire Safe-DL framework pipeline. Its primary goal is to collect, synthesize, and present all the findings from the preceding modules—Threat Modeling (Module 1), Attack Simulation (Module 2), Risk Analysis (Module 3), Defense Application (Module 4), and Defense Evaluation (Module 5)—into a single, coherent, and audit-ready security report. This module transforms raw data and individual module reports into a comprehensive dossier, providing a holistic view of the deep learning system's security posture.

### 1.2 The Importance of Aggregated Reporting

In complex machine learning systems, understanding the security landscape requires more than isolated metrics. Aggregated reporting is crucial for combining diverse data points into a unified narrative, enabling a clearer understanding of vulnerabilities and the effectiveness of security measures.


## 2. Module Objectives

Module 6 is designed with specific aims to ensure that the Safe-DL framework delivers a complete and actionable security assessment. These objectives guide the data aggregation and report generation processes.

### 2.1 Primary Objectives

The core objectives of Module 6 are centered on creating a consolidated and informative security report:

* **Aggregate Data from All Modules**: Collect relevant outputs (JSON, YAML, Markdown snippets) from Modules 1 through 5, ensuring all critical information is brought together.
* **Synthesize and Structure Findings**: Process raw data into human-readable formats, organizing it into logical sections within a single Markdown document.
* **Generate Comprehensive Security Report**: Produce a `final_report.md` file that encapsulates the entire security assessment, from initial threat profiling to defense evaluation and actionable recommendations.
* **Ensure Reproducibility**: By drawing data directly from the `profile.yaml` (which is enriched by each module's results), ensure that the report accurately reflects the specific assessment scenario and can be regenerated.

### 2.2 Secondary Objectives

Beyond its primary goals, Module 6 also serves several important secondary objectives that enhance the framework's overall utility and robustness:

* **Establish a "Single Source of Truth"**: The final report acts as the definitive record of the security assessment, consolidating all findings and decisions in one place, minimizing discrepancies across different partial reports.
* **Facilitate Communication**: Provide a clear, accessible document that can be shared with various stakeholders (developers, security engineers, management) to communicate the system's security posture and identified risks.
* **Support Audit and Compliance**: Generate a structured report that can serve as evidence for security audits and compliance requirements, demonstrating a systematic approach to ML security.
* **Enable Iterative Improvement**: By summarizing key findings and recommendations, the report helps in identifying areas for future research, defense enhancement, or process refinement in an iterative security lifecycle.


## 3. Inputs and Outputs

This section details the various data sources consumed by Module 6 and the primary artifacts it generates, illustrating its role as the reporting hub of the Safe-DL framework.

### 3.1 Inputs (Data Sources)

Module 6 relies on the incrementally updated `profile.yaml` file as its central data source. This profile gathers the aggregated results from all preceding modules, ensuring that the final report is comprehensive and reflects the entire assessment pipeline.

* **Profile YAML (`<profile_name>.yaml`)**: This is the most critical input. The `profile.yaml` file, located in the `profiles/` directory, is continuously enriched by each module (1-5) with their respective findings. It contains:
    * **Module 1 (Threats)**: Initial threat model definitions, including `model_access`, `attack_goal`, `threat_categories`, etc.
    * **Module 2 Results (Attack Metrics)**: Details of simulated attacks, including `attack_metrics.json` summaries which are embedded or referenced in the profile. These include clean accuracy, attack success rates, and other relevant metrics per attack type (e.g., evasion, backdoor, data poisoning).
    * **Module 3 Results (Risk Analysis)**: The `risk_analysis.json` data, which includes calculated severity, probability, visibility, and total risk scores for each identified threat, along with risk matrix data.
    * **Module 4 Results (Defense Application Reports)**: Information on which defenses were applied, their configurations, and potentially any direct output from the defense application process (e.g., details on removed samples, pruning histograms, specific defense parameters used). While not always a single `.json` file, the relevant information is integrated into the profile.
    * **Module 5 Results (Defense Evaluation)**: The `defense_evaluation.json` (and associated `defense_evaluation_report.md` snippets) containing the calculated mitigation scores, clean accuracy drop, cost scores, and final defense scores for each applied defense mechanism.

### 3.2 Outputs (Generated Artifacts)

The primary output of Module 6 is a comprehensive Markdown report, designed for readability and auditability.

* **`final_report.md`**: This is the main artifact generated by Module 6. It's a human-readable Markdown file located in the `reports/` directory that consolidates all information, tables, and summaries from the entire Safe-DL assessment. It is structured to provide a clear narrative from threat identification to defense evaluation, including conclusions and recommendations. As per the framework's design, this file is typically named `final_report_<timestamp>.md` to maintain a history of reports, although a simplified `final_report.md` can also be generated for convenience during development or for quick overwrites.
* **(Future) `final_report.pdf`**: While not currently implemented in the provided code, the framework design anticipates the future capability to generate a PDF version of the report. This would offer a static, printable, and easily shareable document format, suitable for official audits or non-technical stakeholders.

## 4. Implementation and Execution Flow

Module 6's implementation is structured into two main Python scripts that orchestrate the report generation process, from user interaction to final file output.

### 4.1 Key Scripts

The functionality of Module 6 is primarily encapsulated within two distinct Python files, each with well-defined responsibilities:

* **`run_module6.py`**: This is the main executable script for Module 6. It handles:
    * User interaction (e.g., selecting the threat profile).
    * Loading the chosen threat profile.
    * Orchestrating calls to the utility functions in `generate_report_utils.py` to build the report sections.
    * Writing the assembled report content to the final Markdown file.
    * Setting up the necessary directory structure for reports.
    It acts as the command-line interface for the module.

* **`generate_report_utils.py`**: This file serves as a library of utility functions specifically designed for generating various parts of the final report. It contains:
    * Helper functions for safely loading YAML and JSON data from other module results.
    * Constants defining the paths to results directories of other modules (e.g., `MODULE2_RESULTS_DIR`).
    * Individual functions (e.g., `generate_report_header`, `generate_system_details_section`, `generate_attack_simulation_section`) that each construct a specific segment of the Markdown report by extracting and formatting data from the loaded `profile_data`.
    * Logic for formatting tables using the `tabulate` library.
    This separation ensures a clean, modular design where the report generation logic is reusable and independent of the execution flow.

### 4.2 Execution Steps

The execution of Module 6 follows a clear sequence of operations to ensure all data is correctly aggregated and presented in the final report.

1.  **Profile Selection**:
    * Upon execution of `run_module6.py`, the user is prompted to select an existing threat profile (`.yaml` file) from the `profiles/` directory. This profile is the central data source that has been progressively updated by all preceding modules (1-5).
2.  **Data Aggregation Logic**:
    * The selected `profile.yaml` is loaded into memory. This `profile_data` dictionary now contains all the relevant information from threat modeling, attack simulations, risk analysis, defense application, and defense evaluation.
    * Utility functions from `generate_report_utils.py` are called sequentially. Each function takes the `profile_data` as input, extracts the specific information it needs, formats it (e.g., as Markdown tables, bullet points, or paragraphs), and returns a string representing that section of the report.
3.  **Sequential Report Generation**:
    * The `run_module6.py` script appends the output of each report generation utility function to a list of lines, building the report content section by section. This includes:
        * Report Header and General Overview.
        * System Under Evaluation Details.
        * Threat Profile Summary (Module 1).
        * Attack Simulation Results (Module 2).
        * Risk Analysis and Matrix (Module 3).
        * Defense Application Summary (Module 4).
        * Defense Evaluation and Scoring (Module 5).
        * Conclusions, Recommendations, Monitoring, and CI/CD integration.
4.  **File Output**:
    * Once all sections are generated, the complete report content (as a single string) is written to a Markdown file, typically named `final_report.md` (or `final_report_<timestamp>.md` for historical archiving), within the `reports/` directory. A confirmation message with the output file path is displayed to the user.


## 5. Structure of the Final Report

The `final_report.md` generated by Module 6 is meticulously structured to provide a logical flow of information, starting from system details and threat definitions, through attack and defense results, and culminating in actionable conclusions and recommendations. This structure ensures comprehensive coverage and ease of understanding for diverse stakeholders.

### 5.1 Report Header and Overview

This initial section sets the stage for the entire report, providing essential identifying information and a high-level summary of the document's purpose.

* **Report Header**: Includes the title "Safe-DL Framework - Final Security Report," the name of the selected profile (e.g., `test.yaml`), and the exact date and time of report generation.
* **Introduction and Overview**: A high-level summary explaining the report's purpose and the scope of the Safe-DL framework assessment, highlighting its comprehensive nature across all security phases.

### 5.2 System Under Evaluation Details 

This section provides fundamental information about the deep learning system that was the subject of the security assessment, pulled directly from the threat profile.

* **Model Details**: Specifies the name, type (e.g., `builtin`, `custom`), input shape, number of classes, and any specific parameters of the deep learning model.
* **Dataset Details**: Identifies the dataset used for training and evaluation, including its name and type (e.g., `builtin`, `custom`).

### 5.3 Threat Profile Summary 

Drawing directly from Module 1, this section summarizes the defined threat profile, which characterizes the deep learning system's environment and the anticipated adversary.

* **Key Parameters**: Lists crucial aspects like `model_access`, `attack_goal`, `deployment_scenario`, `data_sensitivity`, `training_data_source`, `model_type`, and `interface_exposed`.
* **Threat Categories**: Enumerates the specific types of threats identified as relevant based on the threat model (e.g., `data_poisoning`, `adversarial_examples`, `backdoor_attacks`).
* **Notes on Future Work**: Includes specific notes about threat categories that are defined in the profile but might not yet be fully implemented in subsequent modules (e.g., `model_stealing`, `membership_inference`, `model_inversion`).

### 5.4 Attack Simulation Results 

This section presents the empirical outcomes of the adversarial attack simulations conducted in Module 2, quantifying the model's vulnerabilities prior to defense application.

* **Overview of Simulated Attacks Table**: A central table summarizing key metrics for each simulated attack (e.g., data poisoning, backdoor, evasion). Metrics include:
    * Clean Accuracy (Pre-Attack).
    * Impact on Clean Accuracy (for data poisoning/backdoor).
    * Attack Metric (e.g., Attack Success Rate for backdoors, Adversarial Accuracy for evasion).
    * Key Parameters of the attack.
    * Links to detailed module-specific reports for full results.
* **Metric Definitions**: Clear explanations of the meaning of 'Clean Acc. (Pre-Attack)', 'Impact on Clean Acc.', 'Attack Metric', and their implications for different attack types.

### 5.5 Risk Analysis and Matrix 

Derived from Module 3, this section quantifies and prioritizes the identified threats using a standardized risk assessment methodology.

* **Risk Summary Table**: A table detailing the calculated risk scores for each attack, including `Severity`, `Probability`, `Visibility`, and the aggregated `Risk Score`. Each row links to the detailed attack report.
* **Risk Matrix (Qualitative)**: A qualitative matrix categorizing attacks based on their Severity and Probability levels, providing a visual overview of high-impact, high-likelihood threats.
* **Risk Ranking**: A ranked list of attacks by their calculated Risk Score, from highest to lowest, highlighting the most critical vulnerabilities.
* **Defense Recommendations**: Lists specific defense mechanisms recommended for each identified attack based on the risk analysis.
* **Paths to Details**: Explicitly points to the linked reports for more in-depth information about individual attack details.

### 5.6 Defense Application Summary 

This section provides an overview of the defense mechanisms that were applied to the model, as per the configurations and processes of Module 4.

* **Defense Application Table**: A detailed table listing each applied defense. For each entry, it includes:
    * `Attack Category` and `Attack Method` it targeted.
    * `Defense Applied` (name of the defense).
    * `Clean Acc. (Pre-Defense)` and `Metric on Malicious Inputs (Pre-Defense)`.
    * `Clean Acc. (Post-Defense)` and `Metric on Malicious Inputs (Post-Defense)`.
    * `Key Parameters` used for the defense.
    * A `Link to Details` for the defense's specific report.
* **Notes on Metrics**: Provides definitions for the "Pre-Defense" and "Post-Defense" metrics for various attack types (e.g., ASR for backdoor, Adv. Acc. for evasion).
* **Applied Defenses and their Purposes**: A narrative description of each defense listed in the table, explaining its mechanism and general goal (e.g., Activation Clustering for backdoors, Adversarial Training for evasion).

### 5.7 Defense Evaluation and Scoring 

This critical section, based on Module 5, quantifies the effectiveness of each applied defense, detailing its mitigation capabilities, impact on clean accuracy, and overall score.

* **Summary Table of Defense Evaluation Scores**: A concise table summarizing the calculated scores for each defense-attack pair, including:
    * `Mitigation Score`: Quantifies the reduction in attack effectiveness.
    * `CAD` (Clean Accuracy Drop): Measures the performance degradation on benign inputs.
    * `Cost` (Cost Score): An indicator of the computational or operational cost.
    * `Final Score`: An aggregated score reflecting the overall effectiveness and trade-offs.
* **Top-Performing Defenses**: Highlights the best-performing defense for each attack category based on the `Final Score`.
* **Observations and Recommendations**: Detailed qualitative analysis of the evaluation results, including:
    * Per-attack-method rankings of defenses, with their scores.
    * Identification of defenses with marginal or negative net benefit.
    * Overall recommendations based on the highest-scoring defenses.
* **Link to Full Report**: Provides a link to the complete `defense_evaluation_report.md` for more in-depth analysis.

### 5.8 Conclusions and Executive Summary

This section provides a high-level synthesis of the entire security assessment, summarizing key findings and the overall security posture.

* **Highest-Risk Attack**: Identifies the attack with the highest calculated risk score.
* **Most Effective Defenses Identified**: Lists the top-performing defense for each attack category.
* **Notable Gaps**: Points out attack methods where no defense yielded a positive net benefit.
* **Overall Security Posture**: A concise statement on the general security state of the model.
* **Practical Recommendations**: Actionable advice on which defenses to prioritize based on their effectiveness and the identified risks.

### 5.9 Recommendations for Continuous Monitoring and Post-Deployment 

This final section offers forward-looking advice on maintaining and improving the model's security posture after deployment, emphasizing continuous monitoring and incident response.

* **Monitoring Metrics**: Suggests key metrics to track post-deployment (e.g., Input Distribution, Model Performance, Confidence/Uncertainty, Error Rates per Class, Resource Usage).
* **Periodic Re-assessment**: Recommends regular security audits, retraining with fresh data, staying informed about new threats, and regression testing.
* **Alerting and Thresholds**: Advises on defining alert conditions, automated alerts, runtime anomaly detection, and comprehensive logging for forensic analysis.
* **Security Incident Response**: Outlines a plan for handling detected incidents, including containment, remediation, post-incident review, and stakeholder communication.
* **Integration into CI/CD Pipelines**: Discusses incorporating automated security checks, gatekeeping deployments, and continuous reporting within CI/CD workflows to ensure security throughout the development lifecycle.

## 6. Example Output

To fully understand the comprehensive nature of Module 6's output, this section provides extended snippets from a sample `final_report.md`. These examples clearly illustrate how data from all preceding modules is aggregated and presented in a cohesive and readable format.

### 6.1 Snippet of `final_report.md`

This extensive snippet showcases the beginning of a typical `final_report.md`, demonstrating the report header, system details, the full threat profile summary, and the attack simulation results. This allows for a clear view of how information from Modules 1, 2, and the system's core definition are integrated.

```markdown
# Safe-DL Framework - Final Security Report
**Profile Selected**: `sample_profile.yaml`
**Report Generated On**: 2025-06-15 03:07:57

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
| Backdoor | Static Patch | 67.54% | 66.62% | 92.30% (ASR) | Poison Frac.: 0.05, Target Class: 7, Patch Type: white_square | [Details](../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md) |
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
This section details the performance of the implemented defenses against the simulated attacks identified in the risk analysis. For each attack, the table shows the model's accuracy on clean data *before* and *after* defense, and the metric on malicious inputs *before* and *after* defense (ASR for backdoor, adversarial accuracy for evasion). Key defense parameters are also provided, along with a link to a detailed report.

| Attack Category   | Attack Method   | Defense Applied       | Clean Acc. (Pre-Defense)   | Metric on Malicious Inputs (Pre-Defense)   | Clean Acc. (Post-Defense)   | Metric on Malicious Inputs (Post-Defense)   | Key Parameters                                                       | Link to Details                                                                                                |
|:------------------|:----------------|:----------------------|:---------------------------|:-------------------------------------------|:----------------------------|:--------------------------------------------|:---------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|
| Backdoor          | Static Patch    | Activation Clustering | 66.62%                     | ASR: 92.30%                                | 43.86%                      | ASR: 90.24%                                 | num_clusters: 2                                                      | [Details](../module4_defense_application/results/backdoor/static_patch/activation_clustering_report.md)        |
| Backdoor          | Static Patch    | Spectral Signatures   | 66.62%                     | ASR: 92.30%                                | 50.57%                      | ASR: 85.73%                                 | threshold: 0.9                                                       | [Details](../module4_defense_application/results/backdoor/static_patch/spectral_signatures_report.md)          |
| Backdoor          | Static Patch    | Anomaly Detection     | 66.62%                     | ASR: 92.30%                                | 64.16%                      | ASR: 92.04%                                 | type: isolation_forest                                               | [Details](../module4_defense_application/results/backdoor/static_patch/anomaly_detection_report.md)            |
| Backdoor          | Static Patch    | Pruning               | 66.62%                     | ASR: 92.30%                                | 33.61%                      | ASR: 86.98%                                 | pruning_ratio: 0.2, scope: all_layers                                | [Details](../module4_defense_application/results/backdoor/static_patch/pruning_report.md)                      |
| Backdoor          | Static Patch    | Fine Pruning          | 66.62%                     | ASR: 92.30%                                | 67.33%                      | ASR: 51.61%                                 | pruning_ratio: 0.2                                                   | [Details](../module4_defense_application/results/backdoor/static_patch/fine_pruning_report.md)                 |
| Backdoor          | Static Patch    | Model Inspection      | 66.62%                     | ASR: 92.30%                                | 64.40%                      | ASR: 91.53%                                 | layers: ['conv.0.weight', 'conv.0.bias', 'fc.1.weight', 'fc.1.bias'] | [Details](../module4_defense_application/results/backdoor/static_patch/model_inspection_report.md)             |
| Data Poisoning    | Clean Label     | Provenance Tracking   | 62.76%                     | N/A                                        | 65.44%                      | N/A                                         | granularity: sample                                                  | [Details](../module4_defense_application/results/data_poisoning/clean_label/provenance_tracking_report.md)     |
| Data Poisoning    | Clean Label     | Influence Functions   | 62.76%                     | N/A                                        | 65.50%                      | N/A                                         | method: grad_influence, sample_size: 500                             | [Details](../module4_defense_application/results/data_poisoning/clean_label/influence_functions_report.md)     |
| Data Poisoning    | Label Flipping  | Data Cleaning         | 54.88%                     | N/A                                        | 62.51%                      | N/A                                         | method: loss_filtering, threshold: 0.9                               | [Details](../module4_defense_application/results/data_poisoning/label_flipping/data_cleaning_report.md)        |
| Data Poisoning    | Label Flipping  | Per Class Monitoring  | 54.88%                     | N/A                                        | 54.88%                      | N/A                                         | std_threshold: 2.0                                                   | [Details](../module4_defense_application/results/data_poisoning/label_flipping/per_class_monitoring_report.md) |
| Data Poisoning    | Label Flipping  | Robust Loss           | 54.88%                     | N/A                                        | 63.42%                      | N/A                                         | type: gce                                                            | [Details](../module4_defense_application/results/data_poisoning/label_flipping/robust_loss_report.md)          |
| Data Poisoning    | Label Flipping  | Dp Training           | 54.88%                     | N/A                                        | 36.53%                      | N/A                                         | clip_norm: 1.0, delta: 1e-05, epsilon: 2.0                           | [Details](../module4_defense_application/results/data_poisoning/label_flipping/dp_training_report.md)          |
| Evasion           | Pgd             | Adversarial Training  | 67.54%                     | Adv. Acc.: 0.00%                           | 44.39%                      | Adv. Acc.: 19.42%                           | attack_type: fgsm, epsilon: 0.03                                     | [Details](../module4_defense_application/results/evasion/pgd/adversarial_training_report.md)                   |
| Evasion           | Pgd             | Randomized Smoothing  | 67.54%                     | Adv. Acc.: 0.00%                           | 24.72%                      | Adv. Acc.: 4.76%                            | sigma: 0.25                                                          | [Details](../module4_defense_application/results/evasion/pgd/randomized_smoothing_report.md)                   |
| Evasion           | Spsa            | Gradient Masking      | 68.60%                     | Adv. Acc.: 2.20%                           | 67.08%                      | Adv. Acc.: 3.50%                            | strength: 0.5                                                        | [Details](../module4_defense_application/results/evasion/spsa/gradient_masking_report.md)                      |
| Evasion           | Spsa            | Jpeg Preprocessing    | 68.60%                     | Adv. Acc.: 2.20%                           | 64.45%                      | Adv. Acc.: 36.50%                           | quality: 75                                                          | [Details](../module4_defense_application/results/evasion/spsa/jpeg_preprocessing_report.md)                    |

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

## 7. Defense Evaluation (Module 5)
This section presents the evaluation of applied defenses, summarizing their mitigation effectiveness, impact on clean accuracy, computational/resource cost, and overall final score. We also highlight the top-performing defenses for each attack method and discuss key observations.

### 7.1 Summary Table of Defense Evaluation Scores

| Attack Category   | Attack Method   | Defense               |   Mitigation |   CAD |   Cost |   Final Score |
|:------------------|:----------------|:----------------------|-------------:|------:|-------:|--------------:|
| Backdoor          | Static Patch    | Activation Clustering |        0.035 | 0     |    0.3 |         0     |
| Backdoor          | Static Patch    | Spectral Signatures   |        0.111 | 0     |    0.5 |         0     |
| Backdoor          | Static Patch    | Anomaly Detection     |        0.004 | 0.754 |    0.3 |         0.003 |
| Backdoor          | Static Patch    | Pruning               |        0.09  | 0     |    0.3 |         0     |
| Backdoor          | Static Patch    | Fine Pruning          |        0.691 | 1.071 |    0.4 |         0.528 |
| Backdoor          | Static Patch    | Model Inspection      |        0.013 | 0.778 |    0.2 |         0.008 |
| Data Poisoning    | Clean Label     | Provenance Tracking   |        0.561 | 0.79  |    0.5 |         0.295 |
| Data Poisoning    | Clean Label     | Influence Functions   |        0.573 | 0.796 |    0.5 |         0.304 |
| Data Poisoning    | Label Flipping  | Data Cleaning         |        0.603 | 0.497 |    0.2 |         0.25  |
| Data Poisoning    | Label Flipping  | Per Class Monitoring  |        0     | 0     |    0.2 |         0     |
| Data Poisoning    | Label Flipping  | Robust Loss           |        0.675 | 0.588 |    0.5 |         0.264 |
| Data Poisoning    | Label Flipping  | Dp Training           |       -1.449 | 0     |    0.7 |        -0     |
| Evasion           | Pgd             | Adversarial Training  |        0.288 | 0     |    0.8 |         0     |
| Evasion           | Pgd             | Randomized Smoothing  |        0.07  | 0     |    0.5 |         0     |
| Evasion           | Spsa            | Gradient Masking      |        0.02  | 0.954 |    0.4 |         0.014 |
| Evasion           | Spsa            | Jpeg Preprocessing    |        0.525 | 0.691 |    0.1 |         0.33  |

### 7.2 Top-Performing Defenses

- **Backdoor / Static Patch**: Top defense is **Fine Pruning** (Mitigation: 0.691, CAD: 1.071, Cost: 0.400, Final Score: 0.528).
- **Data Poisoning / Clean Label**: Top defense is **Influence Functions** (Mitigation: 0.573, CAD: 0.796, Cost: 0.500, Final Score: 0.304).
- **Data Poisoning / Label Flipping**: Top defense is **Robust Loss** (Mitigation: 0.675, CAD: 0.588, Cost: 0.500, Final Score: 0.264).
- **Evasion / Pgd**: Top defense is **Adversarial Training** (Mitigation: 0.288, CAD: 0.000, Cost: 0.800, Final Score: 0.000).
- **Evasion / Spsa**: Top defense is **Jpeg Preprocessing** (Mitigation: 0.525, CAD: 0.691, Cost: 0.100, Final Score: 0.330).

### 7.3 Observations and Recommendations

Based on the evaluation scores above, consider the following:

**Detailed per-attack-method rankings:**
**Backdoor / Static Patch**:
- Fine Pruning: Final Score 0.528 (Mitigation 0.691, CAD 1.071, Cost 0.400) 
- Model Inspection: Final Score 0.008 (Mitigation 0.013, CAD 0.778, Cost 0.200) — marginal improvement; likely not worth deploying alone.
- Anomaly Detection: Final Score 0.003 (Mitigation 0.004, CAD 0.754, Cost 0.300) — marginal improvement; likely not worth deploying alone.
- Activation Clustering: Final Score 0.000 (Mitigation 0.035, CAD 0.000, Cost 0.300) — net zero or negative (no effective balance: mitigation too small relative to cost or clean-accuracy impact).
- Spectral Signatures: Final Score 0.000 (Mitigation 0.111, CAD 0.000, Cost 0.500) — net zero or negative (no effective balance: mitigation too small relative to cost or clean-accuracy impact).
- Pruning: Final Score 0.000 (Mitigation 0.090, CAD 0.000, Cost 0.300) — net zero or negative (no effective balance: mitigation too small relative to cost or clean-accuracy impact).

**Data Poisoning / Clean Label**:
- Influence Functions: Final Score 0.304 (Mitigation 0.573, CAD 0.796, Cost 0.500) 
- Provenance Tracking: Final Score 0.295 (Mitigation 0.561, CAD 0.790, Cost 0.500) 

**Data Poisoning / Label Flipping**:
- Robust Loss: Final Score 0.264 (Mitigation 0.675, CAD 0.588, Cost 0.500) 
- Data Cleaning: Final Score 0.250 (Mitigation 0.603, CAD 0.497, Cost 0.200) 
- Per Class Monitoring: Final Score 0.000 (Mitigation 0.000, CAD 0.000, Cost 0.200) — net zero or negative (no effective balance: no effective mitigation).
- Dp Training: Final Score -0.000 (Mitigation -1.449, CAD 0.000, Cost 0.700) — net zero or negative (no effective balance: no effective mitigation).

**Evasion / Pgd**:
- Adversarial Training: Final Score 0.000 (Mitigation 0.288, CAD 0.000, Cost 0.800) — net zero or negative (no effective balance: mitigation too small relative to cost or clean-accuracy impact).
- Randomized Smoothing: Final Score 0.000 (Mitigation 0.070, CAD 0.000, Cost 0.500) — net zero or negative (no effective balance: mitigation too small relative to cost or clean-accuracy impact).

**Evasion / Spsa**:
- Jpeg Preprocessing: Final Score 0.330 (Mitigation 0.525, CAD 0.691, Cost 0.100) 
- Gradient Masking: Final Score 0.014 (Mitigation 0.020, CAD 0.954, Cost 0.400) — marginal improvement; likely not worth deploying alone.

**Overall Recommendation:**
- **Backdoor / Static Patch**: Top defense is **Fine Pruning** (Final Score 0.528) — Recommended.
- **Data Poisoning / Clean Label**: Top defense is **Influence Functions** (Final Score 0.304) — Recommended.
- **Data Poisoning / Label Flipping**: Top defense is **Robust Loss** (Final Score 0.264) — Recommended.
- **Evasion / Pgd**: Top defense is **Adversarial Training** (Final Score 0.000) — No defense shows clear positive net benefit; consider revisiting defense configurations or exploring alternate methods.
- **Evasion / Spsa**: Top defense is **Jpeg Preprocessing** (Final Score 0.330) — Recommended.

For more details, refer to the full defense evaluation report: [Details](../module5_defense_evaluation/results/defense_evaluation_report.md).

## 8. Conclusions and Executive Summary

**Highest-Risk Attack:** Deepfool (Risk Score: 1.900).
- Severity: 1.000, Probability: 1.000, Visibility: 0.100.
**Also high risk:** Fgsm (1.700), Pgd (1.700).

**Most Effective Defenses Identified:**
- Against **Static Patch**, top defense: **Fine Pruning** (Final Score: 0.528).
- Against **Clean Label**, top defense: **Influence Functions** (Final Score: 0.304).
- Against **Label Flipping**, top defense: **Robust Loss** (Final Score: 0.264).
- Against **Pgd**, top defense: **Adversarial Training** (Final Score: 0.000).
- Against **Spsa**, top defense: **Jpeg Preprocessing** (Final Score: 0.330).

**Notable Gaps:**
- The following attack methods showed no defense with positive net benefit at current settings: Pgd.

**Overall Security Posture:**
- Deepfool identified as highest risk. Effective defenses identified for most attacks, except some evasion methods.

**Practical Recommendations:**
- Prioritize deploying **Fine Pruning** against Static Patch.
- Prioritize deploying **Influence Functions** against Clean Label.
- Prioritize deploying **Robust Loss** against Label Flipping.
- Prioritize deploying **Jpeg Preprocessing** against Spsa.
- For Pgd, revisit defense parameters or explore alternative defenses, as none yielded positive net benefit.

---
## 9. Recommendations for Continuous Monitoring and Post-Deployment

This section provides crucial guidance for maintaining and enhancing the security posture of the deep learning system after its deployment, focusing on proactive measures and incident preparedness.

### 9.1 Monitoring Metrics
Continuous monitoring is vital for detecting new threats and ensuring the ongoing effectiveness of defenses. Key metrics to track include:

- **Input Distribution Monitoring**: Continuously track statistics of incoming data (e.g., feature distributions, class frequencies). Unexpected shifts may signal data drift or adversarial attempts.
- **Model Performance Metrics**: Monitor live accuracy or proxy metrics on clean-like validation streams if available. Sudden drops could indicate emerging attacks or data issues.
- **Confidence and Uncertainty**: Log model confidence scores and uncertainty metrics (e.g., softmax entropy). A rise in low-confidence predictions or abnormal confidence patterns can hint at adversarial inputs.
- **Error Rates per Class**: Track per-class error rates over time. Spikes in errors for specific classes may indicate targeted data poisoning or evolving distribution shifts.
- **Resource Usage and Latency**: Monitor inference latency and resource consumption, especially if defenses (e.g., input preprocessing) are in place. Degradation may affect user experience and could be exploited.

### 9.2 Periodic Re-assessment
Regular re-evaluation ensures that the security posture remains robust against evolving threats. This includes:

- **Scheduled Security Audits**: Automate rerunning Modules 2–5 on updated data or model versions at regular intervals (e.g., quarterly or upon major model updates).
- **Retraining with Fresh Data**: If new data is collected over time, include it in retraining pipelines with relevant attack/defense simulations to ensure up-to-date robustness.
- **Threat Landscape Updates**: Stay informed about new attack methods; incorporate new simulations and defenses into the framework as they emerge.
- **Regression Testing**: After model updates or defense adjustments, re-evaluate known attack scenarios to ensure no regressions in vulnerability.

### 9.3 Alerting and Thresholds
Effective alerting mechanisms are crucial for timely incident response. Establishing clear thresholds for monitored metrics is key:

- **Define Alert Conditions**: Establish thresholds for monitored metrics (e.g., sudden shift in input distribution, drop in accuracy beyond X%, unusual rise in low-confidence predictions).
- **Automated Alerts**: Connect monitoring to alerting systems (e.g., email, Slack) to notify stakeholders when thresholds are crossed.
- **Anomaly Detection on Incoming Requests**: Deploy runtime anomaly detection to flag suspicious inputs (e.g., out-of-distribution or adversarial patterns).
- **Logging and Auditing**: Maintain detailed logs of input features, predictions, confidence, and any preprocessing steps, to facilitate forensic analysis after an alert.

### 9.4 Security Incident Response
A well-defined incident response plan is essential for minimizing the impact of successful attacks:

- **Incident Response Plan**: Document steps to take when an attack or anomaly is detected (e.g., isolate affected services, trigger deeper forensic analysis).
- **Containment Strategies**: If an ongoing attack is detected (e.g., data poisoning detected in training pipeline), halt training or deployment until issue is resolved.
- **Remediation Actions**: Procedures for retraining or patching the model, applying additional defenses, or updating preprocessing pipelines.
- **Post-Incident Review**: After an incident, analyze root causes, update threat model assumptions, and refine monitoring and defense strategies accordingly.
- **Stakeholder Communication**: Establish clear communication channels and responsibilities for notifying relevant teams (e.g., security, ML engineering, product) during incidents.

### 9.5 Integration into CI/CD Pipelines
Integrating security into the Continuous Integration/Continuous Deployment pipeline automates security verification and ensures continuous vigilance:

- **Automated Security Checks**: Incorporate automated runs of attack/defense simulations (Modules 2–5) into CI pipelines triggered by code or data changes.
- **Gatekeeping Deployments**: Block deployments if security evaluation metrics fall below predefined thresholds (e.g., risk score above threshold, defense evaluation final score below threshold).
- **Continuous Reporting**: Generate and archive periodic security reports; notify stakeholders of changes in security posture.
```

## 7. Integration with the Framework

Module 6, the Final Report Aggregation, serves as the culmination of the Safe-DL framework, bringing together all the insights and data generated by the preceding modules into a single, cohesive, and actionable document. Its integration is designed to be seamless, leveraging the standardized outputs of Modules 1 through 5.

### 7.1 Execution Flow

The `run_module6.py` script orchestrates the generation of the final report. The typical execution flow is as follows:

1.  **Profile Selection**: The script first prompts the user to select an existing `profile.yaml` file from the `profiles/` directory. This profile is expected to have been incrementally updated by Modules 1 through 5, containing all the necessary metadata, attack results, risk analysis, defense configurations, and defense evaluation scores.
2.  **Data Loading**: `run_module6.py` utilizes utility functions from `generate_report_utils.py` to load data from:
    * The selected `profile.yaml`.
    * Result JSON and Markdown files generated by Module 2 (`module2_attack_simulation/results/`).
    * Result JSON and Markdown files generated by Module 3 (`module3_risk_analysis/results/`).
    * Result JSON and Markdown files generated by Module 4 (`module4_defense_application/results/`).
    * Result JSON and Markdown files generated by Module 5 (`module5_defense_evaluation/results/`).
3.  **Section Generation**: The `generate_report_utils.py` contains dedicated functions for rendering each major section of the final report. These functions retrieve the relevant data, format it into Markdown tables, lists, and narrative text, and construct the full report content. The order of section generation follows the logical flow outlined in Section 5 of this document.
    * `generate_report_header()`
    * `generate_system_details_section()`
    * `generate_threat_profile_section()`
    * `generate_attack_simulation_section()`
    * `generate_risk_analysis_section()`
    * `generate_defense_application_section()`
    * `generate_defense_evaluation_section()`
    * `generate_conclusions_section()`
    * `generate_monitoring_section()`
4.  **Report Output**: The aggregated Markdown content is then written to a file named `final_report.md` within the `reports/` directory. While the current implementation primarily outputs to Markdown, future enhancements could include PDF conversion or other formats.

### 7.2 Dependency on Previous Modules

Module 6 is highly dependent on the successful execution and output generation of all preceding modules. Each module contributes specific data and reports that are crucial for the completeness and accuracy of the final document:

* **Module 1 (Threat Modeling)**: Provides the foundational `threat_profile` details, including system characteristics and identified threat categories, which are presented in Section 3 of the final report.
* **Module 2 (Attack Simulation)**: Contributes the raw attack metrics (e.g., ASR, adversarial accuracy) and links to detailed attack reports, forming Section 4.
* **Module 3 (Risk Analysis)**: Supplies the calculated risk scores, the risk matrix, and initial defense recommendations, which populate Section 5.
* **Module 4 (Defense Application)**: Adds the configurations and initial results of the applied defenses, along with links to their detailed reports, covered in Section 6.
* **Module 5 (Defense Evaluation)**: Generates the comprehensive defense evaluation scores (Mitigation, CAD, Cost, Final Score) and the top-performing defense recommendations, which are the core of Section 7.

Without the outputs from these preceding modules, Module 6 would not be able to generate a complete or meaningful final report. The `profile.yaml` acts as the central hub, continuously enriched with data, ensuring reproducibility and consistency across the entire framework.

### 7.3 `generate_report_utils.py`

The `generate_report_utils.py` file is a critical component of Module 6. It encapsulates all the logic required to parse the `profile.yaml` and the various module-specific result files, and then format this data into the human-readable Markdown structure of the final report. This separation of concerns keeps `run_module6.py` clean and focused on orchestration, while the `_utils.py` handles the complexities of content generation and formatting.

