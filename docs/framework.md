# Introduction to the Deep Neural Network Security Framework

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Module 1 — Threat Modeling](#2-module-1---threat-modeling)
  * [2.1 Goal of the Module](#21-goal-of-the-module)
  * [2.2 Module Structure](#22-module-structure)
    + [2.2.1 Attack Surfaces in the Deep Learning Lifecycle](#221-attack-surfaces-in-the-deep-learning-lifecycle)
    + [2.2.2 Types of Attackers and System Knowledge](#222-types-of-attackers-and-system-knowledge)
    + [2.2.3 Attacker Motivations](#223-attacker-motivations)
    + [2.2.4 General Taxonomy of Attacks](#224-general-taxonomy-of-attacks)
    + [2.2.5 Threat Questionnaire](#225-threat-questionnaire)
    + [2.2.6 Automatic Threat Suggestion Logic](#226-automatic-threat-suggestion-logic)
    + [2.2.7 Threat Profile Checklist](#227-threat-profile-checklist)
  * [2.3 Module Output](#23-module-output)
  * [2.4 Conclusion of Module 1](#24-conclusion-of-module-1)
- [3. Module 2 — Adversarial Attacks](#3-module-2---adversarial-attacks)
  * [3.1 Goal of the Module](#31-goal-of-the-module)
  * [3.2 Workflow Overview](#32-workflow-overview)
  * [3.3 Submodule 2.1 — Data Poisoning Attacks](#33-submodule-21---data-poisoning-attacks)
    + [3.3.1 Objective](#331-objective)
    + [3.3.2 Implemented Attack Types](#332-implemented-attack-types)
      - [3.3.2.1 Label Flipping Attack](#3321-label-flipping-attack)
      - [3.3.2.2 Clean Label Poisoning](#3322-clean-label-poisoning)
    + [3.3.3 Workflow of Attack Simulation](#333-workflow-of-attack-simulation)
    + [3.3.4 Metrics and Reporting](#334-metrics-and-reporting)
    + [3.3.5 YAML Configuration Integration](#335-yaml-configuration-integration)
    + [3.3.6 Tools and Implementation Notes](#336-tools-and-implementation-notes)
  * [3.4 Submodule 2.2 — Backdoor Attacks](#34-submodule-22---backdoor-attacks)
    + [3.4.1 Objective](#341-objective)
    + [3.4.2 Implemented Attack Types](#342-implemented-attack-types)
      - [3.4.2.1 Static Patch Attack](#3421-static-patch-attack)
      - [3.4.2.2 Adversarially Learned Trigger](#3422-adversarially-learned-trigger)
    + [3.4.3 Workflow of Attack Simulation](#343-workflow-of-attack-simulation)
    + [3.4.4 Metrics and Reporting](#344-metrics-and-reporting)
    + [3.4.5 YAML Configuration Integration](#345-yaml-configuration-integration)
    + [3.4.6 Implementation Notes](#346-implementation-notes)
    + [3.4.7 Ethical Considerations](#347-ethical-considerations)
  * [3.5 Submodule 2.3 — Evasion Attacks](#35-submodule-23---evasion-attacks)
    + [3.5.1 Objective](#351-objective)
    + [3.5.2 Implemented Attack Types](#352-implemented-attack-types)
      - [3.5.2.1 White-box Attacks](#3521-white-box-attacks)
      - [3.5.2.2 Black-box Attacks](#3522-black-box-attacks)
    + [3.5.3 Workflow of Attack Simulation](#353-workflow-of-attack-simulation)
    + [3.5.4 Metrics and Reporting](#354-metrics-and-reporting)
    + [3.5.5 YAML Configuration Integration](#355-yaml-configuration-integration)
    + [3.5.6 Implementation Notes](#356-implementation-notes)
    + [3.5.7 Ethical Considerations](#357-ethical-considerations)
  * [3.6 Submodule 2.4 — Model & Data Stealing Attacks *(Future Work)*](#36-submodule-24---model---data-stealing-attacks---future-work--)
    + [3.6.1 Objective](#361-objective)
    + [3.6.2 Planned Attack Mechanisms](#362-planned-attack-mechanisms)
    + [3.6.3 Future Implementation Goals](#363-future-implementation-goals)
    + [3.6.4 Relevance and Importance](#364-relevance-and-importance)
  * [3.7 Conclusion of Module 2](#37-conclusion-of-module-2)
- [4. Module 3 — Risk Analysis](#4-module-3---risk-analysis)
  * [4.1 Goal of the Module](#41-goal-of-the-module)
  * [4.2 Risk Modeling Approach](#42-risk-modeling-approach)
  * [4.3 Quantitative Metrics and Calculation Details](#43-quantitative-metrics-and-calculation-details)
  * [4.4 Workflow of Risk Assessment](#44-workflow-of-risk-assessment)
  * [4.5 Outputs and Reporting](#45-outputs-and-reporting)
  * [4.6 YAML Profile Integration](#46-yaml-profile-integration)
  * [4.7 Implementation Notes](#47-implementation-notes)
  * [4.8 Technical Considerations](#48-technical-considerations)
  * [4.9 Limitations and Future Work](#49-limitations-and-future-work)
  * [4.10 Conclusion of Module 3](#410-conclusion-of-module-3)
- [5. Module 4 — Defense Application](#5-module-4---defense-application)
  * [5.1 Goal of the Module](#51-goal-of-the-module)
  * [5.2 Defense Selection Workflow](#52-defense-selection-workflow)
    + [5.2.1 Interactive Setup Interface](#521-interactive-setup-interface)
    + [5.2.2 Unified Configuration Storage](#522-unified-configuration-storage)
    + [5.2.3 Extensibility and Modularity](#523-extensibility-and-modularity)
  * [5.3 Implemented Defenses](#53-implemented-defenses)
    + [5.3.1 Data Poisoning Defenses](#531-data-poisoning-defenses)
      - [5.3.1.1 Data Cleaning](#5311-data-cleaning)
      - [5.3.1.2 Per-Class Monitoring](#5312-per-class-monitoring)
      - [5.3.1.3 Robust Loss Functions](#5313-robust-loss-functions)
      - [5.3.1.4 Differentially Private Training](#5314-differentially-private-training)
      - [5.3.1.5 Provenance Tracking](#5315-provenance-tracking)
      - [5.3.1.6 Influence Functions](#5316-influence-functions)
    + [5.3.2 Backdoor Defenses](#532-backdoor-defenses)
      - [5.3.2.1 Activation Clustering](#5321-activation-clustering)
      - [5.3.2.2 Spectral Signatures](#5322-spectral-signatures)
      - [5.3.2.3 Anomaly Detection](#5323-anomaly-detection)
      - [5.3.2.4 Neuron Pruning](#5324-neuron-pruning)
      - [5.3.2.5 Fine-Pruning](#5325-fine-pruning)
      - [5.3.2.6 Model Inspection](#5326-model-inspection)
    + [5.3.3 Evasion Defenses](#533-evasion-defenses)
      - [5.3.3.1 Adversarial Training](#5331-adversarial-training)
      - [5.3.3.2 Randomized Smoothing](#5332-randomized-smoothing)
      - [5.3.3.3 Gradient Masking](#5333-gradient-masking)
      - [5.3.3.4 JPEG Preprocessing](#5334-jpeg-preprocessing)
  * [5.4 Outputs and Reporting](#54-outputs-and-reporting)
    + [5.4.1 JSON Reports](#541-json-reports)
    + [5.4.2 Markdown Reports](#542-markdown-reports)
    + [5.4.3 Visual Logs](#543-visual-logs)
    + [5.4.4 Reproducibility and Traceability](#544-reproducibility-and-traceability)
  * [5.5 Technical Considerations and Future Work](#55-technical-considerations-and-future-work)
    + [5.5.1 Current Limitations](#551-current-limitations)
    + [5.5.2 Computational Costs](#552-computational-costs)
    + [5.5.3 Planned Improvements](#553-planned-improvements)
  * [5.6 Conclusion of Module 4](#56-conclusion-of-module-4)
- [6. Module 5 — Evaluation & Benchmarking](#6-module-5---evaluation---benchmarking)
  * [6.1 Goal of the Module](#61-goal-of-the-module)
  * [6.2 Analysis Workflow](#62-analysis-workflow)
  * [6.3 Output and Reporting](#63-output-and-reporting)
  * [6.4 Technical Considerations](#64-technical-considerations)
  * [6.5 Future Enhancements](#65-future-enhancements)
  * [6.6 Conclusion of Module 5](#66-conclusion-of-module-5)
- [7. Module 6 — Final Report Aggregation](#7-module-6---final-report-aggregation)
  * [7.1 Goal of the Module](#71-goal-of-the-module)
  * [7.2 Aggregation Workflow](#72-aggregation-workflow)
  * [7.3 Input Sources](#73-input-sources)
  * [7.4 Report Contents](#74-report-contents)
  * [7.5 Example Output Snippet](#75-example-output-snippet)
  * [7.6 Technical Considerations](#76-technical-considerations)
  * [7.7 Future Enhancements](#77-future-enhancements)
  * [7.8 Conclusion of Module 6](#78-conclusion-of-module-6)
- [8. Example Application of the Safe-DL Framework (End-to-End Walk-through)](#8-example-application-of-the-safe-dl-framework--end-to-end-walk-through-)
  * [8.1 Scenario](#81-scenario)
  * [8.2 Module 1 — Threat Modeling](#82-module-1---threat-modeling)
  * [8.3 Module 2 — Attack Simulation](#83-module-2---attack-simulation)
  * [8.4 Module 3 — Risk Analysis](#84-module-3---risk-analysis)
  * [8.5 Module 4 — Defense Application](#85-module-4---defense-application)
  * [8.6 Module 5 — Evaluation & Benchmarking](#86-module-5---evaluation---benchmarking)
  * [8.7 Module 6 — Final Report Aggregation](#87-module-6---final-report-aggregation)
  * [8.8 Final Outcome](#88-final-outcome)


## 1. Introduction 
In recent years, deep neural networks (DNNs) have become the backbone of many modern applications, ranging from autonomous systems and biometrics to medical diagnostics and cybersecurity. However, with their increasing adoption, there has also been a growing concern regarding their vulnerability to adversarial attacks and other forms of malicious exploitation.

These attacks can occur at various stages of the model lifecycle — from training to inference — and may compromise the integrity, confidentiality, and availability of the systems in which they are embedded.

This framework provides a systematic and modular approach to securing deep learning models. It serves as a comprehensive and practical guide for researchers and engineers alike. Throughout the document, I propose a clear division into modules, each with actionable tools and recommendations. This allows anyone to develop, evaluate, and deploy models with robustness and security from the ground up.

The framework can be used both for theoretical analysis and for practical integration into real-world machine learning pipelines.

## 2. Module 1 — Threat Modeling

### 2.1 Goal of the Module

Before applying any defense technique, it is essential to understand the most likely threats in the specific context of your project. This module provides a structured approach to identify risks, attack surfaces, and attacker goals, enabling informed defensive strategies in the following modules.

### 2.2 Module Structure

#### 2.2.1 Attack Surfaces in the Deep Learning Lifecycle

| Phase               | Possible Threats                                     |
|--------------------|-------------------------------------------------------|
| Data acquisition    | Data poisoning, backdoors, sensor manipulation        |
| Training            | Malicious data injection, weight extraction          |
| Final model         | Model stealing, inversion, membership inference      |
| Inference/Deployment | Adversarial examples, evasion, physical attacks      |
| API Services        | Query-based extraction, denial of service attacks    |

>See [Peng et al., 2024; Liu et al., 2021; Li et al., 2021] for examples and references to these threats.

#### 2.2.2 Types of Attackers and System Knowledge

- **White-box attacker**: full access to architecture, parameters, and data.
- **Black-box attacker**: access only to model outputs (e.g., via API).
- **Gray-box**: partial knowledge (e.g., architecture but not weights).
- **Offline vs Online**: attacks during training (e.g., poisoning) or during inference (e.g., adversarial examples).

#### 2.2.3 Attacker Motivations

- Sabotage model performance
- Replace model with a compromised version
- Steal private information from training data
- Extract model’s intellectual property
- Fool critical systems (e.g., biometrics, autonomous vehicles)

#### 2.2.4 General Taxonomy of Attacks

| Main Category        | Subcategories                                      |
|----------------------|----------------------------------------------------|
| Poisoning (training) | Label flipping, clean-label, targeted poisoning   |
| Backdoor (training)  | Trigger-based, federated backdoors                |
| Adversarial (inference) | White-box, Black-box, Physical, Universal     |
| Privacy/Stealing     | Model inversion, extraction, membership inference |

#### 2.2.5 Threat Questionnaire

To operationalize threat modeling, the framework provides an interactive command-line questionnaire. Users are guided step-by-step in defining:

- Model access level (white-box, gray-box, black-box),
- Attack goals (targeted, untargeted),
- Deployment scenario (cloud, edge, mobile, API public, on-device),
- Data sensitivity level (high, medium, low),
- Training data source (internal, external, user-generated),
- Exposed interface (API, SDK, local_app, none),
- Relevant threat categories.

This questionnaire outputs a structured YAML file (`threat_profile.yaml`) encoding all user inputs. The generated profile acts as a single source of truth for configuring subsequent modules.

#### 2.2.6 Automatic Threat Suggestion Logic

To assist users in creating a coherent threat profile, the framework implements an automatic suggestion mechanism based on simple, rule-based logic. For instance:

- **External or user-generated data** suggests vulnerability to `data_poisoning`.
- **Public APIs** expose risks of `model_stealing` and `membership_inference`.
- **Mobile or edge deployments** often imply susceptibility to `adversarial_examples`.
- **High sensitivity data** increases the risk of `model_inversion`.

These automated suggestions help users quickly identify realistic and applicable threats, ensuring comprehensive coverage.


#### 2.2.7 Threat Profile Checklist

- [ ] What is the level of attacker access? (white-box, gray-box, black-box)
- [ ] Is the attack goal targeted or untargeted?
- [ ] Where will the model be deployed? (cloud, edge, mobile, API public, on-device)
- [ ] How sensitive is the training data? (high, medium, low)
- [ ] What is the source of the training data? (internal, external, user-generated)
- [ ] Is the model exposed through an interface? (API, SDK, local_app, none)
- [ ] Which threat categories apply to your scenario? (data_poisoning, adversarial_examples, backdoor_attacks, model_stealing, membership_inference, model_inversion)


### 2.3 Module Output

After completing the threat profile, the user receives a configuration file that can be reused in the following modules.

```yaml
threat_model:
  model_access: white-box
  attack_goal: targeted
  deployment_scenario: cloud
  data_sensitivity: high
  training_data_source: internal_clean
  model_type: cnn
  interface_exposed: api
  threat_categories:
    - data_poisoning
    - adversarial_examples

```



### 2.4 Conclusion of Module 1

This module provides the essential foundation for robust security strategies. Through the structured threat questionnaire and automatic threat suggestions, users gain a clear understanding of relevant risks and attacker goals. The resulting `threat_profile.yaml` serves as a configurable blueprint for the attack simulations and defensive strategies that follow, ensuring informed and effective decision-making throughout the framework.


## 3. Module 2 — Adversarial Attacks

### 3.1 Goal of the Module

Despite the success of deep neural networks in various tasks, they remain highly vulnerable to a wide range of attacks that exploit flaws in their statistical behavior, architectures, or training processes. These attacks — whether during training or inference — can severely compromise the reliability, security, and privacy of deep learning systems.

This module aims to:

- Understand the most relevant attack mechanisms based on recent literature.
- Simulate realistic attacks on custom models to assess vulnerabilities.
- Map previously identified risks (from Module 1) to corresponding attack techniques.
- Prepare the application of targeted defenses in subsequent modules.

### 3.2 Workflow Overview
Module 2 uses the threat profile generated in Module 1 to guide attack simulations. The structured workflow consists of:

1. Dataset and model selection (built-in or custom).
2. Baseline model training (clean data evaluation).
3. Execution of specific adversarial attacks defined in the threat profile.
4. Retraining the model on poisoned data and evaluating impact.
5. Automatic generation of detailed reports (.json, .md).

### 3.3 Submodule 2.1 — Data Poisoning Attacks


#### 3.3.1 Objective
Data poisoning attacks involve injecting malicious samples into the training dataset to compromise the integrity of the model. They are subtle, difficult to detect, and can cause long-lasting damage.

#### 3.3.2 Implemented Attack Types

##### 3.3.2.1 Label Flipping Attack
- The attacker swaps correct labels intentionally, causing the model to misclassify specific classes.
- Two implemented strategies:
  - **One-to-one**: flips labels from a specific source class to one specific target class.
  - **Many-to-one**: flips labels from multiple classes into one target class.
- Configurable parameters include:
  - `flip_rate`: fraction of labels to flip.
  - `source_class`: the original class labels (optional).
  - `target_class`: the new, incorrect class label.

##### 3.3.2.2 Clean Label Poisoning
- Poisoned samples retain correct labels but contain subtle feature manipulations that mislead the model.
- Particularly stealthy and hard to detect.

#### 3.3.3 Workflow of Attack Simulation
The submodule follows these structured steps:

1. Load the selected dataset (built-in or user-defined).
2. Train a baseline model on clean training data and evaluate accuracy.
3. Perform the data poisoning attack based on user-defined parameters from the YAML threat profile.
4. Retrain the model on the poisoned dataset.
5. Evaluate and record the impact of the attack by comparing against baseline results.

#### 3.3.4 Metrics and Reporting
For each attack simulation, the submodule generates:

- **Detailed JSON reports** capturing:
  - Global accuracy reduction.
  - Per-class accuracy metrics.
  - Specific indices and classes of flipped or poisoned samples.

- **Markdown reports (.md)** summarizing:
  - Attack configuration parameters.
  - Summary tables of affected classes.
  - Visual examples clearly showing poisoned versus original samples.

#### 3.3.5 YAML Configuration Integration
Attack parameters are stored in a centralized YAML configuration file under the `attack_overrides` section, ensuring reproducibility and ease of experiment management.

Example YAML snippet:

```yaml
attack_overrides:
  data_poisoning:
    label_flipping:
      strategy: many_to_one
      flip_rate: 0.1
      source_class: null
      target_class: 2
```

#### 3.3.6 Tools and Implementation Notes

-   All simulations rely on custom-built scripts provided in the Safe-DL framework.
    
-   Built-in compatibility for popular datasets (MNIST, CIFAR-10, etc.) and neural network architectures (CNN, ResNet, ViT).
    
-   Supports custom user-defined datasets and models through standardized Python interfaces.


### 3.4 Submodule 2.2 — Backdoor Attacks

#### 3.4.1 Objective
Backdoor attacks embed hidden behaviors in deep neural networks, activated only when specific trigger patterns appear at inference time. The model behaves normally on clean inputs but misclassifies triggered inputs to a predefined target class.

#### 3.4.2 Implemented Attack Types

##### 3.4.2.1 Static Patch Attack
- A fixed visual trigger (patch) is applied consistently to selected training samples.
- Implemented in two modes:
  - **Clean-label mode**: Triggered samples maintain their original labels (more stealthy).
  - **Corrupted-label mode**: Triggered samples explicitly have their labels set to a target class.
- **Configurable parameters**:
  - `position`: location of the trigger (top-left, bottom-right, etc.).
  - `blend_alpha`: blending factor to make the trigger subtle.
  - `target_class`: class label activated by the trigger.

##### 3.4.2.2 Adversarially Learned Trigger
- Trigger patterns are learned adversarially during model training to maximize stealth and effectiveness.
- Implemented exclusively in **corrupted-label mode**.
- **Configurable parameters**:
  - `target_class`: the class to misclassify triggered samples into.
  - `poison_rate`: fraction of training samples poisoned.
  - Training-specific parameters (epochs, learning rate, etc.).

#### 3.4.3 Workflow of Attack Simulation
The structured workflow for backdoor attacks involves:

1. Selection and configuration of dataset and model (built-in or custom).
2. Baseline model training and clean dataset evaluation.
3. Interactive configuration via CLI questionnaire, stored in YAML (`attack_overrides`).
4. Generation of a poisoned training dataset (static or learned triggers).
5. Retraining the model on the poisoned dataset.
6. Model evaluation on both clean and triggered datasets to measure attack effectiveness.

#### 3.4.4 Metrics and Reporting
The module generates detailed and structured outputs:

- **JSON Reports** capturing:
  - `accuracy_clean`: accuracy on the clean test set.
  - `accuracy_triggered`: accuracy on a triggered test set.
  - `attack_success_rate` (ASR): proportion of triggered samples classified into the target class.
  - Parameters and per-class accuracy statistics.

- **Markdown Reports**:
  - Summary of results in table form.
  - Visual examples clearly demonstrating applied triggers.

Example JSON snippet:

```json
{
  "attack": "static_patch",
  "label_mode": "corrupted",
  "accuracy_clean": 0.91,
  "accuracy_triggered": 0.15,
  "attack_success_rate": 0.85,
  "trigger_params": {
    "position": "bottom-right",
    "blend_alpha": 0.2
  }
}
```

#### 3.4.5 YAML Configuration Integration

Attack-specific parameters are defined in a centralized YAML configuration, ensuring reproducibility.

Example YAML configuration snippet:

```yaml
attack_overrides:
  backdoor:
    static_patch:
      label_mode: corrupted
      position: bottom-right
      blend_alpha: 0.2
      target_class: 5

```
#### 3.4.6 Implementation Notes

-   All attacks are fully modular, enabling easy extension and customization.
    
-   Built-in support for common datasets (MNIST, CIFAR) and models (CNN, ResNet, ViT), with simple interfaces for custom options.
    
-   Automatic and structured reporting greatly simplifies vulnerability assessments and defense evaluations.
    

#### 3.4.7 Ethical Considerations

These attack simulations are intended solely for the purpose of assessing vulnerabilities and validating defense strategies. Misuse of such techniques is strictly against the intended purpose of the Safe-DL framework.

### 3.5 Submodule 2.3 — Evasion Attacks

#### 3.5.1 Objective
Evasion attacks target deep neural networks at inference time by generating minimally perturbed inputs (adversarial examples) to cause misclassifications. These perturbations are typically imperceptible to human observers yet highly effective in compromising model predictions.

#### 3.5.2 Implemented Attack Types

##### 3.5.2.1 White-box Attacks

- **FGSM (Fast Gradient Sign Method)**  
  Simple and efficient gradient-based attack.  
  Parameters:  
  - `epsilon` (maximum perturbation).

- **PGD (Projected Gradient Descent)**  
  Iterative gradient-based attack, more powerful than FGSM.  
  Parameters:  
  - `epsilon`, `num_steps`, `step_size`.

- **Carlini & Wagner (C&W)**  
  Highly effective optimization-based attack.  
  Parameters:  
  - Confidence, learning rate, iterations.

- **DeepFool**  
  Finds minimal perturbations to cross decision boundaries.  
  Parameters:  
  - Max iterations, overshoot factor.

##### 3.5.2.2 Black-box Attacks

- **NES (Natural Evolution Strategies)**  
  Gradient-free evolutionary strategy for attack optimization.  
  Parameters:  
  - Population size, perturbation limit.

- **SPSA (Simultaneous Perturbation Stochastic Approximation)**  
  Black-box optimization without true gradients.  
  Parameters:  
  - Learning rate, iterations.

- **Boundary Attack**  
  Decision-based iterative approach exploiting model outputs.  
  Parameters:  
  - Steps, step size.

- **Transfer-based Attacks**  
  Generates adversarial examples from substitute models.  
  Parameters:  
  - Substitute model choice, epsilon.

#### 3.5.3 Workflow of Attack Simulation
This submodule systematically applies evasion attacks via the following steps:

1. Selection and configuration of dataset and model (built-in/custom).
2. Train baseline (clean) model and evaluate its accuracy.
3. Configure specific evasion attacks via interactive CLI questionnaire and YAML (`attack_overrides`).
4. Generate adversarial examples using the selected attack method.
5. Evaluate model performance on adversarial examples.
6. Generate comprehensive evaluation reports automatically.

#### 3.5.4 Metrics and Reporting
Detailed evaluation reports are produced automatically:

- **JSON Reports** capturing:
  - Clean and adversarial accuracies (`accuracy_clean`, `accuracy_adversarial`).
  - Attack-specific parameters (`epsilon`, `num_steps`, etc.).
  - Per-class accuracy statistics (clean and adversarial).

- **Markdown Reports**:
  - Summary of metrics and attack configurations.
  - Visual comparison between original and adversarial samples.

Example JSON snippet:

```json
{
  "attack": "pgd",
  "accuracy_clean": 0.93,
  "accuracy_adversarial": 0.27,
  "epsilon": 0.03,
  "num_steps": 40,
  "per_class_accuracy_clean": {...},
  "per_class_accuracy_adversarial": {...}
}
```
#### 3.5.5 YAML Configuration Integration

Attack parameters are centralized in YAML, allowing consistent and reproducible simulations.

Example YAML snippet:

```yaml
attack_overrides:
  evasion:
    pgd:
      epsilon: 0.03
      num_steps: 40
      step_size: 0.007

```
#### 3.5.6 Implementation Notes

-   All attacks are implemented without external wrappers, offering detailed control and transparency.
    
-   Supports common datasets and architectures (CNN, ResNet, ViT), facilitating rapid evaluation of vulnerabilities.
    
-   Modular design simplifies extension and customization to other evasion methods.
    

#### 3.5.7 Ethical Considerations

These evasion attack simulations are exclusively for security assessment and defense validation. Their misuse outside controlled environments violates the intended ethical use of this framework.



### 3.6 Submodule 2.4 — Model & Data Stealing Attacks *(Future Work)*

#### 3.6.1 Objective
Model and data stealing attacks aim to compromise the privacy of training data or extract intellectual property embedded within deep neural network models. These attacks typically involve querying a deployed model (e.g., via public API) to reconstruct sensitive training data (model inversion) or replicate model functionality (model extraction).

This submodule is conceptually included in the Safe-DL framework but has not yet been implemented. It is planned as part of future developments to further expand the security evaluation capabilities of the framework.

#### 3.6.2 Planned Attack Mechanisms
- **Model Extraction**: Replicating a model's functionality through extensive queries, creating unauthorized surrogate models.
- **Membership Inference**: Determining whether specific samples were part of the training set, raising data privacy concerns.
- **Model Inversion**: Reconstructing sensitive input data based on model outputs.

#### 3.6.3 Future Implementation Goals
- Provide a simulation environment to evaluate vulnerability against these privacy-centric attacks.
- Measure surrogate model accuracy, membership inference effectiveness, and model inversion risks.
- Recommend appropriate countermeasures, such as differential privacy, regularization, and output obfuscation.

#### 3.6.4 Relevance and Importance
In real-world deployments, particularly those involving sensitive or proprietary data, addressing these types of attacks is crucial. Explicitly acknowledging these attack vectors within the Safe-DL framework underscores its comprehensive approach to deep learning security.

*(Detailed implementation, metrics, and reporting will be described once this submodule is fully integrated.)*



### 3.7 Conclusion of Module 2

With the completion of this module, users have conducted a comprehensive adversarial vulnerability assessment covering key areas:

- **Data Poisoning Attacks**, including Label Flipping and Clean-label poisoning.
- **Backdoor Attacks**, implementing Static Patch (clean-label and corrupted-label modes) and adversarially learned triggers.
- **Evasion Attacks**, comprising various White-box (FGSM, PGD, C&W, DeepFool) and Black-box (NES, SPSA, Boundary, Transfer-based) methods.

Users have now clearly identified vulnerabilities specific to their model and scenario, leveraging structured simulations, detailed metrics, and automated reporting. The generated results (`JSON` and `Markdown`) provide actionable insights that form the basis for informed, targeted defensive strategies in subsequent modules.

Additionally, the framework acknowledges **Model & Data Stealing Attacks** as a critical area for future development, further emphasizing its comprehensive approach to securing deep neural networks.



## 4. Module 3 — Risk Analysis

### 4.1 Goal of the Module

After executing attack simulations in Module 2, Module 3 provides a structured approach to transform raw attack results into interpretable security risk assessments. This step helps users:

- Quantify the severity of each attack based on concrete metrics.
- Estimate the likelihood (probability) and detectability (visibility) of each attack.
- Prioritize defense mechanisms effectively for Module 4.
- Automatically generate detailed risk assessment reports for inclusion in documentation.

---

### 4.2 Risk Modeling Approach

Risk assessment in Safe-DL is structured around three core dimensions:

- **Severity**: The impact of the attack on model performance, typically measured by accuracy degradation or Attack Success Rate (ASR).
- **Probability**: The estimated likelihood of successful exploitation under the specified threat model.
- **Visibility**: How detectable or stealthy the attack is, considering perturbation magnitudes or data manipulation intensity.

The overall risk score is computed as follows:

 ```text
 risk_score = severity × probability × [1 + (1 - visibility)]
 ```

This calculation prioritizes highly impactful, probable, and stealthy attacks, enabling targeted and informed defense decisions.

---

### 4.3 Quantitative Metrics and Calculation Details

The module computes and normalizes the following metrics for each attack:

- **Severity**:  
  Primarily based on accuracy drop or ASR compared to baseline performance:
  
 ```text
severity = min((accuracy_clean - accuracy_attack) / 0.3, 1.0)
 ```

- **Probability**:  
Derived from attacker knowledge and difficulty of execution:
- `White-box`: probability = 1.0
- `Black-box`: probability = 0.8 (default heuristic)
*(values adjustable based on threat profile)*

- **Visibility**:  
Estimated based on perturbation size, poison rate, or trigger blending factor (`blend_alpha`). Lower visibility means higher stealth.

---

### 4.4 Workflow of Risk Assessment

The structured analysis workflow involves the following steps:

1. **Load Threat Profile**  
 Automatically reads `profile.yaml` containing attack details and threat categories from Module 1 and Module 2.

2. **Import Attack Metrics**  
 Retrieves results from attack simulations (`*_metrics.json`) such as:
 - Clean vs adversarial accuracy.
 - Attack-specific details (flip rate, ASR, perturbation magnitude).

3. **Compute Risk Metrics**  
 Calculates severity, probability, and visibility metrics for each attack.

4. **Generate Risk Scores**  
 Computes final risk scores for prioritization purposes.

5. **Report Generation**  
 Automatically creates structured outputs (JSON and Markdown) summarizing the risk assessments.

---

### 4.5 Outputs and Reporting

Module 3 automatically generates comprehensive reports in two formats:

- **risk_analysis.json**  
Structured JSON file summarizing all computed risk metrics:
```json
{
  "fgsm": {
    "severity": 0.95,
    "probability": 1.0,
    "visibility": 0.2,
    "risk_score": 1.71
  },
  "label_flipping": {
    "severity": 0.75,
    "probability": 0.8,
    "visibility": 0.5,
    "risk_score": 0.9
  }
}
```
-   **risk_report.md**  
    Human-readable Markdown document containing:
    
    -   A risk matrix visualizing severity vs. probability.
        
    -   Rankings and detailed tables comparing attacks.
        
    -   Clear explanations of each attack’s risk components.
        
    -   Recommended defensive measures for each identified vulnerability.
        

Example Markdown snippet:

```markdown
## Risk Analysis Summary

| Attack Type   | Severity | Probability | Visibility | Risk Score |
|---------------|----------|-------------|------------|------------|
| FGSM          | 0.95     | 1.0         | 0.2        | 1.71       |
| Label Flipping| 0.75     | 0.8         | 0.5        | 0.9        |

### Risk Matrix

| Probability \ Severity | Low (0–0.3) | Medium (0.3–0.7) | High (0.7–1) |
|------------------------|-------------|------------------|--------------|
| High (0.8–1.0)         |             |                  | FGSM         |
| Medium (0.4–0.8)       |             | Label Flipping   |              |
| Low (0–0.4)            |             |                  |              |

### Recommendations
- **FGSM**: Highly severe and likely. Recommended defense: Adversarial Training (PGD), Input Preprocessing.
- **Label Flipping**: Moderate severity and visibility. Recommended defense: Robust Loss Functions, Data Cleaning.

```

Additionally, the risk metrics and recommendations are automatically integrated into the existing `profile.yaml` for seamless integration with Module 4.

----------

### 4.6 YAML Profile Integration

Risk assessment results update the user’s threat profile YAML (`profile.yaml`) with structured summaries:

```yaml
risk_analysis:
  summary:
    fgsm:
      severity: 0.95
      probability: 1.0
      visibility: 0.2
      risk_score: 1.71
  recommendations:
    fgsm:
      - Adversarial Training (PGD)
      - Input Preprocessing
```

----------

### 4.7 Implementation Notes

-   All scripts are modular and designed for easy extension to additional attack types or customized risk assessment metrics.
    
-   Automated, reproducible workflows ensure consistent evaluations and comparisons across different experiments and scenarios.
    
-   Structured integration of risk metrics within the YAML profiles simplifies downstream selection of defenses.
    

----------

### 4.8 Technical Considerations

-   Module 3 does not directly implement defenses; instead, it serves as a structured vulnerability triage mechanism, informing the targeted defense strategies implemented in Module 4.
    
-   Future extensions may incorporate dynamic weighting of risk metrics or integration with defense effectiveness feedback loops.
    

----------

### 4.9 Limitations and Future Work

Current implementation constraints and planned improvements include:

-   **Fixed Risk Weights**:  
    Currently heuristic; adaptive weighting based on historical data is planned.
    
-   **Attack-Type Coverage**:  
    Extend to additional attack categories (e.g., privacy attacks).
    
-   **Risk History Tracking**:  
    Introduce historical tracking of risk assessments for long-term monitoring.
    
-   **Defense Effectiveness Feedback**:  
    Integration with Module 4 to adjust risk assessments dynamically based on defense outcomes.
    
-   **Visualization and UI**:  
    Future dashboard implementation for interactive risk exploration and enhanced usability.
    

----------

### 4.10 Conclusion of Module 3

By systematically transforming raw attack simulation results into structured risk assessments, Module 3 provides crucial insights for informed and prioritized decision-making. This structured triage significantly enhances the practical utility of the Safe-DL framework, directly facilitating targeted, effective defenses applied in Module 4.


## 5. Module 4 — Defense Application

### 5.1 Goal of the Module

This module provides a systematic approach for applying targeted defense mechanisms based on the risk analysis conducted in Module 3.

The primary goals are:

- To mitigate the specific vulnerabilities identified through attack simulations.
- To apply defense strategies tailored to each attack type: data poisoning, backdoor, and evasion.
- To offer users a flexible and modular way to select and configure defenses through a command-line interface.
- To retrain and evaluate defended models, generating performance and robustness metrics.
- To document the effects of each defense via standardized JSON and Markdown reports.

Defenses are automatically suggested based on the `risk_analysis.recommendations` section in the user profile and can be customized interactively. All active defenses and their parameters are stored in a unified configuration block (`defense_config`) within the main YAML file. This ensures consistency and reproducibility throughout the framework.

The final output includes:
- A model retrained with applied defenses.
- Updated threat profile with defense metadata.
- Standardized evaluation metrics (clean vs. adversarial performance).
- Visual and structured reports suitable for inclusion in audits or documentation.

This module prepares the defended model for final robustness evaluation in Module 5.




### 5.2 Defense Selection Workflow

To ensure that defenses are both effective and relevant to the user’s specific threat model, the Safe-DL framework employs a guided and dynamic defense selection process.

#### 5.2.1 Interactive Setup Interface

Upon launching Module 4, users are presented with an interactive command-line interface that:

- Loads the existing `profile.yaml`, which includes the `risk_analysis.recommendations` block generated by Module 3.
- Displays the recommended defenses for each attack category based on prior risk scores.
- Allows users to **accept**, **skip**, or **customize** each recommendation.
- Enables users to override parameters such as pruning ratios, adversarial training ε values, or blending factors.

This setup step ensures that the selected defenses align with both the risk profile and the user's practical constraints (e.g., training time, deployment environment).

#### 5.2.2 Unified Configuration Storage

All selected defense strategies and their parameters are stored in the `defense_config` block of the main YAML profile, nested under their respective attack sub-categories. Example:

```yaml
defense_config:
  data_poisoning:
    label_flipping:
      defenses:
        - data_cleaning
        - robust_loss
      data_cleaning:
        method: loss_filtering
        threshold: 0.9
      robust_loss:
        type: gce
  backdoor:
    static_patch:
      defenses:
        - activation_clustering
        - fine_pruning
      activation_clustering:
        num_clusters: 2
  evasion:
    pgd:
      defenses:
        - adversarial_training
      adversarial_training:
        attack_type: pgd
        epsilon: 0.03
```
This structured configuration enables seamless re-execution, reproducibility, and downstream analysis.

#### 5.2.3 Extensibility and Modularity

The system is designed to support:

-   Multiple defense strategies per category.
    
-   Independent execution and evaluation of each defense.
    
-   Future integration of new techniques with minimal reconfiguration.
    

By decoupling configuration from execution, the framework empowers users to adapt defenses to their evolving security needs while maintaining clarity and traceability.

### 5.3 Implemented Defenses


#### 5.3.1 Data Poisoning Defenses

This category focuses on mitigating the effects of data poisoning attacks, particularly label flipping and clean-label variants. The framework supports several defense strategies that aim to detect and neutralize malicious training samples before they degrade model performance.

##### 5.3.1.1 Data Cleaning

This defense applies clustering or density-based outlier detection (e.g., k-NN or PCA) to identify and remove poisoned samples from the training set before model retraining.

- Automatically compares original and cleaned datasets.
- Generates a list of removed samples (with images and labels).
- Configurable parameters include:
  - Number of clusters or neighbors
  - Distance thresholds
- Reports include:
  - Number and percentage of removed samples
  - Clean and poisoned accuracy before/after
  - Per-class accuracy breakdown

##### 5.3.1.2 Per-Class Monitoring

This defense evaluates the distribution of predictions and loss across classes during training.

- Identifies anomalies in individual class behavior (e.g., abnormal error rates or loss divergence).
- Flags suspicious classes for inspection or exclusion.
- Produces a visual report highlighting:
  - Accuracy per class (clean vs poisoned)
  - Divergence scores or heuristics
- Works best for attacks targeting specific class pairs.

##### 5.3.1.3 Robust Loss Functions

This method replaces standard cross-entropy with loss functions that are more resilient to label noise and poisoned samples.

- Supported options include:
  - Symmetric Cross Entropy
  - Generalized Cross Entropy
- Parameters such as β or q are configurable in the YAML file.
- No need to remove data — the training itself becomes more robust.
- Final reports compare clean vs poisoned accuracy using each loss function.

##### 5.3.1.4 Differentially Private Training

This strategy adds noise to gradients during training to reduce sensitivity to individual data points.

- Uses a simplified DP-SGD approach (ε configurable).
- Protects against memorization of poisoned samples.
- YAML-configurable with parameters such as:
  - Noise multiplier
  - Clipping norm
- Reports include utility metrics and a basic ε-privacy bound (if desired).

##### 5.3.1.5 Provenance Tracking

This method tracks the origin and transformation history of each sample to flag suspicious input sources.

- Particularly useful in multi-source datasets (e.g., user uploads).
- Supports tagging and exclusion of data from untrusted origins.
- Visualization of origin statistics and performance impact.
- YAML allows defining trusted/untrusted source labels.

##### 5.3.1.6 Influence Functions

This defense uses influence functions to estimate which training points have the highest impact on predictions for certain test samples.

- Helps identify poisoned points that disproportionately influence decision boundaries.
- Computes influence scores per sample.
- Removes top-k influential outliers prior to retraining.
- Outputs:
  - Influence histograms
  - Accuracy before/after retraining
  - Visual samples with high influence values




#### 5.3.2 Backdoor Defenses

Backdoor attacks compromise models by embedding a hidden trigger during training that causes targeted misclassification. The Safe-DL framework supports several defensive techniques to detect and remove such hidden behaviors.

##### 5.3.2.1 Activation Clustering

This method analyzes the internal activations of the model (typically the penultimate layer) in response to samples from the target class.

- Uses unsupervised clustering (e.g., K-Means) to separate clean vs backdoored inputs.
- Assumes poisoned samples form a distinct cluster due to abnormal activations.
- Removed samples are logged, visualized, and retraining is performed.
- YAML configuration allows setting:
  - Number of clusters
  - Layer to extract activations from

##### 5.3.2.2 Spectral Signatures

Spectral analysis is applied to intermediate activations to detect backdoor-related outliers.

- Identifies directions in the feature space with unusually high singular values.
- Removes samples contributing most to these directions.
- No labels required — operates in unsupervised mode.
- Reports include:
  - Singular value spectrum
  - Number of removed samples
  - Accuracy before/after

##### 5.3.2.3 Anomaly Detection

Statistical anomaly detectors are applied to model activations to isolate suspicious data points.

- Currently supported methods:
  - Isolation Forest
  - Local Outlier Factor (LOF)
- Operates on the latent space of the model.
- Removes outliers and retrains the model.
- YAML allows selecting the algorithm and setting thresholds.

##### 5.3.2.4 Neuron Pruning

This technique removes neurons that are overly responsive to the backdoor trigger.

- Identifies neurons with large activation gaps between clean and triggered inputs.
- Prunes the most affected units from the model architecture.
- Retrains model if necessary to recover performance.
- Configurable pruning ratio in YAML (e.g., 20% of sensitive neurons).

##### 5.3.2.5 Fine-Pruning

A variation of pruning combined with fine-tuning:

- Prune backdoor-sensitive neurons,
- Then fine-tune the model on clean data to recover accuracy.
- Often more effective than pruning alone, especially for large networks.
- Outputs include:
  - Accuracy on clean and triggered inputs
  - Visual summary of pruning impact

##### 5.3.2.6 Model Inspection

This defense audits model behavior by visualizing activation patterns and confidence heatmaps.

- Helps identify unusually confident predictions for trigger patterns.
- No automatic removal — user-guided inspection.
- Produces:
  - Confidence maps
  - Activation overlays
  - Optional trigger visualization



#### 5.3.3 Evasion Defenses

Evasion attacks are performed at inference time by crafting adversarial examples that lead to misclassification. These perturbations are often imperceptible to humans but highly effective against vulnerable models. The Safe-DL framework implements the following practical defenses:

##### 5.3.3.1 Adversarial Training

Adversarial training strengthens the model by including adversarial examples during training.

- Supported methods: FGSM and PGD.
- Generates adversarial inputs on-the-fly during training.
- Improves robustness at the cost of longer training and possible clean accuracy drop.
- YAML-configurable parameters:
  - `attack_type` (fgsm or pgd)
  - `epsilon` (perturbation strength)
- Reports include:
  - Clean and adversarial accuracy
  - Per-class performance breakdown

##### 5.3.3.2 Randomized Smoothing

This defense applies Gaussian noise to inputs during both training and inference to smooth the decision boundary.

- Enhances robustness by making predictions more stable under perturbation.
- Based on certified defense concepts but implemented here in a practical form.
- Configurable parameter:
  - `sigma` (standard deviation of noise)
- Outputs:
  - Clean and adversarial accuracy
  - Robustness curves vs. noise

##### 5.3.3.3 Gradient Masking

This method obfuscates the loss gradient to make it harder for attackers to compute adversarial examples.

- Trains the model with random noise injections or non-differentiable components.
- Focused on black-box transferability reduction.
- YAML options include:
  - Enable/disable flag
  - Type of masking (e.g., input noise)
- Reports include:
  - Drop in attack success rate
  - Robustness under black-box settings

##### 5.3.3.4 JPEG Preprocessing

Applies JPEG compression as a preprocessing step to remove small, high-frequency adversarial noise.

- Particularly effective against black-box attacks like NES and SPSA.
- Can be applied before or after model training.
- Configurable quality factor (e.g., 50%, 75%, 90%) in the YAML file.
- Outputs:
  - Clean and adversarial accuracy
  - Comparison before/after compression


### 5.4 Outputs and Reporting

After applying each defense, the framework generates standardized and structured outputs to facilitate evaluation, reproducibility, and comparison. These outputs are typically provided in both JSON and Markdown formats.

#### 5.4.1 JSON Reports

Each defense generates a `.json` file (e.g., `[defense_name]_results.json`) containing key metrics tailored to the attack type it mitigates. Common and specific fields include:

-   `accuracy_clean`: Accuracy on clean (unperturbed) test data after applying the defense.
-   `per_class_accuracy_clean`: Class-wise accuracy on clean inputs after defense.

**Specific metrics based on attack type:**

-   **For Evasion attacks (e.g., PGD, FGSM):**
    -   `accuracy_adversarial`: Accuracy on adversarial/test-time perturbed data after defense.
    -   `per_class_accuracy_adversarial`: Class-wise accuracy on adversarial inputs after defense.
-   **For Backdoor attacks (e.g., Static Patch, Learned Trigger):**
    -   `asr_after_defense`: Attack Success Rate (ASR) on backdoor-infected samples after defense. (A lower ASR indicates better mitigation).
-   **For Data Poisoning attacks (e.g., Label Flipping, Clean Label):**
    -   Metrics related to sample removal or anomaly detection, such as `num_removed` (number of removed samples) or identified anomalous classes.

**Additional fields may include:**

-   `removed_indices` or `example_removed`: Lists or examples of filtered/flagged training inputs (when applicable, e.g., for data cleaning or activation clustering defenses).
-   `parameters`: Configuration and hyperparameters used during defense execution.

These JSON reports are designed to be easily parsed and used by downstream modules (e.g., Module 5 — Defense Evaluation).

#### 5.4.2 Markdown Reports

Complementing the JSON outputs, a human-readable Markdown report (e.g., `[defense_name]_report.md`) is generated for each defense. These reports provide:

-   A concise summary of the defense's performance, including key accuracy metrics and ASR.
-   Visualizations (where applicable), such as pruning histograms or galleries of removed samples.
-   Detailed breakdowns of per-class metrics.
-   The specific parameters and configuration used for the defense's execution.

These Markdown reports offer a clear and shareable overview of the defense's impact, making it easier for users to understand the results without needing to parse the raw JSON data.

#### 5.4.3 Visual Logs

Whenever relevant (e.g., in data cleaning or activation clustering), the framework saves:

- Original vs filtered sample visualizations.
- Highlighted samples (e.g., highly influential or anomalous points).
- Examples of adversarial correction or backdoor trigger removal.

All outputs are saved in:

```text
results/module4_defense_application/<attack_type>/<defense_name>/
```


with consistent naming to allow downstream integration.

#### 5.4.4 Reproducibility and Traceability

- Each defense run is reproducible via the stored YAML configuration under `defense_config`.
- Results are tagged with timestamps and saved independently to avoid overwriting.
- This ensures that results from different runs or defense strategies can be easily compared later.



### 5.5 Technical Considerations and Future Work

#### 5.5.1 Current Limitations

While the implemented defenses provide solid baseline protection, the current version of the framework has some limitations:

- **Performance trade-offs**: Techniques like adversarial training and randomized smoothing increase training time and may reduce clean accuracy.
- **Static configuration**: Defense parameters are currently configured manually or through rule-based suggestions. There is no automatic hyperparameter tuning.
- **Dataset dependency**: Some defenses rely on assumptions about the dataset (e.g., label distribution or data quality), which may not generalize across domains.
- **Single-model scope**: All evaluations are currently performed on a single architecture at a time — multi-architecture or ensemble analysis is not yet supported.

#### 5.5.2 Computational Costs

- Adversarial training and fine-pruning are computationally intensive and may require GPUs with sufficient memory.
- Clustering-based methods (e.g., activation clustering) scale poorly with large datasets or deep architectures.
- Differential privacy introduces significant noise that can degrade utility if not carefully tuned.

To mitigate these issues, the framework supports:
- Modular execution (one defense at a time),
- Intermediate caching of model checkpoints and filtered datasets,
- Selective evaluation (e.g., subset of test data for adversarial inference).

#### 5.5.3 Planned Improvements

To improve flexibility and coverage, the following features are considered for future versions:

- **New defense techniques**: 
  - Certified defenses (e.g., randomized smoothing with formal bounds),
  - Defensive distillation,
  - Online poisoning detection.
  
- **Dynamic defense adaptation**: 
  - Auto-selection and tuning of defenses based on attack feedback,
  - Meta-learning strategies to optimize the defense stack per dataset.

- **Dashboard integration**:
  - Interactive visualization of attacks and defenses,
  - Real-time monitoring of training robustness.

- **Support for new domains**:
  - Expand beyond image classification to text, tabular data, and time series.

These extensions will further strengthen Safe-DL’s applicability in real-world ML security pipelines.

### 5.6 Conclusion of Module 4

This module marked a pivotal stage in the Safe-DL pipeline: the transition from identifying vulnerabilities to actively mitigating them. Leveraging the attack simulations and risk assessment outputs from Modules 2 and 3, Module 4 applies targeted, configurable, and modular defenses tailored to the model's specific weaknesses.

The framework supports a wide range of defenses spanning the three main attack categories explored in this project:

- **For data poisoning**, the system implements pre-training sanitization techniques, robust training objectives, and influence-based filtering to reduce the effect of mislabeled or malicious inputs.
- **For backdoor threats**, it applies both detection and mitigation strategies such as activation clustering, spectral analysis, and structural pruning to suppress or remove trigger dependencies.
- **For evasion attacks**, the module provides training- and inference-time defenses such as adversarial training, gradient masking, and input preprocessing techniques designed to harden the model’s decision boundary.

Each defense is applied independently, and its effects are measured through a standardized evaluation process. This includes:
- Accuracy on both clean and adversarial inputs,
- Per-class performance analysis,
- Visual evidence of removed or altered examples,
- JSON and Markdown reports for reproducibility and auditability.

The resulting model is not only hardened against its specific threat profile, but also fully traceable in terms of which defenses were applied and what impact they had. 

This prepares the user to proceed to **Module 5 — Comparative Evaluation**, where models trained with different defenses (or defense combinations) can be benchmarked side-by-side in terms of robustness, generalization, and computational trade-offs.

Module 4 thus serves as the operational backbone of the framework’s security loop — turning diagnosis into action, and ensuring that models are not only aware of their vulnerabilities, but actively protected against them.


    


----------

## 6. Module 5 — Evaluation & Benchmarking


### 6.1 Goal of the Module

Module 5 serves as the critical evaluation hub of the Safe-DL framework, assessing the effectiveness of all defenses applied in Module 4. It does not execute new attacks or defenses; instead, it systematically aggregates and analyzes the JSON outputs from both the initial attack simulations (Module 2) and the subsequent defense applications (Module 4). Its primary aim is to answer:

>**"How effective was each defense, and was it worth the associated trade-offs in performance or cost?"**

This module is crucial for transforming raw attack and defense metrics into actionable insights, providing a standardized, data-driven approach to understand the true impact of implemented countermeasures. It helps ensure that security enhancements are justified, do not introduce unacceptable costs in clean accuracy or model usability, and facilitates informed decision-making for deployment.


### 6.2 Analysis Workflow

The `run_module5.py` script orchestrates the defense evaluation process. The workflow is designed to be systematic and data-driven:

1.  **Profile Parsing**: The module first loads the `profile.yaml` to identify the configured attacks (from Module 2) and the corresponding defenses applied (from Module 4). This acts as the blueprint for the evaluation.
    
2.  **Data Aggregation**: For each attack identified in the profile, the system retrieves:
    * The baseline model's clean accuracy from `baseline_accuracy.json` (from Module 2).
    * The raw attack metrics (e.g., adversarial accuracy, Attack Success Rate) from `attack_metrics.json` (from Module 2's results for that specific attack).
    * For each defense applied against that attack, the post-defense performance metrics (`accuracy_clean`, `accuracy_adversarial`, `asr_after_defense`, etc.) from `defense_results.json` (from Module 4's results for that specific defense).
    
3.  **Evaluation and Scoring**: Using the aggregated data, Module 5 computes a set of standardized scores for each defense, reflecting its overall effectiveness and trade-offs. These scores are calculated using specific evaluation functions (e.g., `evaluate_backdoor_defense`, `evaluate_evasion_defense`, `evaluate_data_poisoning_defense`) based on the attack type:
    * **Mitigation Score**: Quantifies how effectively the defense restored the model's performance on adversarial inputs, relative to the attack's initial impact. A higher score indicates better recovery.
    * **Clean Accuracy Drop (CAD) Score**: Measures the degradation in the model's performance on clean, benign data due to the defense. A score closer to 1 indicates minimal impact on clean accuracy.
    * **Defense Cost Score**: An estimated numerical value reflecting the computational or implementation overhead associated with applying the defense. This is based on a predefined mapping in the framework.
    * **Final Score**: A weighted aggregation of the Mitigation, CAD, and Cost Scores, providing a single metric to compare defenses. This score encapsulates the balance between robustness gain, clean accuracy preservation, and resource expenditure.

----------


### 6.3 Output and Reporting

Module 5 generates structured outputs to provide a clear and actionable summary of the defense evaluation. Each evaluated defense is summarized with:

* **Quantitative Scores**: A set of standardized scores are provided, including:
    * **Mitigation Score**: Quantifies the defense's effectiveness in restoring model performance post-attack.
    * **Clean Accuracy Drop (CAD) Score**: Indicates the impact of the defense on the model's clean accuracy.
    * **Defense Cost Score**: An estimated measure of the defense's computational or implementation overhead.
    * **Final Score**: An aggregated score providing an overall measure of the defense's cost-benefit.
* **Structured JSON Output**: All evaluation results are saved to a comprehensive JSON file, `defense_evaluation.json`, designed for machine readability and easy integration into subsequent modules (like Module 6 for final reporting).
    ```json
    {
      "backdoor": {
        "static_patch": {
          "fine_pruning": {
            "mitigation_score": 0.691,
            "cad_score": 1.071,
            "defense_cost_score": 0.4,
            "final_score": 0.528
          }
        }
      },
      "data_poisoning": {
        "clean_label": {
          "provenance_tracking": {
            "mitigation_score": 0.561,
            "cad_score": 0.790,
            "defense_cost_score": 0.5,
            "final_score": 0.295
          }
        }
      },
      "evasion": {
        "pgd": {
          "adversarial_training": {
            "mitigation_score": 0.288,
            "cad_score": 0.0,
            "defense_cost_score": 0.8,
            "final_score": 0.0
          }
        }
      }
    }
    ```
* **Human-Readable Markdown Report**: A detailed Markdown report, `defense_evaluation_report.md`, is automatically generated, presenting the evaluation results in clear tables, along with explanatory notes and a summary overview. This report is designed for easy review and documentation.

Optional future enhancements include more detailed performance summaries (e.g., training time, memory overhead), per-class analysis visualizations, and robustness curves for a more granular view of defense impact.


----------

### 6.4 Technical Considerations

Module 5 is designed with the following key technical considerations:

* **Read-only Analysis**: The module operates solely by reading and processing JSON outputs from Modules 2 and 4. It does not perform any new model training, inference, or re-execution of attacks/defenses, ensuring a lightweight and efficient evaluation process.
* **Data-Driven Evaluation**: All computations and scoring are based purely on the comparison of quantitative metrics extracted from the `.json` results of prior stages, making the evaluation objective and transparent.
* **Reproducible and Modular**: The design ensures that the evaluation is fully reproducible and aligns with the framework's modular philosophy, allowing for independent execution and easy integration into automated pipelines.

----------

### 6.5 Future Enhancements

While the current evaluation focuses on accuracy-based metrics and defense costs, planned improvements aim to broaden its scope and utility:

* **Measuring Operational Overhead**: Incorporating metrics beyond accuracy, such as training time, inference latency, and memory footprint, to provide a more holistic view of defense costs.
* **Dynamic Risk Re-assessment**: Integrating feedback from defense evaluation scores back into Module 3 (Risk Analysis) to enable a dynamic re-assessment of residual risks after defenses are applied, offering a complete picture of the updated security posture.
* **Advanced Metric Aggregation**: Expanding the analysis to include the aggregation of robustness curves (e.g., accuracy vs. epsilon for PGD) and exploring the integration of formal verification methods for certified robustness.
* **Cross-Context Comparison**: Facilitating the comparison of defense effectiveness across different model architectures, datasets, or deployment scenarios to identify universally robust solutions.

----------

### 6.6 Conclusion of Module 5

Module 5 serves as the critical evaluation hub of the Safe-DL framework, transforming raw attack and defense metrics into actionable insights. By systematically quantifying defense effectiveness, trade-offs, and costs, this module provides users with a standardized, data-driven approach to understanding the true impact of implemented countermeasures. The generated `defense_evaluation.json` and `defense_evaluation_report.md` empower users to:

* **Make Informed Decisions**: Compare different defense strategies based on concrete performance metrics.
* **Understand Trade-offs**: Clearly identify the balance between adversarial robustness gains and potential degradations in clean accuracy or increases in operational cost.
* **Drive Iterative Improvements**: Highlight areas where defenses are most effective and where further research or tuning might be required.

Ultimately, Module 5 ensures that defense application within the Safe-DL framework is not a blind deployment but a strategic, evidence-based process aimed at building more resilient deep learning systems.

## 7. Module 6 — Final Report Aggregation


### 7.1 Goal of the Module

The sixth and final module, Module 6, culminates the Safe-DL framework by aggregating all generated artifacts into **one self-contained, human-readable dossier: `final_report.md`**. Its core objectives are to:

1.  **Consolidate Comprehensive Data**: Gather and integrate all threat-model definitions, attack simulation results, risk analysis findings, defense application details, and defense evaluation scores from the preceding modules.
2.  **Provide a Unified Overview**: Present auditors, collaborators, and stakeholders with a single, authoritative source of truth that offers a holistic view of the deep learning model's security posture and chronicles its entire security lifecycle.
3.  **Ensure Reproducibility and Clarity**: Embed all relevant YAML/JSON metadata and results alongside clear narrative explanations to guarantee reproducibility and facilitate understanding of every decision and outcome.

The primary deliverable is `final_report.md`, meticulously structured to be directly appended to technical documentation or an academic dissertation.

----------


### 7.2 Aggregation Workflow

The `run_module6.py` script orchestrates the generation of the final report, following a precise sequence to ensure all data is correctly integrated.

1.  **Profile Selection & Data Loading**: The workflow begins by prompting the user to select a `profile.yaml` file, which serves as the central source of truth. This file, along with various JSON and Markdown reports generated by Modules 2 through 5, is loaded using dedicated utility functions.
    
2.  **Results Harvesting**: The module systematically collects all necessary data from previous modules, ensuring a comprehensive final report. The key JSON outputs utilized are:

| Module | Expected JSON(s)                                   | Purpose in Final Report                                    |
| :----- | :------------------------------------------------- | :--------------------------------------------------------- |
| 2      | `attack_metrics.json` (from specific attack folders) | Baseline & attack-time performance (Attack Simulation)     |
| 3      | `risk_analysis.json`                               | Severity / Probability / Visibility scores, Risk Ranking   |
| 4      | `defense_results.json` (from specific defense folders) | Post-defense performance & defense parameters              |
| 5      | `defense_evaluation.json`                          | Defense evaluation scores (Mitigation, CAD, Cost, Final)   |

3.  **Section Generation**: Each major section of the `final_report.md` is dynamically generated by dedicated functions within `generate_report_utils.py`. These functions retrieve and format the relevant data into structured Markdown, ensuring logical flow and human readability.
    
4.  **Report Output**: The fully aggregated Markdown content is then written to `final_report.md` in the `reports/` directory. While the primary output is Markdown, the framework is designed to allow for future expansion to other formats (e.g., PDF conversion via tools like Pandoc, though this is currently an optional flag not directly implemented by `run_module6.py`).
    

----------


### 7.3 Input Sources

Module 6 relies on the successful completion and JSON output generation of all preceding modules. The `profile.yaml` serves as the central orchestrator, accumulating key data, while detailed results are harvested primarily from specific JSON files within each module's `results/` directory.

```text
profiles/
└── <profile>.yaml          ← Central source of truth, incrementally updated by all modules

module2_attack_simulation/
└── results/
    └── <attack_name>/<attack_method>/
        └── attack_metrics.json    ← Raw performance metrics for each simulated attack

module3_risk_analysis/
└── results/
    └── risk_analysis.json  ← Calculated risk scores, ranking, and recommendations

module4_defense_application/
└── results/
    └── <attack_name>/<attack_method>/<defense_name>/
        └── defense_results.json    ← Post-defense performance metrics & parameters

module5_defense_evaluation/
└── results/
    └── defense_evaluation.json ← Comprehensive defense evaluation scores

```

All paths are automatically resolved by `run_module6.py` and its utility functions, eliminating the need for manual bookkeeping as long as the default directory layout of the framework is preserved.

----------


### 7.4 Report Contents

The `final_report.md` is meticulously structured into distinct sections, each consolidating specific information from the various stages of the Safe-DL framework. The table below outlines the key contents and their primary sources.

| **Final Report Section No.** | **Section Title** | **Key Elements** | **Primary Source** |
| :--------------------------- | :----------------------------------------- | :-------------------------------------------------------------------- | :----------------------------------------------- |
| **1.** | Report Header and Overview                 | Report title, profile name, generation date/time, introduction        | `run_module6.py`, general overview               |
| **2.** | System Under Evaluation Details            | Model and dataset details (name, type, shape, classes)                | `profile.yaml`                                   |
| **3.** | Threat Profile Summary (Module 1)          | Model access, attack goal, deployment scenario, data sensitivity, threat categories | `profile.yaml` (from Module 1)                   |
| **4.** | Attack Simulation Results (Module 2)       | Overview table of simulated attacks (clean acc, impact, attack metric), key parameters, links | Module 2 `attack_metrics.json`, `profile.yaml`   |
| **5.** | Risk Analysis and Matrix (Module 3)        | Risk summary table (Severity, Probability, Visibility, Risk Score), qualitative risk matrix, ranking, defense recommendations, links | Module 3 `risk_analysis.json`, `profile.yaml`    |
| **6.** | Defense Application Summary (Module 4)     | Overview table of applied defenses (pre/post metrics), key parameters, links, defense purposes | Module 4 `defense_results.json`, `profile.yaml`  |
| **7.** | Defense Evaluation and Scoring (Module 5)  | Summary table of defense evaluation scores (Mitigation, CAD, Cost, Final), top-performing defenses, observations | Module 5 `defense_evaluation.json`, `profile.yaml` |
| **8.** | Conclusions and Executive Summary          | Highest-risk attack, most effective defenses, notable gaps, overall security posture, practical recommendations | Aggregation of Modules 2-5 outputs, `profile.yaml` |
| **9.** | Recommendations for Continuous Monitoring and Post-Deployment | Monitoring metrics, periodic re-assessment, alerting, incident response, CI/CD integration | Static content within `generate_report_utils.py` |

----------


----------

### 7.5 Example Output Snippet

To illustrate the structure and content of the generated `final_report.md`, below is a concise snippet from the "Attack Simulation Results" section (corresponding to Section 4 in the final report), demonstrating how key attack metrics are presented.

```markdown
## 4. Attack Simulation (Module 2)
This section summarizes the outcomes of the adversarial attack simulations performed against the model based on the defined threat profile. These simulations quantify the model's vulnerability to various attack types before any defenses are applied.

### 4.1 Overview of Simulated Attacks

| Attack Category | Attack Method | Clean Acc. (Pre-Attack) | Impact on Clean Acc. | Attack Metric | Key Parameters | Full Results |
|:----------------|:--------------|:------------------------|:---------------------|:--------------|:---------------|:-------------|
| Data Poisoning | Clean Label | 67.54% | 62.76% | 62.76% (Degraded Acc.) | Poison Fraction: 0.05, Target Class: 5 | [Details](../module2_attack_simulation/results/data_poisoning/clean_label/clean_label_report.md) |
| Backdoor | Static Patch | 67.54% | 66.62% | 92.30% (ASR) | Poison Frac.: 0.05, Target Class: 7, Patch Type: white_square | [Details](../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md) |
| Evasion | Pgd | 67.54% | 67.54% | 0.00% (Adv. Acc.) | Epsilon: 0.03, Num Iter: 50 | [Details](../module2_attack_simulation/results/evasion/pgd/pgd_report.md) |

**Note**: 'Clean Acc. (Pre-Attack)' represents the model's accuracy on clean data before any attack preparations. 'Impact on Clean Acc.' shows the model's accuracy on clean data *after* being subjected to the attack (e.g., trained with poisoned data, or backdoor injected). For Data Poisoning attacks, 'Attack Metric' displays the degraded accuracy of the model on clean inputs after poisoning. For Backdoor attacks, 'Attack Metric' displays the Attack Success Rate (ASR), indicating the percentage of adversarial samples (with trigger) successfully misclassified to the target class. For Evasion attacks, 'Attack Metric' displays the Adversarial Accuracy (Adv. Acc.) on perturbed inputs, where a lower value indicates a more successful attack.

```

This snippet directly reflects a part of Section 4 (`Attack Simulation`) of the `final_report.md` as documented in `module6.md`.


----------

### 7.6 Technical Considerations

Module 6 is designed with several key technical principles:

* **Read-only and Lightweight**: The module primarily focuses on data aggregation and formatting. It performs no new training, inference, or computationally intensive operations, making it fast, lightweight, and highly repeatable.
* **Utility-Function Driven**: Instead of a generic templating engine, the report generation is driven by dedicated utility functions within `generate_report_utils.py`. These functions are responsible for parsing the collected data and dynamically constructing each section of the Markdown report, ensuring precise control over content and formatting.
* **Dependency Enforcement**: Module 6 relies heavily on the outputs of the preceding modules (2, 3, 4, and 5). The execution expects these outputs to be present and properly formatted, ensuring that the final report is comprehensive and accurate. Missing critical input data will result in execution warnings or errors, guiding the user to complete the pipeline.
    

----------

### 7.7 Future Enhancements

Potential future enhancements for Module 6 and the framework's reporting capabilities include:

* **Integrated PDF Export**: Developing built-in functionality for automatic PDF export of the `final_report.md`, potentially with customizable styling to meet institutional or journal compliance.
* **Interactive Dashboard**: Implementing an interactive web-based dashboard (e.g., using Streamlit or Gradio) that allows for dynamic drill-down exploration of attack results, defense effectiveness, and risk analysis.
* **CI/CD Integration Hook**: Providing a dedicated hook or script for seamless integration into Continuous Integration/Continuous Deployment (CI/CD) pipelines, enabling automated generation of updated final reports with every new experiment or code branch.
    

----------

### 7.8 Conclusion of Module 6

Module 6 effectively seals the Safe-DL pipeline, weaving every intermediate artifact into a coherent, audit-ready narrative. The resulting `final_report.md` serves as the authoritative document, certifying **what was threatened, how it was attacked, how it was protected, and how effective those protections proved to be**. Through this comprehensive aggregation, the framework not only aids in securing deep learning models but also meticulously **secures the story** of their robustness journey, providing invaluable insights for continuous security improvement.


## 8. Example Application of the Safe-DL Framework (End-to-End Walk-through)

### 8.1 Scenario

A team is building an on-board traffic-sign-recognition model for an autonomous-vehicle prototype.  
Key constraints:

-   **Training data**: public datasets (GTSRB + synthetic augmentations).
    
-   **Deployment**: edge devices inside the vehicle; inference exposed through an _internal_ REST API.
    
-   **Stake-holders**: safety auditors require a full security dossier before road testing.
    

----------

### 8.2 Module 1 — Threat Modeling

The team runs the CLI questionnaire and obtains:

```yaml
threat_model:
  model_access: gray-box          # insiders may inspect weights, outsiders only via API
  attack_goal: targeted
  deployment_scenario: physical_world
  data_sensitivity: medium
  training_data_source: external_public
  interface_exposed: api
  threat_categories:
    - data_poisoning
    - backdoor_attacks
    - adversarial_examples

```

> Public data, physical deployment and API exposure push the framework to flag **poisoning**, **backdoor** and **evasion** as high-priority threats.

----------

### 8.3 Module 2 — Attack Simulation

| Attack Category | Implementation |  Key Result
|--|--|-- 
|Data Poisoning  | Clean-label + label flipping (10 % ) |**–21 pp** clean-accuracy drop
|Backdoor|Static patch (white square)|**ASR = 92 %** (target = “Speed-30”)
|Evasion|PGD ε = 0.03 (40 steps)|Robust-accuracy **34 %**

_(Model-stealing attacks are planned future work, so no simulation yet.)_

----------

### 8.4 Module 3 — Risk Analysis

```yaml
risk_analysis:
  summary:
    backdoor_static_patch:
      severity: 0.92
      probability: 0.9
      visibility: 0.3
      risk_score: 1.48
    data_poisoning_clean_label:
      severity: 0.70
      probability: 0.8
      visibility: 0.6
      risk_score: 0.92
    pgd_epsilon_0.03:
      severity: 0.89
      probability: 1.0
      visibility: 0.35
      risk_score: 1.46
  recommendations:
    backdoor_static_patch:
      - activation_clustering
      - neuron_pruning
    data_poisoning_clean_label:
      - data_cleaning
      - robust_loss
    pgd_epsilon_0.03:
      - adversarial_training

```

The **backdoor** and **PGD** evasion attacks surface as “critical”, informing the defense plan.

----------

### 8.5 Module 4 — Defense Application

Accepted recommendations (CLI):

```yaml
defense_config:
  data_poisoning:
    - method: data_cleaning
    - method: robust_loss        # symmetric CE, β = 0.1
  backdoor_attacks:
    - method: activation_clustering
    - method: neuron_pruning
      pruning_ratio: 0.20
  adversarial_examples:
    - method: adversarial_training
      attack_type: pgd
      epsilon: 0.03
      num_steps: 10

```

Actions executed:

-   **Data-cleaning** removed 218 suspect images (2 %).
    
-   **Robust-loss** retraining finished with negligible clean-accuracy loss.
    
-   **Activation clustering + 20 % pruning** excised backdoor neurons and retrained for 5 epochs.
    
-   **PGD adversarial-training** ran 10 epochs (≈ 1.4× baseline training time).
    

----------


### 8.6 Module 5 — Evaluation & Benchmarking

Module 5 provides a quantitative assessment of defense effectiveness. The table below illustrates a summary of the defense evaluation, showcasing key scores for different attack-defense pairs:

| Attack           | Defense             | Mitigation | CAD   | Cost  | Final Score* |
| :--------------- | :------------------ | :--------- | :---- | :---- | :----------- |
| Backdoor (Static)| Fine-Pruning        | 0.69       | 1.07  | 0.40  | **0.53** |
| Data Poisoning   | Provenance Tracking | 0.56       | 0.79  | 0.50  | **0.30** |
| Evasion (PGD)    | Adversarial Training| 0.29       | 0.00  | 0.80  | **0.00** |

* **Mitigation Score**: Effectiveness in restoring model performance after an attack (higher is better, max 1.0).
* **CAD (Clean Accuracy Drop) Score**: Degree of performance degradation on clean data (higher is better, max 1.0+).
* **Cost Score**: Relative computational/resource impact of the defense (lower is better, min 0.1).
* **Final Score**: Aggregated score combining Mitigation (60%), CAD (30%), and Cost (10%).

> This evaluation provides concrete, data-driven insights into the trade-offs of each defense, enabling informed decisions on which countermeasures are most suitable for a given threat.
----------

### 8.7 Module 6 — Final Report Aggregation

Running `run_module6.py` produces **`final_report.md`** (and optionally other formats) that bundles:

-   Consolidated threat-model definitions and system details.
-   Comprehensive attack simulation results, including performance metrics and links to detailed reports.
-   In-depth risk analysis, featuring risk scores, matrices, and prioritized recommendations.
-   Detailed summary of applied defenses, outlining their configurations and initial impact.
-   Quantitative evaluation of defense effectiveness, including mitigation, clean accuracy drop, cost, and final scores.
-   Executive summary, conclusions, and practical recommendations for current and post-deployment security.
-   Recommendations for continuous monitoring, periodic re-assessment, alerting, incident response, and CI/CD integration.

The final report is generated in the `reports/` directory as `final_report.md` and serves as the ultimate audit-ready dossier, ready to be integrated into project documentation.

----------

### 8.8 Final Outcome

The Safe-DL framework's culmination, delivered through `final_report.md`, ensures a robust security posture by:

1.  **Risks Identified and Prioritized**: High-impact vulnerabilities (e.g., specific backdoor and evasion weaknesses) are clearly defined and ranked based on quantitative risk analysis.
2.  **Attacks Quantitatively Reproduced**: Simulated adversarial attacks are presented with verifiable quantitative evidence of their impact.
3.  **Targeted Defenses Applied**: Specific, context-aware defense mechanisms are deployed against identified threats.
4.  **Effectiveness Quantified**: The efficacy and trade-offs of each applied defense are rigorously evaluated and scored.
5.  **Actionable Recommendations Provided**: Concrete, data-driven advice for immediate mitigation and continuous security improvement is given.
6.  **Comprehensive Documentation Generated**: A single, self-contained, and reproducible report is created, narrating the entire security assessment journey from threat modeling to defense evaluation.

This process ensures that deep learning models are not only assessed for vulnerabilities but are also provided with a clear, documented path to enhanced security and resilience.




Tiago Barbosa, 2025
