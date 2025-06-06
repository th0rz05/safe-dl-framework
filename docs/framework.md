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

All selected defense strategies and their parameters are stored in the `defense_config` block of the main YAML profile. Example:

```yaml
defense_config:
  data_poisoning:
    - method: data_cleaning
    - method: robust_loss
  backdoor:
    - method: activation_clustering
  evasion:
    - method: adversarial_training
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

After applying each defense, the framework generates standardized and structured outputs to facilitate evaluation, reproducibility, and comparison.

#### 5.4.1 JSON Reports

Each defense generates a `defense_results.json` file containing:

- `accuracy_clean`: Accuracy on clean (unperturbed) test data after applying the defense.
- `accuracy_adversarial`: Accuracy on adversarial/test-time perturbed data.
- `per_class_accuracy_clean`: Class-wise accuracy on clean inputs.
- `per_class_accuracy_adversarial`: Class-wise accuracy on adversarial inputs.

These fields are consistent across all defense types and are designed to be parsed in downstream modules (e.g., Module 5 — Comparative Evaluation).

Additional fields may include:
- `removed_samples`: List of filtered/flagged training inputs (when applicable).
- `parameters`: Configuration used during defense execution.

#### 5.4.2 Markdown Reports

A human-readable `defense_report.md` is also generated for each defense, containing:

- Overview of the defense method used.
- Visualizations such as:
  - Accuracy bars (before/after)
  - Sample images (e.g., removed inputs, adversarial recovery)
  - Confusion matrices or heatmaps if applicable
- Key results (clean and adversarial accuracy)
- Notes about defense performance, limitations, or warnings.

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

This module performs a structured evaluation of all defenses applied in Module 4. Unlike previous stages, it does not run any new attacks or defenses. Instead, it systematically aggregates and analyzes the JSON outputs of both the original attack simulations (Module 2) and the defense applications (Module 4), with the aim of answering:

> **"How effective was each defense, and was it worth the trade-off?"**

This module is critical to ensure that security enhancements do not come at an unjustifiable cost in clean accuracy or model usability. It also provides a comparative view across multiple defenses, enabling informed decisions for deployment.

----------

### 6.2 Analysis Workflow

1.  **Threat Profile Parsing**  
    The system loads the `.yaml` profile to identify which defenses were applied per threat category and attack.
    
2.  **Data Aggregation**  
    For each defense, it loads the corresponding `defense_results.json` (generated in Module 4), and retrieves:
    
    -   `accuracy_clean`
        
    -   `accuracy_adversarial`
        
    -   `per_class_accuracy_clean`
        
    -   `per_class_accuracy_adversarial`
        
3.  **Baseline Metrics Retrieval**  
    For each attack, the system also loads the original `attack_metrics.json` (from Module 2) to obtain:
    
    -   Clean accuracy before any defense
        
    -   Adversarial accuracy or Attack Success Rate (ASR), depending on the threat type
        
4.  **Evaluation and Scoring**  
    For each defense method, the system computes:
    
   - Δ Clean Accuracy  
  *(Change in performance on clean data)*  
  `Δ_clean = accuracy_defended_clean - accuracy_baseline_clean`

- Δ Adversarial Accuracy or Δ ASR  
  *(Improvement against adversarial or poisoned inputs)*  
  `Δ_adv = accuracy_defended_adv - accuracy_baseline_adv`

- Defense Score 
  A weighted heuristic combining accuracy deltas and per-class stability:  
  `score = α * Δ_adv + β * Δ_clean + γ * avg_per_class_gain`

        
        _(Default weights: α = 0.4, β = 0.2, γ = 0.4 — configurable in future versions)_
        

----------

### 6.3 Output and Reporting

Each evaluated defense is summarized with:

-   Delta metrics (clean and adversarial)
    
-   Per-class analysis
    
-   Defense score
    
-   Performance summary (future work: training time, memory overhead)
    
-   YAML/JSON summary with structured results:
    

```yaml
benchmark_summary:
  data_poisoning:
    label_flipping:
      data_cleaning:
        delta_clean: -0.2
        delta_adv: +26.5
        defense_score: 18.9
      robust_loss:
        delta_clean: -0.1
        delta_adv: +12.3
        defense_score: 9.1

```

Optional reports can also be exported in `.md`, `.csv`, and visual formats (e.g., bar plots, heatmaps, radar charts).

----------

### 6.4 Technical Considerations

-   No attacks or defenses are re-executed.
    
-   All evaluations are based purely on reading and comparing `.json` results.
    
-   The evaluation is reproducible, configurable, and aligned with the framework’s modular philosophy.
    

----------

### 6.5 Future Enhancements

The current evaluation focuses on accuracy-based metrics. Planned improvements include:

-   Measuring additional overhead: training time, inference latency, memory footprint.
    
-   Aggregating robustness curves (e.g., accuracy vs epsilon for PGD).
    
-   Certifying robustness using formal methods (e.g., interval bound propagation).
    
-   Comparing defenses across different model architectures or datasets.
    

----------

### 6.6 Conclusion of Module 5

This module closes the evaluation loop of the Safe-DL framework. It provides evidence-based insight into the cost-benefit trade-offs of each defense, grounded in concrete metrics. By leveraging previously generated results, it enables reproducible and lightweight benchmarking, and prepares the ground for robust deployment or further optimization in future iterations.

## 7. Module 6 — Final Report Aggregation

### 7.1 Goal of the Module

The sixth and final module turns every artifact generated by the Safe-DL pipeline into **one self-contained, human-readable dossier**.  
Its objectives are to

1.  Consolidate threat-model decisions, attack results, risk scores, defense configurations, and benchmark metrics;
    
2.  Provide auditors, collaborators, or thesis reviewers with a single source of truth that chronicles the entire security lifecycle;
    
3.  Guarantee reproducibility by embedding all YAML/JSON metadata alongside narrative explanations.
    

The deliverable is `final_report.md` (optionally `final_report.pdf`), ready to be appended to technical documentation or an academic dissertation.

----------

### 7.2 Aggregation Workflow

1.  **Profile Parsing** Read `profile.yaml` to recover the threat profile, chosen datasets/models, enabled attacks, and applied defenses.
    
2.  **Results Harvesting** Locate and load every JSON produced by earlier modules:
    

|Module|Expected JSON(s)|Purpose in final report
|---|--------|-----
| 2  |   `*_metrics.json`     |Baseline & attack-time performance
|3|`risk_analysis.json`|Severity / probability / visibility scores
|4|`*defense_results.json`|Post-defense performance & parameters
|5|`benchmark_summary.json`|Clean vs. adversarial deltas & defense scores

3.  **Section Generation** Render each report section (see § 7.4) via templating (Jinja2 in the reference implementation).
    
4.  **Asset Embedding** Pull in plots and sample images (if present) from the `results/` tree.
    
5.  **Export** Write `final_report.md`; optional flags `--html` / `--pdf` enable additional formats via Pandoc.
    

----------

### 7.3 Input Sources

```text
profiles/
└── <profile>.yaml          ← single source of truth
module2_attack_simulation/
└── results/**/             ← *_metrics.json for each attack
module3_risk_analysis/
└── results/risk_analysis.json
module4_defense_application/
└── results/**/             ← *defense_results.json for each defense
module5_evaluation/
└── results/benchmark_summary.json   (optional if module 5 not yet implemented)

```

Paths are resolved automatically; no manual bookkeeping is required as long as the default directory layout is preserved.

----------

### 7.4 Report Contents

| **Section** | **Key Elements** |**Primary Source**
|--|--|--
| **7.4.1 Threat-Model Overview** | Attacker access, data sensitivity, deployment scenario |`profile.yaml`
|**7.4.2 Attack-Simulation Summary**|Parameters and impact of each attack|Module 2 JSONs
|**7.4.3 Risk Analysis**|Severity × Probability × (1 + (1 – Visibility)) matrix; ranking|`risk_analysis.json`
|**7.4.4 Defensive Actions**|Configuration of every defense, rationale, YAML snippet|`defense_config` in profile
|**7.4.5 Benchmark Results**|Δ clean accuracy, Δ adversarial accuracy, defense score|Module 5 (or on-the-fly recompute)
|**7.4.6 Visual Appendix**|Poisoned samples, backdoor triggers, robustness curves|`results/**/examples/`
|**7.4.7 Full Configuration Dump**|Embedded YAML & JSON for reproducibility|all above

----------

### 7.5 Example Output Snippet

```markdown
## 7.4.2 Attack-Simulation Summary – Backdoor (Static Patch)

| Metric | Value |
|--------|-------|
| Clean accuracy (baseline) | **91.2 %** |
| Accuracy on triggered set | **15.0 %** |
| Attack-success rate (ASR) | **85.0 %** |
| Trigger position | bottom-right |
| Blend-alpha | 0.20 |

![backdoor_trigger](results/module2_attack_simulation/backdoor/static_patch/overlay.png)

```

----------

### 7.6 Technical Considerations

-   **Read-only module** – it performs no training or inference, so it is lightweight and repeatable.
    
-   **Template-driven** – switching from Markdown to HTML or LaTeX merely requires a new template.
    
-   **Fail-soft** – missing sections (e.g., if Module 5 has not been run) are flagged but do not halt generation.
    

----------

### 7.7 Future Enhancements

-   **Automatic PDF export** with institutional or journal-compliant styling.
    
-   **Interactive dashboard** (Streamlit/Gradio) for drill-down exploration of attacks and defenses.
    
-   **Continuous-integration hook** so every new experiment branch produces an updated final report.
    

----------

### 7.8 Conclusion of Module 6

Module 6 seals the Safe-DL pipeline, weaving every intermediate artifact into a coherent, audit-ready narrative. The resulting document certifies **what was threatened, how it was attacked, how it was protected, and how effective those protections proved to be**. With this, the framework not only secures deep-learning models but also **secures the story** of their robustness journey.


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

|Attack / Metric| Before Defense  |After Defense|Δ Adv Acc ↑|Δ Clean Acc ↓|Defense Score*
|--|--|--|--|--|--
| PGD (ε 0.03) | 34 % |**72 %**|**+38 pp**|–2.8 pp|**18.4**
|Backdoor ASR|92 %|**5 %**|–87 pp|–0.9 pp|16.7
|Data Poisoning (Clean)|79 %|**88 %**|n/a|**+9 pp**|11.2

* `score = 0.4·Δ_adv + 0.2·Δ_clean + 0.4·avg_per_class_gain` (normalised).

> All three defenses drastically cut attack success while keeping total clean-accuracy within ±3 pp of the original baseline.

----------

### 8.7 Module 6 — Final Report Aggregation

Running `generate_final_report.py` produces **`final_report.md`** (and optional PDF) that bundles:

-   Threat-model YAML
    
-   Attack metrics, risk matrix, and recommendations
    
-   Defense configs, removed-sample galleries, pruning histograms
    
-   Benchmark tables and plots
    
-   Full reproducibility appendix (all JSON / YAML dumps)
    

The dossier is stored under `reports/final_report_<timestamp>.md` and attached to the project repository.

----------

### 8.8 Final Outcome

1.  **Risks identified** (high-impact backdoor & PGD weaknesses).
    
2.  **Attacks reproduced** with quantitative evidence.
    
3.  **Critical vulnerabilities mitigated** via targeted defenses.
    
4.  **Effectiveness benchmarked** (defense scores 11 – 18).
    
5.  **Comprehensive report generated** for auditors and future maintenance.
    

The traffic-sign model is now cleared for closed-track testing with continuous logging and anomaly alerts enabled.



Tiago Barbosa
