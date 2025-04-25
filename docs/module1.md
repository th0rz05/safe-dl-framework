
# Module 1 – Threat Modeling

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Design of the Threat Questionnaire](#2-design-of-the-threat-questionnaire)
  * [2.1 Design Principles](#21-design-principles)
  * [2.2 Format and Execution](#22-format-and-execution)
- [3. Explanation of Each Field](#3-explanation-of-each-field)
  * [3.1 model_access](#31-model-access)
  * [3.2 attack_goal](#32-attack-goal)
  * [3.3 deployment_scenario](#33-deployment-scenario)
  * [3.4 data_sensitivity](#34-data-sensitivity)
  * [3.5 training_data_source](#35-training-data-source)
  * [3.6 model_type](#36-model-type)
  * [3.7 interface_exposed](#37-interface-exposed)
  * [3.8 threat_categories](#38-threat-categories)
- [4. Automatic Threat Suggestion Logic](#4-automatic-threat-suggestion-logic)
  * [4.1 Purpose of Automatic Suggestion](#41-purpose-of-automatic-suggestion)
  * [4.2 Rules Used for Suggesting Threats](#42-rules-used-for-suggesting-threats)
  * [4.3 Integration with the Framework](#43-integration-with-the-framework)
- [5. Example Profiles](#5-example-profiles)
  * [5.1 Mobile Application Example](#51-mobile-application-example)
  * [5.2 Cloud API Example](#52-cloud-api-example)
- [6. Discussion and Impact](#6-discussion-and-impact)
  * [6.1 Alignment with Realistic Threats](#61-alignment-with-realistic-threats)
  * [6.2 Educational Value](#62-educational-value)
  * [6.3 Modularity and Integration](#63-modularity-and-integration)
  * [6.4 Limitations and Future Improvements](#64-limitations-and-future-improvements)


## 1. Introduction

As deep learning systems become increasingly deployed in real-world, high-stakes applications, understanding the threats that may compromise their integrity, availability, or confidentiality becomes essential. Module 1 of this framework addresses this challenge by guiding users through a structured process of **threat modeling**, specifically adapted for deep learning workflows.

Traditional threat modeling techniques used in cybersecurity are often too abstract or general to account for the unique characteristics of machine learning systems — such as susceptibility to data poisoning, adversarial perturbations, or model inversion. Therefore, this module proposes a tailored approach that allows users to **explicitly define the attacker assumptions and deployment context** of their model before moving on to attack simulations and defense strategies.

The core deliverable of this module is a `threat_profile.yaml`, generated through an interactive questionnaire. This profile includes structured information about the model's accessibility, the deployment context, the data pipeline, the expected attack goals, and the likely threat categories to be considered in the next phases.

By formalizing this first step, the framework ensures that any user — regardless of experience — can identify the most relevant threats for their use case. This also allows the framework to adapt dynamically in subsequent modules, such as selecting which attacks to simulate or which defenses to prioritize.

Ultimately, this module provides not only a practical tool to structure the threat landscape of a given project but also serves an educational purpose by making implicit security assumptions explicit.

## 2. Design of the Threat Questionnaire

The threat modeling process is operationalized through a command-line questionnaire that guides the user step-by-step in defining the most relevant aspects of their model’s deployment and threat landscape. This questionnaire is not only meant to collect structured input but also to **educate the user during the process** through contextual help and real-world examples.

The questionnaire outputs a standardized YAML file (`threat_profile.yaml`) that encodes all user decisions and can be reused across the remaining modules of the framework.

### 2.1 Design Principles

The design principles ensure the questionnaire is accessible, reusable, informed by evidence, and dynamically responsive to user inputs.

-   **Clarity and usability:** Each question is phrased in accessible language and includes a help option that explains the available choices.
    
-   **Modularity and reusability:** The questionnaire is implemented in Python using `questionary` and `pyyaml`, making it portable and integratable into automation pipelines.
    
-   **Evidence-based structure:** The structure and content of the questionnaire are based on the most common threat vectors and deployment-specific risks in deep learning systems.
    
-   **Dynamic response logic:** Based on user answers, the questionnaire can suggest relevant threat categories automatically, reducing user error and ensuring internal consistency.
    

### 2.2 Format and Execution

The questionnaire is executed through a command-line interface to maintain simplicity and platform independence.

Example command:

```bash
python threat_model_cli.py

```

Upon execution, users are guided through structured questions, and the resulting profile is saved for subsequent modules.

Example output:

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

This modular format allows users to save multiple profiles, share them across teams, and ensure consistent threat assumptions throughout simulations, defenses, and evaluations.

## 3. Explanation of Each Field

Each field included in the threat modeling questionnaire serves a specific purpose in defining the model's risk exposure and defense requirements.

### 3.1 model_access

This field defines the attacker’s assumed level of access to the model, influencing which attacks are feasible and what defenses may be necessary.

**Possible values:**

-   `white-box`
    
-   `gray-box`
    
-   `black-box`
    

### 3.2 attack_goal

This field specifies the objective of the attacker, which affects the type of attacks and the evaluation metrics used.

**Possible values:**

-   `targeted`
    
-   `untargeted`
    

### 3.3 deployment_scenario

This field identifies where the model will be deployed, affecting the available attack surfaces and defenses.

**Possible values:**

-   `cloud`
    
-   `edge`
    
-   `mobile`
    
-   `api_public`
    
-   `on_device`
    

### 3.4 data_sensitivity

This field measures how sensitive the training data is, which has direct implications on privacy-related attack risks.

**Possible values:**

-   `high`
    
-   `medium`
    
-   `low`
    

### 3.5 training_data_source

This field describes where the training data originates, influencing risks like poisoning and backdoor attacks.

**Possible values:**

-   `internal_clean`
    
-   `external_public`
    
-   `user_generated`
    
-   `mixed`
    

### 3.6 model_type

This field specifies the neural network architecture type, which affects susceptibility to specific attack types and defenses.

**Possible values:**

-   `cnn`
    
-   `transformer`
    
-   `mlp`
    
-   `other`
    

### 3.7 interface_exposed

This field describes how external users access the model, which shapes the external threat vectors.

**Possible values:**

-   `api`
    
-   `local_app`
    
-   `sdk`
    
-   `none`
    

### 3.8 threat_categories

This field lists the categories of threats that will be considered in subsequent modules.

**Possible values:**

-   `data_poisoning`
    
-   `backdoor_attacks`
    
-   `adversarial_examples`
    
-   `model_stealing`
    
-   `membership_inference`
    
-   `model_inversion`
    

## 4. Automatic Threat Suggestion Logic

The framework includes automatic threat suggestion logic to assist users in building a relevant and coherent threat profile.

### 4.1 Purpose of Automatic Suggestion

The purpose of the automatic suggestion system is to guide users by translating system architecture and operational details into recommended security concerns, creating a baseline that can be refined manually.

### 4.2 Rules Used for Suggesting Threats

The suggestion logic uses simple rule-based mappings based on user answers.

-   **Training data source:**
    
    -   External or user-generated data suggests `data_poisoning`.
        
-   **Interface exposure:**
    
    -   Public APIs suggest `model_stealing` and `membership_inference`.
        
-   **Deployment scenario:**
    
    -   Mobile and edge deployments suggest `adversarial_examples`.
        
-   **Model access level:**
    
    -   White-box access suggests `backdoor_attacks`.
        
-   **Data sensitivity:**
    
    -   High sensitivity suggests `model_inversion`.
        

These rules ensure that suggestions align with practical risk scenarios and encourage more realistic threat modeling.

### 4.3 Integration with the Framework

The threat categories selected or confirmed by the user are recorded in the YAML profile. Subsequent modules use these entries to drive attack simulation choices and defense recommendations.

Users always have the flexibility to modify the automatically suggested threat list.

## 5. Example Profiles

Realistic profiles show how different threat modeling decisions impact the security assumptions for different types of systems.

### 5.1 Mobile Application Example

Example for a mobile app using a CNN model embedded within an SDK:

```yaml
threat_model:
  model_access: black-box
  attack_goal: targeted
  deployment_scenario: mobile
  data_sensitivity: medium
  training_data_source: external_public
  model_type: cnn
  interface_exposed: sdk
  threat_categories:
    - data_poisoning
    - adversarial_examples

```

### 5.2 Cloud API Example

Example for a transformer model exposed via a public API in a sensitive healthcare context:

```yaml
threat_model:
  model_access: gray-box
  attack_goal: untargeted
  deployment_scenario: cloud
  data_sensitivity: high
  training_data_source: mixed
  model_type: transformer
  interface_exposed: api
  threat_categories:
    - data_poisoning
    - adversarial_examples
    - model_stealing
    - membership_inference
    - model_inversion

```

## 6. Discussion and Impact

The threat modeling module plays a central role in the framework by ensuring that the attacks and defenses considered are grounded in a realistic understanding of the system’s risk exposure.

### 6.1 Alignment with Realistic Threats

Explicit definition of the attacker's assumptions and goals helps prevent irrelevant or impractical defenses from being recommended.

### 6.2 Educational Value

The interactive questionnaire increases security literacy by teaching users how architectural and operational choices affect threat exposure.

### 6.3 Modularity and Integration

The threat modeling process is fully modular, producing simple, reusable YAML profiles that integrate easily into experimentation pipelines and version control workflows.

### 6.4 Limitations and Future Improvements

Potential improvements include:

-   Incorporating data-driven refinement of threat suggestions.
    
-   Adding weighted scoring systems for nuanced recommendations.
    
-   Expanding the questionnaire to capture more specific operational contexts.
