# Module 1 – Threat Modeling

## 1. Introduction

As deep learning systems become increasingly deployed in real-world, high-stakes applications, understanding the threats that may compromise their integrity, availability, or confidentiality becomes essential. Module 1 of this framework aims to address this challenge by guiding users through a structured process of **threat modeling**, specifically adapted for deep learning workflows.

Traditional threat modeling techniques used in cybersecurity (e.g., STRIDE) are often too abstract or general to account for the unique characteristics of machine learning systems — such as susceptibility to data poisoning, adversarial perturbations, or model inversion. Therefore, this module proposes a tailored approach that allows users to **explicitly define the attacker assumptions and deployment context** of their model before moving on to attack simulations and defense strategies.

The core deliverable of this module is a `threat profile`, generated through an interactive questionnaire. This profile includes structured information about:
- The model's accessibility (e.g., black-box vs white-box)
- The deployment context (e.g., cloud, mobile)
- The data pipeline (e.g., source and sensitivity of the training data)
- The expected attack goals (e.g., targeted vs untargeted misclassification)
- The likely threat categories to be considered in the next phases

Each of these dimensions was selected based on **evidence and recommendations found across several surveys and studies** on deep learning security, including:
- Liu et al. (2021) – highlighting the importance of access level and data sensitivity in model privacy risks
- Hu & Hu (2020) – identifying the link between training data origin and poisoning/backdoor risks
- Khamaiseh et al. (2022) – discussing the impact of deployment scenario on defense viability
- Sun et al. (2020) – advocating for structured threat assessment before applying defenses

By formalizing this first step, the framework ensures that any user — regardless of experience — can identify the most relevant threats for their use case. This also allows the framework to adapt its behavior dynamically in subsequent modules, such as selecting which attacks to simulate or which defenses to prioritize.

Ultimately, this module provides not only a practical tool to structure the threat landscape of a given project, but also serves an educational purpose by making implicit security assumptions explicit.

## 2. Design of the Threat Questionnaire

The threat modeling process is operationalized through a command-line questionnaire that guides the user step-by-step in defining the most relevant aspects of their model’s deployment and threat landscape. This questionnaire is not only meant to collect structured input, but also to **educate the user during the process** through contextual help and real-world examples.

The questionnaire outputs a standardized YAML file (`threat_profile.yaml`) that encodes all user decisions and can be reused across the remaining modules of the framework.

### 2.1 Guiding Principles

Several principles guided the design of this questionnaire:

- **Clarity and usability**  
  Each question is phrased in accessible language and includes a “Help” option that explains the meaning of the available choices. This allows users with limited background in security to make informed decisions.

- **Modularity and reusability**  
  The questionnaire is implemented in Python using the `questionary` and `pyyaml` libraries. This modular CLI tool can be executed independently, integrated into automation pipelines, or embedded into more complex UI interfaces in the future.

- **Evidence-based structure**  
  The structure and content of the questionnaire are directly informed by the most common threat vectors and defense mechanisms discussed in recent surveys:
  - Liu et al. (2021) and Sun et al. (2020) highlight the value of understanding model access levels and interface types before selecting defenses.
  - Khamaiseh et al. (2022) argue that different deployment environments (e.g., mobile vs cloud) strongly affect the feasibility of adversarial defenses.
  - Hu & Hu (2020) emphasize the role of training data provenance in determining poisoning risks.

- **Dynamic response logic**  
  Rather than asking users to manually select all relevant threats, the framework includes logic to **suggest likely threat categories automatically** based on the answers given. This reduces error and ensures consistency between attacker assumptions and future attack simulations.

Together, these principles ensure that the threat modeling module is both **practically useful and theoretically sound**, forming a solid foundation for the downstream security workflow.

### 2.2 Format and Execution

The threat questionnaire is implemented as a standalone **Python CLI tool**, using the `questionary` library for interactive prompts and `pyyaml` for serializing the collected information into a structured format. This design choice offers several benefits:

- **Cross-platform usability:**  
  The tool runs seamlessly on Windows, macOS, and Linux environments without requiring a graphical interface.

- **Integration with automation pipelines:**  
  Since the questionnaire outputs a machine-readable YAML file, it can easily be integrated into CI/CD pipelines or experiment managers that require threat context before model evaluation.

- **Separation of concerns:**  
  By isolating threat modeling from the rest of the pipeline, the framework respects the principle of modular design and keeps the user workflow clean and focused.

The output profile, stored typically as `profile.yaml`, contains all relevant information about the model's context and anticipated threat surface. This file is **directly consumed by Module 2 (Attack Simulation)**, which uses the selected threat categories to determine which submodules should be activated.

Example command to launch the questionnaire:
```bash
python threat_model_cli.py
```

Example of generated output:

```yaml
threat_model:
  attack_goal: targeted
  data_sensitivity: high
  deployment_scenario: cloud
  interface_exposed: api
  model_access: white-box
  model_type: cnn
  threat_categories:
  - data_poisoning
  training_data_source: internal_clean
```

This modular format makes it easy to:

-   Save multiple threat profiles for different projects
    
-   Share profiles across teams or researchers
    
-   Reuse threat context consistently in simulations, defenses, and evaluations
    

By prioritizing simplicity in execution and standardization in output, this module ensures smooth integration into the broader security lifecycle of deep learning workflows.

## 3. Explanation of Each Field

This section explains each of the fields included in the threat modeling questionnaire. For every field, we justify its inclusion, describe the possible values, and explain how each choice impacts the threat landscape — in terms of both potential attacks and defense mechanisms. References to relevant literature are provided throughout to support each design decision.

---

### 3.1 Field: `model_access`

#### What it asks:
What level of access might a potential attacker have to the model?

#### Why it's included:
The attacker's access level is one of the most fundamental factors in determining which attacks are feasible. It distinguishes between attacks that require introspection of the model's internal components and those that only require input/output access.

#### Possible values:
- `white-box`: Full access to model architecture, parameters, and possibly training data.
- `gray-box`: Partial access (e.g., architecture known, weights unknown).
- `black-box`: Only query access; attacker can submit inputs and observe outputs.

#### Attacks influenced by this field:

| Access Level | Common Attacks Enabled                                       |
|--------------|--------------------------------------------------------------|
| white-box    | Adversarial examples (PGD, Carlini-Wagner), Backdoor insertion, Gradient-based model stealing |
| gray-box     | Transfer attacks, Backdoor analysis, Partial model stealing  |
| black-box    | Model stealing (query-based), Membership inference, Score-based adversarial attacks |

(Refs: Akhtar et al., 2021; Wang et al., 2022; Li et al., 2021)

#### Defenses affected:

| Access Level | Relevant Defenses                                             |
|--------------|---------------------------------------------------------------|
| white-box    | Gradient masking, Defensive distillation, Robust training     |
| gray-box     | Model hardening, Transfer-robust architectures                |
| black-box    | Output obfuscation, Query rate limiting, Confidence suppression |

(Refs: Liu et al., 2021; Sun et al., 2020)

#### Example:
- A publicly deployed API with no published architecture is considered `black-box`.
- A model embedded in an SDK where architecture is known but weights are encrypted could be `gray-box`.
- A model released as open source with weights is `white-box`.

This field is fundamental to controlling the threat scope in subsequent modules and has direct impact on which attacks and defenses are selected.

### 3.2 Field: `attack_goal`

#### What it asks:
What is the likely objective of the attacker when targeting the model?

#### Why it's included:
Different attacks aim for different outcomes. Some attackers may want the model to make a specific mistake (e.g., classify a stop sign as a speed limit sign), while others only need it to make any mistake. This distinction helps in selecting the types of attacks to simulate and the corresponding evaluation metrics.

#### Possible values:
- `targeted`: The attacker wants to cause a specific misclassification.
- `untargeted`: The attacker wants to degrade the model’s performance in general, without requiring a specific incorrect output.

#### Attacks influenced by this field:

| Attack Goal  | Common Attack Examples                                  |
|--------------|----------------------------------------------------------|
| targeted     | Backdoor attacks, Targeted adversarial examples, Clean-label poisoning |
| untargeted   | Label flipping, Untargeted adversarial noise, Universal perturbations |

(Refs: Akhtar et al., 2021; Hu & Hu, 2020; Costa et al., 2024)

#### Defenses affected:

| Goal Type    | Relevant Defense Strategies                              |
|--------------|-----------------------------------------------------------|
| targeted     | Trigger detection (e.g., STRIP), Class-specific robust training, Pruning |
| untargeted   | Adversarial training, Label smoothing, Randomized input defenses |

(Refs: Sun et al., 2020; Li et al., 2021)

#### Example:
- In a facial recognition system, an attacker who wants to impersonate a specific person is launching a targeted attack.
- In an online moderation tool, an attacker who wants to reduce model accuracy in general is performing an untargeted attack.

This field informs which submodules in attack simulation will be configured and which evaluation metrics will prioritize precision vs. global robustness.

### 3.3 Field: `deployment_scenario`

#### What it asks:
In what type of environment will the model be deployed?

#### Why it's included:
The deployment scenario determines the available computational resources, the attack surface, and the feasibility of deploying certain defenses. It also influences the attacker's capabilities (e.g., physical access, network-based interaction) and the level of security hardening required.

#### Possible values:
- `cloud`: The model runs on remote servers or cloud platforms, typically accessed via API.
- `edge`: The model runs on devices close to data sources (e.g., surveillance cameras, sensors).
- `mobile`: The model runs on smartphones or tablets, either locally or via embedded APIs.
- `api_public`: The model is accessible through a publicly exposed API.
- `on_device`: The model is embedded in hardware or software without direct user-facing interfaces.

#### Attacks influenced by this field:

| Scenario     | Attack Vectors Enabled                                              |
|--------------|---------------------------------------------------------------------|
| cloud        | API abuse, Model stealing, Membership inference, Query-based attacks |
| edge         | Physical tampering, Adversarial examples, Evasion attacks           |
| mobile       | Local adversarial examples, Model extraction from memory or binaries |
| api_public   | High-frequency model stealing, Automated enumeration attacks        |
| on_device    | Side-channel attacks, Offline model extraction (less common)        |

(Refs: Liu et al., 2021; Khamaiseh et al., 2022; Wang et al., 2022)

#### Defenses affected:

| Scenario     | Recommended Defenses                                                |
|--------------|----------------------------------------------------------------------|
| cloud        | Logging, Watermarking, Output obfuscation, Query rate limiting       |
| edge         | Lightweight defenses (e.g., feature squeezing, quantization)         |
| mobile       | Input pre-processing, Secure packaging, Obfuscated weights           |
| api_public   | Confidence thresholding, Output filtering, Robust architecture design |
| on_device    | Hardening, Static analysis protection, Privacy-preserving inference  |

(Refs: Sun et al., 2020; Li et al., 2024; Peng et al., 2024)

#### Example:
- A medical diagnostic model running in a hospital server is considered `cloud`.
- A traffic sign classifier in a smart camera is `edge`.
- A speech recognition model running in a smartphone app is `mobile`.

This field is critical for understanding the trade-offs between robustness, latency, and privacy, and is used throughout the framework to guide defense viability decisions.

### 3.4 Field: `data_sensitivity`

#### What it asks:
How sensitive or confidential is the training data used for the model?

#### Why it's included:
The sensitivity of the training data determines the **potential consequences of privacy-related attacks** such as model inversion, membership inference, or data leakage. It also affects regulatory compliance requirements and the need for privacy-preserving defenses.

#### Possible values:
- `high`: Data contains sensitive personal, biometric, medical, or confidential information.
- `medium`: Data includes contextual or semi-personal information that could still raise concerns.
- `low`: Data is non-sensitive, public, or synthetic.

#### Attacks influenced by this field:

| Sensitivity Level | Associated Attack Risks                                         |
|-------------------|-----------------------------------------------------------------|
| high              | Membership inference, Model inversion, Data leakage             |
| medium            | Weak inference attacks, Re-identification, Cross-model transfer |
| low               | Minimal privacy risk, but still susceptible to model stealing   |

(Refs: Liu et al., 2021; Hu & Hu, 2020)

#### Defenses affected:

| Sensitivity Level | Recommended Defenses                                            |
|-------------------|------------------------------------------------------------------|
| high              | Differential Privacy (DP), Output regularization, Access control |
| medium            | Pruning, Data anonymization, Limited output disclosure          |
| low               | Basic obfuscation, No specific privacy-preserving mechanisms     |

(Refs: Sun et al., 2020; Li et al., 2021)

#### Example:
- A facial recognition model trained with biometric data is considered `high` sensitivity.
- A model trained on user interaction logs may be `medium`.
- A model trained on publicly available images of traffic signs is `low`.

Understanding this field allows the framework to determine whether additional privacy constraints should be enforced during training or inference.

### 3.5 Field: `training_data_source`

#### What it asks:
Where does the training data come from?

#### Why it's included:
The origin of the training data is directly tied to the **risk of poisoning and backdoor attacks**. Datasets collected from external or user-submitted sources are more susceptible to manipulation. This field also helps define the attacker’s position in the pipeline — whether they can poison data before or during training.

#### Possible values:
- `internal_clean`: Fully controlled and curated dataset collected internally.
- `external_public`: Public datasets available online (e.g., ImageNet, CIFAR, COCO).
- `user_generated`: Data submitted by users or third parties (e.g., app uploads, crowd-labeling).
- `mixed`: A combination of the above sources.

#### Attacks influenced by this field:

| Source Type      | Primary Threats                                                  |
|------------------|------------------------------------------------------------------|
| internal_clean   | Lower poisoning risk, but privacy attacks still possible         |
| external_public  | Poisoned samples, backdoors, mislabeled data                     |
| user_generated   | Continuous poisoning, trigger injection, clean-label attacks     |
| mixed            | Aggregated risks from multiple sources                           |

(Refs: Hu & Hu, 2020; Liu et al., 2021; Khamaiseh et al., 2022)

#### Defenses affected:

| Source Type      | Recommended Defenses                                              |
|------------------|-------------------------------------------------------------------|
| internal_clean   | Regularization, Membership inference mitigation                   |
| external_public  | Data sanitization, Pre-training inspection, Trigger analysis      |
| user_generated   | Online validation, STRIP detection, Label verification            |
| mixed            | Source isolation, Confidence-based filtering, Hybrid approaches   |

(Refs: Sun et al., 2020; Akhtar et al., 2021)

#### Example:
- A self-collected dataset used in a closed research lab is `internal_clean`.
- ImageNet or CIFAR-10 are typical examples of `external_public`.
- A voice assistant trained with user-submitted audio is `user_generated`.
- A commercial product using public data plus app user uploads is `mixed`.

Correctly identifying the data source helps the framework decide which types of poisoning and integrity threats are plausible and which mitigation strategies are appropriate.

### 3.6 Field: `model_type`

#### What it asks:
What type of neural network architecture will the model use?

#### Why it's included:
The architecture type influences the **attack surface and the effectiveness of certain defenses**. Some attacks exploit structural properties specific to convolutional networks or transformers. Similarly, some defenses are designed to target specific model families. This field also helps submodules optimize attack and defense parameters.

#### Possible values:
- `cnn`: Convolutional Neural Networks, typically used in image classification.
- `transformer`: Transformer-based architectures (e.g., ViT, BERT, DETR).
- `mlp`: Multi-Layer Perceptrons (fully connected networks).
- `other`: Any architecture outside the scope of the above, such as RNNs or GNNs.

#### Attacks influenced by this field:

| Model Type   | Common Attack Vectors                                                  |
|--------------|------------------------------------------------------------------------|
| cnn          | Adversarial examples (FGSM, PGD), Backdoors (trigger localization)     |
| transformer  | Transferability attacks, Feature inversion, Gradient masking bypass    |
| mlp          | Gradient-based attacks, Label flipping, Weight perturbations           |
| other        | Architecture-specific vulnerabilities (e.g., time-series poisoning)    |

(Refs: Akhtar et al., 2021; Costa et al., 2024; Li et al., 2021)

#### Defenses affected:

| Model Type   | Recommended Defenses                                                   |
|--------------|------------------------------------------------------------------------|
| cnn          | Feature squeezing, Adversarial training, STRIP for backdoor detection  |
| transformer  | Token perturbation defenses, Gradient regularization, Attention masking|
| mlp          | Output smoothing, Defensive distillation                               |
| other        | Depends on domain-specific adaptations                                 |

(Refs: Sun et al., 2020; Li et al., 2024)

#### Example:
- A ResNet-based image classifier is a `cnn`.
- A ViT or BERT-like model used for image or text is a `transformer`.
- A basic fully connected classifier with no spatial awareness is an `mlp`.

While this field does not alone determine the threat landscape, it refines the selection of attack parameters and the suitability of available defenses.

### 3.7 Field: `interface_exposed`

#### What it asks:
How is the model exposed or accessed by external users or systems?

#### Why it's included:
The interface through which a model is accessed determines **how attackers interact with it**, what kind of queries they can perform, and whether they can exploit its outputs. This field is essential for assessing **API-based threats**, including model stealing, inference attacks, and abuse of access frequency.

#### Possible values:
- `api`: The model is accessed via a remote or public-facing API.
- `local_app`: The model runs inside a standalone application (e.g., mobile or desktop).
- `sdk`: The model is embedded into a software development kit distributed to third parties.
- `none`: The model is fully embedded or internal, with no external interface.

#### Attacks influenced by this field:

| Interface Type | Threat Vectors                                                        |
|----------------|------------------------------------------------------------------------|
| api            | Model stealing (Knockoff Nets), Membership inference, Query probing   |
| local_app      | Adversarial input crafting, Memory extraction, Model inversion        |
| sdk            | Offline model cloning, Reverse engineering, Binary analysis           |
| none           | Physical attacks, Side-channel attacks (limited remote exposure)      |

(Refs: Liu et al., 2021; Wang et al., 2022; Li et al., 2021)

#### Defenses affected:

| Interface Type | Suitable Defenses                                                      |
|----------------|------------------------------------------------------------------------|
| api            | Output obfuscation, Rate limiting, Confidence suppression, Watermarking|
| local_app      | Input sanitization, Model encryption, Lightweight runtime detectors     |
| sdk            | Obfuscated weights, API key checks, Post-deployment pruning            |
| none           | Minimal runtime defenses; emphasis on training-phase robustness         |

(Refs: Sun et al., 2020; Akhtar et al., 2021)

#### Example:
- A vision model served via a REST API on the cloud uses `api`.
- A desktop application that runs inference locally is `local_app`.
- A pre-trained model embedded in a mobile SDK is `sdk`.
- A model running entirely on a closed embedded system is `none`.

This field is critical for understanding the **attack surface** and for defining where runtime monitoring or query-based defenses are necessary.

### 3.8 Field: `threat_categories`

#### What it asks:
Which categories of threats are relevant to this project or system?

#### Why it's included:
This field serves as the **bridge between threat modeling and attack simulation**. Based on the user's answers to previous questions, the framework suggests a list of likely threat categories. The user can then confirm or edit this list manually. This controls which attack submodules are activated in Module 2 and which defenses are prioritized in later stages.

#### Possible values:
This is a multi-selection field. Each value represents a distinct threat category:

- `data_poisoning`: Insertion of malicious data into the training set to compromise model behavior.
- `backdoor_attacks`: Implanting hidden triggers that cause misclassification only under specific conditions.
- `adversarial_examples`: Inputs crafted at inference time to manipulate predictions.
- `model_stealing`: Cloning a model by querying it and analyzing outputs.
- `membership_inference`: Determining if a specific input was part of the training data.
- `model_inversion`: Reconstructing training data based on model predictions.

#### How values are suggested:
A set of heuristic rules is applied based on earlier responses. For example:

- Using external or user-generated training data → `data_poisoning`
- Public API exposure → `model_stealing`, `membership_inference`
- High data sensitivity → `model_inversion`
- Edge/mobile deployment → `adversarial_examples`

These suggestions can be overridden by the user during the questionnaire.

(Refs: Hu & Hu, 2020; Liu et al., 2021; Wang et al., 2022; Sun et al., 2020)

#### Attacks influenced:
This field directly activates submodules in Module 2. Each selected threat triggers corresponding simulations.

#### Defenses affected:
Downstream modules (Module 4 and 5) filter recommended defenses and robustness evaluations based on selected threats.

#### Example:
- If a user selects `data_poisoning` and `adversarial_examples`, only those attacks will be simulated, and defenses will be tailored accordingly.

This field ensures **customization and modularity** in the framework, avoiding unnecessary computation and focusing efforts on threats that are realistically relevant to the use case.

## 4. Automatic Threat Suggestion Logic

To support users who may not have deep experience in security, the threat modeling module includes a mechanism to automatically suggest relevant `threat_categories` based on the user’s answers to earlier questions. This feature improves usability, ensures alignment between system context and attack surface, and reduces the likelihood of incorrect or incomplete threat selection.

### 4.1 Purpose

The automatic suggestion logic is intended to:
- Translate architectural and contextual information into actionable security concerns.
- Pre-select threat categories that are logically consistent with the provided information.
- Serve as a baseline that users can accept or manually adjust.

This balances flexibility with guidance, helping novice users without restricting expert users.

### 4.2 Rules Used in Suggestion

The logic relies on a small set of deterministic rules. These rules were derived from findings in multiple survey papers that link deployment patterns and threat vectors.

#### Rule Examples:

| Condition                                                 | Suggested Threats                   | Justification                                               |
|-----------------------------------------------------------|-------------------------------------|-------------------------------------------------------------|
| `training_data_source` ∈ [external_public, user_generated]| `data_poisoning`                   | Hu & Hu (2020): increased risk of poisoning/backdoors       |
| `interface_exposed` = api                                 | `model_stealing`, `membership_inference` | Liu et al. (2021); Wang et al. (2022): query-based attacks |
| `deployment_scenario` ∈ [mobile, edge, api_public]        | `adversarial_examples`             | Khamaiseh et al. (2022): edge devices are vulnerable        |
| `model_access` ∈ [white-box, gray-box]                    | `backdoor_attacks`                 | Li et al. (2021): white-box access enables stealthy triggers|
| `data_sensitivity` = high                                 | `model_inversion`                  | Liu et al. (2021): inversion attacks leak sensitive inputs  |

The logic is implemented in Python as simple `if`-statements, making it easy to audit and extend.

### 4.3 Editable Suggestions

After the threat categories are suggested, users are prompted to confirm or modify them. This step ensures:
- Transparency of the recommendation process
- Customization for edge cases not covered by rules
- Increased user engagement with the threat modeling process

### 4.4 Integration in the Framework

The selected or confirmed list of threat categories is included in the YAML profile. This profile is consumed directly by the Attack Simulation module (Module 2), which uses it to activate only the necessary submodules. The same list also informs defense suggestions and robustness benchmarks in Modules 4 and 5.

This approach ensures coherence across modules while preserving user control over threat definition.

## 5. Example Profiles

To illustrate how the threat modeling module works in practice, two complete example profiles are provided below. These profiles were generated based on realistic deployment scenarios and reflect different security contexts.

Each profile demonstrates how the values selected in the questionnaire lead to distinct threat categories and influence downstream security decisions.

---

### 5.1 Mobile Application Profile

This profile represents a mobile application that uses a convolutional neural network (CNN) for tasks such as image classification or object detection, with a public dataset and embedded SDK deployment.

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

**Explanation:**

-   The model is embedded in a mobile SDK, limiting user access to outputs only (`black-box`).
    
-   Public datasets increase the likelihood of `data_poisoning`.
    
-   Mobile environments are prone to adversarial input manipulation.
    
-   The use of an SDK raises concerns about offline reverse engineering and unauthorized usage.
    

(Refs: Akhtar et al., 2021; Liu et al., 2021; Khamaiseh et al., 2022)

----------

### 5.2 Cloud API Profile

This profile describes a large model deployed via a public API, possibly transformer-based, trained on mixed data sources, and used in a sensitive context such as biometrics or healthcare.

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

**Explanation:**

-   A gray-box setting is assumed due to potential architecture disclosure via publications or documentation.
    
-   API-based access opens the door to `model_stealing` and `membership inference`.
    
-   The high sensitivity of training data suggests the need for `model_inversion` defenses.
    
-   The mixed data source includes public and user-contributed content, justifying `data_poisoning` concerns.
    

(Refs: Liu et al., 2021; Hu & Hu, 2020; Wang et al., 2022; Sun et al., 2020)

----------

These profiles can be used:

-   As templates for quick setup of the framework
    
-   As case studies for evaluation in the thesis
    
-   For benchmarking the system under different threat conditions
   
## 6. Discussion and Impact

The threat modeling module serves as the foundation of the entire security framework. By collecting structured, justified information about the model's context, it enables the rest of the system to adapt intelligently and consistently. Its impact is both practical and methodological.

### 6.1 Alignment with Realistic Threats

One of the main challenges in deep learning security is the mismatch between generic defenses and real-world threats. This module mitigates that problem by:

- Forcing the user to explicitly define attacker capabilities and goals.
- Identifying threat categories that are contextually appropriate.
- Avoiding unrealistic or redundant defenses in later stages.

This approach is grounded in the recommendations found in multiple surveys, including Liu et al. (2021), Sun et al. (2020), and Akhtar et al. (2021), which all stress the need for scenario-aware security mechanisms.

### 6.2 Educational Value

Beyond its practical use, the questionnaire has an educational role. By providing “Help” explanations for each field and suggesting threats based on clear rules, it guides users to understand:

- What types of threats exist
- How they relate to system architecture and data flow
- Which conditions increase exposure to specific attacks

This increases security awareness, especially in development teams without dedicated security experts.

### 6.3 Modularity and Integration

The module is fully decoupled from other stages of the framework. Its output is a simple YAML file, which:

- Can be reused across experiments
- Can be stored in version control
- Can be generated manually or programmatically

This flexibility makes it suitable for integration into pipelines, experiment tracking systems, or external tools.

### 6.4 Limitations and Future Improvements

While the current ruleset for automatic threat suggestion is effective, it is static and manually curated. Possible future improvements include:

- Learning threat mappings from historical data
- Using scoring or weighting systems for more nuanced suggestions
- Extending the questionnaire with more deployment-specific parameters

Such enhancements could increase accuracy and adaptivity while maintaining the module’s interpretability.

---

By clearly connecting system characteristics to threat categories, this module empowers the user to build not only more secure models, but also more transparent and auditable machine learning pipelines.


## 7. References

The following works were directly used to support the structure, logic, and content of the threat modeling module. Each field in the questionnaire and its implications for attacks and defenses are grounded in the insights and recommendations from these surveys and studies.

- Akhtar, N., et al. (2021). *Advances in Adversarial Attacks and Defenses in Computer Vision: A Survey*.  
  Referenced for model-type-specific attacks and defenses (CNN, backdoors, adversarial robustness).

- Costa, L., et al. (2024). *How Deep Learning Sees the World: A Survey on Adversarial Attacks & Defenses*.  
  Used to distinguish between attack goals (targeted vs. untargeted) and their corresponding defense strategies.

- Hu, W., & Hu, L. (2020). *Data Poisoning on Deep Learning Models*.  
  Core source for threats related to training data origin, including poisoning and backdoor risks from public and user-submitted datasets.

- Khamaiseh, M., et al. (2022). *Adversarial Deep Learning: A Survey on Adversarial Attacks and Defense Mechanisms on Image Classification*.  
  Supported the inclusion of deployment scenario as a determining factor in attack feasibility and defense viability.

- Li, Y., et al. (2021). *Adversarial Attacks and Defenses for Deep Learning Models*.  
  Referenced for relationships between model access levels and applicable attacks.

- Liu, W., et al. (2021). *Privacy and Security Issues in Deep Learning: A Survey*.  
  One of the most comprehensive sources used to justify the inclusion of model access, interface exposure, and data sensitivity fields.

- Peng, T., et al. (2024). *A Survey of Security Protection Methods for Deep Learning Models*.  
  Used for motivation behind threat modeling and for contextual defense mapping based on deployment type.

- Sun, Y., et al. (2020). *Complete Defense Framework to Protect Deep Neural Networks against Adversarial Examples*.  
  Source for defense modularity, pipeline integration, and structured threat assessment practices.

- Wang, X., et al. (2022). *Black-Box Adversarial Attacks on Deep Neural Networks: A Survey*.  
  Cited to support concerns around API exposure and model stealing in black-box or public interface settings.

These references ensure that the module is not only practical but also aligned with state-of-the-art knowledge in adversarial machine learning and deep learning security.

