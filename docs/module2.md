# Module 2: Attack Simulation

## Table of Contents

- [Module 2: Attack Simulation](#module-2-attack-simulation)
   * [1. What is Module 2: Attack Simulation?](#1-what-is-module-2-attack-simulation)
   * [2. Workflow Overview](#2-workflow-overview)
      + [Step 1: Dataset and Model Selection](#step-1-dataset-and-model-selection)
      + [Step 2: Threat Profile Selection](#step-2-threat-profile-selection)
      + [Step 3: Submodule Execution and Setup](#step-3-submodule-execution-and-setup)
      + [Step 4: Clean Baseline Evaluation](#step-4-clean-baseline-evaluation)
      + [Step 5: Attack Execution](#step-5-attack-execution)
      + [Step 6: Reporting](#step-6-reporting)
   * [3. Dataset and Model Customization Rules](#3-dataset-and-model-customization-rules)
      + [3.1 Custom Datasets (`user_dataset.py`)](#31-custom-datasets-user_datasetpy)
      + [3.2 Custom Models (`user_model.py`)](#32-custom-models-user_modelpy)
      + [3.3 Built-In Compatibility](#33-built-in-compatibility)
   * [4. The Role of the Profile YAML](#4-the-role-of-the-profile-yaml)
      + [4.1 What Is Stored in the Profile?](#41-what-is-stored-in-the-profile)
      + [4.2 Updating the Profile from the Setup Wizard](#42-updating-the-profile-from-the-setup-wizard)
      + [4.3 Reusability and Reproducibility](#43-reusability-and-reproducibility)
   * [5. Attack Simulation Flow](#5-attack-simulation-flow)
      + [5.1 Overview of the Execution Pipeline](#51-overview-of-the-execution-pipeline)
      + [5.2 Modular Architecture](#52-modular-architecture)
      + [5.3 Clean Execution Environment](#53-clean-execution-environment)
   * [6. Built-in and Custom Components](#6-built-in-and-custom-components)
      + [6.1 Built-in Datasets](#61-built-in-datasets)
      + [6.2 Built-in Models](#62-built-in-models)
      + [6.3 Custom Datasets](#63-custom-datasets)
      + [6.4 Custom Models](#64-custom-models)
      + [6.5 Summary](#65-summary)
   * [7. Profile YAML Specification](#7-profile-yaml-specification)
      + [7.1 Profile Structure](#71-profile-structure)
      + [7.2 Fields Explained](#72-fields-explained)
      + [7.3 Updating the Profile](#73-updating-the-profile)
      + [7.4 Benefits](#74-benefits)
   * [8. Execution Flow](#8-execution-flow)
      + [8.1 High-Level Steps](#81-high-level-steps)
      + [8.2 Summary Diagram](#82-summary-diagram)
      + [8.3 Automatic Decisions](#83-automatic-decisions)
      + [8.4 Notes on Reusability](#84-notes-on-reusability)
   * [9. Reporting and Outputs](#9-reporting-and-outputs)
      + [9.1 Output Directory](#91-output-directory)
      + [9.2 `baseline_accuracy.json`](#92-baseline_accuracyjson)
      + [9.3 Attack-Specific Reports](#93-attack-specific-reports)
      + [9.4 Visual Examples Folder](#94-visual-examples-folder)
      + [9.5 Customization Notes](#95-customization-notes)
   * [10. Submodule 2.1 — Data Poisoning Attacks](#10-submodule-21-data-poisoning-attacks)
      + [10.1 Overview](#101-overview)
      + [10.2 Configuration via Setup Script](#102-configuration-via-setup-script)
      + [10.3 Label Flipping Attack](#103-label-flipping-attack)
         - [10.3.1 Objective](#1031-objective)
         - [10.3.2 Available Strategies](#1032-available-strategies)
         - [10.3.3 Configuration Parameters](#1033-configuration-parameters)
         - [10.3.4 YAML Integration](#1034-yaml-integration)
         - [10.3.5 Reporting and Metrics](#1035-reporting-and-metrics)
         - [10.3.6 Design Considerations](#1036-design-considerations)
      + [10.4 Clean Label Poisoning](#104-clean-label-poisoning)
         - [10.4.1 Objective](#1041-objective)
         - [10.4.2 Perturbation Methods](#1042-perturbation-methods)
         - [10.4.3 Configuration via Setup Script](#1043-configuration-via-setup-script)
         - [10.4.4 YAML Integration](#1044-yaml-integration)
         - [10.4.5 Reporting and Metrics](#1045-reporting-and-metrics)
         - [10.4.6 Design Considerations](#1046-design-considerations)

## 1. What is Module 2: Attack Simulation?

Module 2 of the Safe-DL framework is responsible for simulating adversarial attacks against deep learning models in controlled environments. Its primary objective is to evaluate the model’s robustness when exposed to malicious manipulations — including input perturbations and training data poisoning.

This module operates based on the threat profile defined in Module 1. After selecting a dataset and model, Module 2 activates only the submodules relevant to the selected threat categories (e.g., data poisoning, backdoor attacks, model inversion).

Each submodule is responsible for:

-   Configuring its attack parameters via an interactive setup,
    
-   Modifying the training pipeline accordingly,
    
-   Evaluating the model post-attack,
    
-   And producing detailed reports in `.json` and `.md` formats.
    

Currently implemented attack types under the **Data Poisoning** category include:

-   **Label Flipping**: Random or targeted manipulation of training labels.
    
-   **Clean Label Poisoning**: Subtle perturbations of correctly labeled samples to cause misclassification.
    

The ultimate goals of Module 2 are to:

-   Measure how attacks degrade model accuracy (overall and per class),
    
-   Expose model weaknesses under realistic adversarial conditions,
    
-   Assist users in improving robustness through empirical insights.
    

The module is fully interactive, modular, and extensible.

## 2. Workflow Overview

The attack simulation workflow in Module 2 is structured to be both flexible and rigorous. Below is an overview of each step involved in the process:

### Step 1: Dataset and Model Selection

At the beginning of Module 2, the user is prompted to select a dataset and a model architecture. The framework offers a collection of built-in datasets and models, as well as support for user-defined custom options.

- **Built-in Datasets:**
  - MNIST
  - FashionMNIST
  - KMNIST
  - CIFAR-10
  - CIFAR-100
  - SVHN
  - EMNIST

- **Built-in Models:**
  - Convolutional Neural Network (CNN)
  - Multilayer Perceptron (MLP)
  - ResNet-18
  - ResNet-50
  - ViT (Vision Transformer)

- **Custom Datasets and Models:**
  - Users can define their own datasets in `user_dataset.py` and their own models in `user_model.py`.
  - Custom datasets must implement the function `get_dataset()` and return `(trainset, testset, valset, class_names, num_classes)` following the standard format.
  - Custom models must implement the function `get_model()` and return a valid PyTorch `nn.Module`.

### Step 2: Threat Profile Selection

The user selects a threat profile generated in Module 1. This profile, saved in a YAML file, contains critical metadata such as the attack goal (targeted or untargeted), training data origin, threat categories, and model access level.

The selected profile guides which attack submodules will be executed.

### Step 3: Submodule Execution and Setup

For each attack submodule that corresponds to a threat listed in the profile, the framework:
- Launches a setup questionnaire to customize the attack configuration.
- Suggests recommended values (e.g., flip rate, attack strategy) based on the threat scenario.
- Allows the user to accept or modify these values interactively.
- Saves all parameters into the same `.yaml` profile to maintain a unified configuration.

This ensures full transparency and reproducibility of the experiments.

### Step 4: Clean Baseline Evaluation

Before running the attack, the selected model is trained on the clean (non-poisoned) training data. This provides a baseline accuracy value, including:
- **Overall accuracy**
- **Per-class accuracy** (e.g., performance on specific classes like "cat" or "airplane")

This is crucial to later measure the degradation in performance caused by the attack.

### Step 5: Attack Execution

The specific submodule performs the configured attack, modifies the training data accordingly, and re-trains the model using the poisoned dataset. 

After the attack:
- The model is evaluated again on the clean test set.
- Metrics are computed and compared to the baseline.

### Step 6: Reporting

Two types of reports are automatically generated:
- **JSON Report:** Contains structured data with detailed metrics, such as flip counts, per-class accuracy, attack parameters, and sample indices.
- **Markdown Report (.md):** A human-readable version of the report that includes tables, attack summaries, and visual examples of poisoned inputs.

Reports are saved under the `results/` directory and include visual flip samples when applicable.

## 3. Dataset and Model Customization Rules

To support flexibility and extensibility, Module 2 allows users to define their own datasets and models. However, to ensure compatibility with the attack framework, all custom implementations must follow strict conventions described below.

----------

### 3.1 Custom Datasets (`user_dataset.py`)

To use a custom dataset, create a file named `user_dataset.py` that includes the function `get_dataset()`. This function must return:

```python
(trainset, testset, valset, class_names, num_classes)

```

Where:

-   `trainset`, `testset`, `valset`: Should be either:
    
    -   `torch.utils.data.Subset` created from a `torchvision.datasets.Dataset`, or
        
    -   a custom `torch.utils.data.Dataset` object that exposes:
        
        -   `dataset.targets` (a list or tensor of integer class labels),
            
        -   `dataset.data` or `dataset.tensors` (the actual image data),
            
        -   and supports indexing (`__getitem__`).
            
-   `class_names`: A list of human-readable class labels (e.g., `["cat", "dog", "truck"]`).
    
-   `num_classes`: An integer representing the total number of classes.
    

#### Example:

```python
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split

def get_dataset():
    transform = transforms.ToTensor()
    full_train = MNIST(root=\"./data\", train=True, download=True, transform=transform)
    testset = MNIST(root=\"./data\", train=False, download=True, transform=transform)

    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    trainset, valset = random_split(full_train, [train_size, val_size])

    class_names = [str(i) for i in range(10)]
    num_classes = 10
    return trainset, testset, valset, class_names, num_classes

```

> **Important:** If you return a `Subset` object, ensure the underlying dataset exposes:
	> -   `dataset.targets` → Required for selecting poisoning candidates in attacks like label flipping and clean label. 
	>-   `dataset.classes` or at least `dataset.targets` and known class ordering.
	
>These attributes are used for selecting classes and generating logs and reports. Built-in datasets like `MNIST`, `CIFAR10`, and `FashionMNIST` are already compatible.

---------

### 3.2 Custom Models (`user_model.py`)

To define a custom model architecture, create a file named `user_model.py` that includes a function `get_model()` which returns a PyTorch `nn.Module`.

#### Example:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

    # (Optional but recommended)
    def features(self, x):
        x = self.net[0](x)  # Flatten
        x = self.net[1](x)  # First Linear
        return self.net[2](x)  # ReLU

def get_model():
    return MyModel()

```

Your model should be compatible with the chosen dataset input shape and output the correct number of classes. For example, a grayscale image input from MNIST would require shape `(1, 28, 28)` and an output of `10` classes.
If you want to support **clean-label attacks using feature collision**, your model should also implement a method:

```python

def  features(self, x: torch.Tensor) -> torch.Tensor:
    ...
   ```

This method should return the internal feature representation used before the final classification head. This is crucial for optimizing poisoned samples to mimic the representation of target images.
___

### 3.3 Built-In Compatibility

When using built-in datasets and models:

-   The input shape is automatically inferred (but can be customized).
    
-   The user may adjust built-in model parameters such as:
    
    -   **CNN:** number of convolution filters, hidden layer size.
        
    -   **MLP:** input size, hidden size.
        
-   These parameters are set during the initial setup and saved in the profile YAML.
    

This makes it easy to switch between experiments and maintain full reproducibility.

## 4. The Role of the Profile YAML

Throughout the Safe-DL framework, the profile YAML file plays a central role in coordinating the configuration, customization, and execution of attack simulations. This file is first generated during Module 1 (Threat Modeling) and then updated by Module 2 during the attack setup phase.

### 4.1 What Is Stored in the Profile?

The profile file (e.g., `cloud_api_profile.yaml`) is a structured YAML file that contains all necessary information to:

- Identify the **deployment scenario** and **threats** (from Module 1).
- Select and configure the **dataset** to be used.
- Specify the **model architecture** and its parameters.
- Define **attack-specific overrides** like flipping strategies and hyperparameters.

Below is an example snippet of a profile YAML:

```yaml
dataset:
  name: cifar10
  type: builtin

model:
  name: cnn
  type: builtin
  num_classes: 10
  input_shape: [3, 32, 32]
  params:
    conv_filters: 32
    hidden_size: 128

threat_model:
  attack_goal: untargeted
  model_access: gray-box
  training_data_source: mixed
  threat_categories:
    - data_poisoning
    - model_stealing

attack_overrides:
  data_poisoning:
    label_flipping:
	  strategy: many_to_one
	  flip_rate: 0.08
	  target_class: 2

```

----------

### 4.2 Updating the Profile from the Setup Wizard

When the user runs `setup_module2.py`, the framework:

1.  **Prompts the user** to select a dataset and model.
    
2.  **Suggests an attack configuration** based on the threat model.
    
3.  **Allows the user to customize parameters**.
    
4.  **Updates the profile YAML** in-place with the new configuration.
    

This makes the profile a **single source of truth** for running repeatable experiments.

----------

### 4.3 Reusability and Reproducibility

Once a profile is configured, it can be reused across experiments without re-running the questionnaire. The user can:

-   Re-execute Module 2 with the same profile.
    
-   Clone the profile and modify it for different experiments.
    
-   Store results in a structured and comparable way.
    

**Important:** Any time the attack configuration or dataset/model is updated via the setup wizard, the new state is immediately saved to the profile YAML — ensuring consistency across runs.

## 5. Attack Simulation Flow

Once the profile YAML is fully configured, the attack simulation phase begins. This phase orchestrates the training, evaluation, and execution of selected attacks based on the profile information. It is triggered by running the script `run_attacks.py`.

### 5.1 Overview of the Execution Pipeline

The simulation follows this high-level sequence:

1. **Load the Profile**
   - The user selects a previously configured profile from the `../profiles/` directory.
   - The profile is parsed to extract all relevant information: dataset, model, and threat model configuration.

2. **Load and Prepare the Dataset**
   - The selected dataset (built-in or custom) is loaded.
   - The training set is split into training and validation sets (if not already split).
   - The framework extracts metadata such as the number of classes and class names.

3. **Instantiate the Model**
   - The model is loaded and constructed with the appropriate architecture, number of output classes, and any customized parameters (e.g., number of layers, input size).
   - If the model is a user-defined one, it is dynamically loaded from `user_model.py`.

4. **Baseline Training and Evaluation**
   - Before applying any attacks, the model is trained on the clean training data.
   - A baseline accuracy is computed on the untouched test set.
   - Results are saved in `results/baseline_accuracy.json` and printed to the console.
   - Example:
     ```json
     {
       "accuracy": 0.581,
       "per_class_accuracy": {
         "airplane": 0.573,
         "automobile": 0.812,
         ...
       }
     }
     ```

5. **Attack Selection and Execution**
   - The framework reads the `threat_categories` defined in the profile and determines which submodules to run.
   - For each selected attack submodule (e.g., Label Flipping, Clean Label Poisoning), a dedicated attack script is executed using the configuration under `attack_overrides`.
   - Before running each attack, a clean version of the model is instantiated from the profile settings to ensure that results are not contaminated by previously applied attacks. This allows the framework to isolate the effects of each attack and evaluate their specific impact.

6. **Poisoned Model Training and Evaluation**
   - A copy of the model is trained on the poisoned dataset generated by the attack.
   - Post-attack accuracy is measured on the clean test set.
   - Per-class accuracies are also computed for detailed impact analysis, especially useful in targeted attacks.

7. **Report Generation**
   - Results are saved in:
     - `results/data_poisoning_metrics.json`: contains all key metrics and flip logs.
     - `results/data_poisoning_report.md`: human-readable report including summary tables and visual examples.
   - Visualizations of flipped samples are stored in `results/flipped_examples/`.

---

### 5.2 Modular Architecture

Each submodule is fully isolated, allowing easy extension with new types of attacks (e.g., Clean Label, Backdoors). Every submodule is responsible for:

- Loading its own configuration from the profile.
- Running a dedicated setup phase (interactive if needed).
- Executing the attack logic.
- Producing structured outputs.

This modularity ensures the framework is scalable and maintainable.

---

### 5.3 Clean Execution Environment

Each run:

- Clears previous visualizations and results to avoid clutter.
- Stores all metrics and reports under the `results/` folder.
- Maintains reproducibility by locking all parameters inside the profile file.

## 6. Built-in and Custom Components

The Safe-DL Framework provides flexibility by supporting both built-in and user-defined datasets and models. During the Module 2 setup phase, users can choose from a predefined set of components or supply their own implementations. This section outlines what is currently available and the requirements for adding custom components.

---

### 6.1 Built-in Datasets

The following datasets are supported natively within the framework and can be selected during setup:

| Dataset      | Channels | Size  | Classes    |
|--------------|----------|-------|------------|
| MNIST        | 1        | 28x28 | 10 digits  |
| FashionMNIST | 1        | 28x28 | 10 apparel |
| KMNIST       | 1        | 28x28 | 10 kana    |
| CIFAR-10     | 3        | 32x32 | 10 objects |
| CIFAR-100    | 3        | 32x32 | 100 objects|
| SVHN         | 3        | 32x32 | 10 digits  |
| EMNIST       | 1        | 28x28 | 47 classes |

These datasets are automatically downloaded from torchvision and split into training, validation, and test sets. For each dataset, the framework also extracts the number of classes and their corresponding names (if available).

---

### 6.2 Built-in Models

Users can select one of the following model architectures during setup:

| Model      | Type      | Configurable Parameters                  |
|------------|-----------|------------------------------------------|
| CNN        | ConvNet   | `conv_filters`, `hidden_size`            |
| MLP        | Feedforward | `hidden_size`, `input_size`           |
| ResNet-18  | ResNet    | None (standard architecture)             |
| ResNet-50  | ResNet    | None                                     |
| ViT        | Vision Transformer | None                            |

**Input shape configuration:**  
Users can customize the input shape (channels, height, width) for compatible models. This is especially useful when using custom datasets.

---

### 6.3 Custom Datasets

To integrate a custom dataset, users must create a `user_dataset.py` file that exposes the following function:

```python
def get_dataset():
    # Must return (trainset, testset, valset)
    ...

```

Each returned dataset should be a `torch.utils.data.Subset` with the following conditions:

-   The base dataset must expose `targets` (list of labels).
    
-   The dataset must be indexable and return (image_tensor, label_int) on access.
    
-   A validation set must be included (either pre-split or split inside `get_dataset()`).
    

Additionally, the dataset is expected to define class names via:

```python
dataset.classes  # Example: ["cat", "dog", "car", ...]

```

If not available, the framework will default to numerical labels.

----------

### 6.4 Custom Models

To integrate a custom model, users must define a `user_model.py` file with a `get_model()` function:

```python
def get_model():
    # Return a torch.nn.Module instance
    ...

```

The model must support:

-   A configurable number of output classes (set via `num_classes` or equivalent).
    
-   Accept input tensors shaped as defined in the profile.
    

If a user provides both a custom dataset and custom model, the framework will dynamically load both during setup and simulation.

----------

### 6.5 Summary

-   Built-in datasets and models allow rapid testing.
    
-   Custom components provide extensibility and research flexibility.
    
-   All parameters, including architecture and shape, are saved in the `profile.yaml` file for reproducibility.
    
## 7. Profile YAML Specification

Each attack simulation in Module 2 relies on a structured YAML configuration file, referred to as the **profile**. This file serves as a unified representation of:

- The dataset and model in use
- Threat modeling information (from Module 1)
- Attack parameters for each submodule (e.g., label flipping)

The framework reads this profile to load all relevant components and make decisions about which submodules to run and how.

---

### 7.1 Profile Structure

Below is a sample structure of a complete `profile.yaml`:

```yaml
dataset:
  type: builtin
  name: cifar10

model:
  type: builtin
  name: cnn
  num_classes: 10
  input_shape: [3, 32, 32]
  params:
    conv_filters: 32
    hidden_size: 128

threat_model:
  attack_goal: untargeted
  data_sensitivity: high
  deployment_scenario: cloud
  interface_exposed: api
  model_access: gray-box
  model_type: cnn
  training_data_source: mixed
  threat_categories:
    - data_poisoning
    - model_inversion
    - membership_inference

attack_overrides:
  data_poisoning:
    label_flipping:
	  strategy: many_to_one
	  flip_rate: 0.08
	  source_class: null
	  target_class: 3

```

----------

### 7.2 Fields Explained

-   **dataset**
    
    -   `type`: `builtin` or `custom`
        
    -   `name`: Name of the dataset (or `user_dataset.py`)
        
-   **model**
    
    -   `type`: `builtin` or `custom`
        
    -   `name`: Model identifier
        
    -   `num_classes`: Number of output classes
        
    -   `input_shape`: Shape of the input tensor (channels, height, width)
        
    -   `params`: Dictionary of optional architecture-specific settings
        
-   **threat_model** (from Module 1)
    
    -   Contains detailed threat characteristics
        
    -   Used by Module 2 to determine which submodules to activate
        
-   **attack_overrides**
    
    -   Holds configuration overrides for specific submodules
        
    -   For example, `data_poisoning.label_flipping` stores parameters for the Label Flipping attack
        

----------

### 7.3 Updating the Profile

The profile is updated during the setup phase of each submodule. If a threat category is matched (e.g., `data_poisoning`), the corresponding setup script:

1.  Loads the profile
    
2.  Runs an interactive CLI to ask attack-specific questions
    
3.  Updates the `attack_overrides` section with the new configuration
    
4.  Saves the changes to disk (overwriting the original file)
    

This allows seamless integration of threat modeling, dataset/model configuration, and attack customization — all in one place.

----------

### 7.4 Benefits

-   **Reproducibility**: Profiles can be reused to replicate experiments
    
-   **Modularity**: Different parts of the framework rely only on their relevant sections
    
-   **Flexibility**: Custom models and datasets can be mixed with threat categories and attack parameters
    
Great! Here's the next section of the documentation.

## 8. Execution Flow

The execution flow of Module 2 is carefully designed to provide flexibility, transparency, and reproducibility for testing a wide range of adversarial scenarios. This section outlines the full lifecycle of a simulation run, from selecting the configuration to generating the final reports.

---

### 8.1 High-Level Steps

1. **Interactive Setup**
   - User selects the dataset and model (built-in or custom)
   - Parameters like input shape, hidden layer size, or filter count can be customized
   - A threat profile generated in Module 1 is selected
   - If applicable threats are detected, the corresponding submodule(s) are triggered

2. **Submodule Setup Phase**
   - Before each attack, the framework runs an interactive setup phase
   - This phase suggests values based on the threat profile and allows users to override them
   - Attack parameters are stored in the `attack_overrides` section of the profile YAML

3. **Baseline Model Training**
   - The selected model is trained on the clean (non-poisoned) dataset
   - The baseline accuracy and per-class accuracies are computed and saved to a results file

4. **Adversarial Attack Execution**
   - Each triggered submodule executes its respective attack using the parameters defined earlier
   - For example, the Label Flipping submodule flips a subset of training labels
   - The model is retrained on the poisoned dataset
   - The new accuracy and class-specific performance metrics are collected

5. **Result Saving and Reporting**
   - All metrics and flipped sample logs are saved to a `.json` report file
   - A corresponding `.md` Markdown report is also generated for easy human inspection
   - Visual examples of flipped samples are included in the Markdown report

---

### 8.2 Summary Diagram

```text
┌────────────────────────────┐
│   Module 2 Setup Wizard    │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Select Model & Dataset    │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│     Load Threat Profile    │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│   Setup Submodules (Q&A)   │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Train Clean Baseline Model│
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│   Run Enabled Submodules   │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│     Save JSON + MD Report  │
└────────────────────────────┘

```

----------

### 8.3 Automatic Decisions

Some aspects of the flow are handled automatically by the framework:

-   The appropriate input shape is inferred from the dataset unless customized by the user
    
-   Model parameters (e.g., hidden size, filter count) default to sensible values but are user-editable
    
-   Each attack module decides internally how to poison data based on selected strategy and flip rate
    
-   Submodules only run if their corresponding threat is detected in the selected profile
    

----------

### 8.4 Notes on Reusability

-   Profiles can be reused to test the same attack configurations on different models
    
-   Results are stored under the `results/` directory and can be inspected or versioned
    
-   The modular design allows new submodules or attacks to be integrated with minimal changes
    
----------

## 9. Reporting and Outputs

After executing any configured attack, Module 2 automatically generates structured reports summarizing the results. These outputs are essential for:

-   Quantifying the effectiveness of each attack strategy
    
-   Comparing performance between clean and poisoned training
    
-   Supporting reproducibility and analysis across different scenarios

### 9.1 Output Directory

All files are saved in the `results/` directory inside the `module2_attack_simulation` folder. Each attack submodule stores its outputs in a dedicated subfolder, maintaining separation and clarity:

```
results/
├── baseline_accuracy.json
├── data_poisoning/
│   ├── label_flipping/
│   │   ├── label_flipping_metrics.json
│   │   ├── label_flipping_report.md
│   │   └── examples/
│   └── clean_label/
│       ├── clean_label_metrics.json
│       ├── clean_label_report.md
│       └── examples/
...

```

----------

### 9.2 `baseline_accuracy.json`

This file logs the performance of the clean model (trained without any attack). It serves as a baseline for comparison across all attacks.

Example structure:

```json
{
  "accuracy": 0.612,
  "per_class_accuracy": {
    "airplane": 0.653,
    "automobile": 0.748,
    ...
  }
}

```

----------

### 9.3 Attack-Specific Reports

Each submodule produces its own metrics file (e.g., `label_flipping_metrics.json`) and a corresponding Markdown report (`label_flipping_report.md`). These files include:

-   Parameters and configuration used during the attack
    
-   Model performance after the attack
    
-   Additional metadata and visual examples
    

Details vary by submodule and are documented in the corresponding section of each attack.

----------

### 9.4 Visual Examples Folder

Each attack includes an `examples/` folder containing visual outputs. These files serve as illustrative samples of the modified data.

File formats vary depending on the submodule. Examples:

```
flip_1234_cat_to_dog.png     # Label Flipping
poison_4567_dog.png          # Clean Label

```

----------

### 9.5 Customization Notes

-   The number of visual samples saved per run is configurable.
    
-   Developers may extend the `.json` or `.md` formats with new fields as long as the structure remains consistent.
    
-   Markdown generation scripts are modular and can be adapted to new attack types easily.
    

----------

## 10. Submodule 2.1 — Data Poisoning Attacks

This submodule of the Safe-DL Framework explores *data poisoning* attacks — a family of adversarial strategies where the attacker introduces manipulated training samples to degrade the model’s performance. These attacks are stealthy and often hard to detect, especially when the attacker does not modify the input distribution directly.

### 10.1 Overview

Data poisoning attacks aim to compromise the training process by injecting malicious or mislabeled samples into the dataset. These attacks can significantly degrade model performance or steer it toward attacker-defined objectives. In **Submodule 2.1**, we simulate such poisoning scenarios using multiple techniques, allowing for evaluation of the model’s robustness under adversarial data conditions.

This submodule currently includes two attack types:

-   **Label Flipping:** Corrupts training labels to mislead the learning process.
    
-   **Clean Label:** Applies imperceptible perturbations to inputs while keeping the labels unchanged, making them appear benign.
    

Each technique is implemented independently with configurable parameters. The system uses threat profile cues to suggest realistic attack configurations, which the user can accept or override.

The poisoning process is dataset- and model-agnostic, and all outputs—including poisoned samples, evaluation metrics, and reports—are stored for reproducibility and later analysis.

### 10.2 Configuration via Setup Script

All data poisoning attacks are configured in advance through the `setup_module2.py` script. This script provides a guided, interactive questionnaire in the terminal, helping users define key attack parameters before simulations are run.

During this process, the framework analyzes the threat model defined in the selected `.yaml` profile (e.g., attack goal, data source, deployment scenario) and proposes **automated suggestions** tailored to that scenario. Users may either accept these suggestions or manually override them.

Once confirmed, all values are saved under the `attack_overrides` section of the profile:

```yaml
attack_overrides:
  data_poisoning:
    label_flipping:
      ...
    clean_label:
      ...

```

This ensures that the attack configuration is reproducible and traceable, and can be reused across different simulations or models.

Each data poisoning attack—such as **Label Flipping** or **Clean Label**—has its own specific parameters and decision logic. These are documented in detail in the following subsections.

### 10.3 Label Flipping Attack

Label flipping is one of the most studied and effective forms of data poisoning attacks. In this sub-attack module, we simulate realistic scenarios where an attacker manipulates a fraction of training labels with the goal of degrading model performance—either broadly (untargeted) or in a focused manner (targeted).

----------

#### 10.3.1 Objective

The objective of this attack is to evaluate the model’s robustness against label-level manipulation by simulating different label flipping strategies. This helps assess how a model might behave when trained on noisy, tampered, or adversarially mislabeled data.

----------

#### 10.3.2 Available Strategies

This submodule supports three distinct attack strategies:

-   **Fully Random (`fully_random`)**  
    Randomly selects a subset of samples and replaces their labels with different random class labels.
    
-   **Many-to-One (`many_to_one`)**  
    Randomly flips samples from _any class except one_ to a fixed target class (e.g., all non-dog images labeled as dogs).
    
-   **One-to-One (`one_to_one`)**  
    Flips a specific source class to a specific target class (e.g., all cats labeled as dogs).
    

These strategies can be selected manually or suggested automatically based on the selected threat profile.

----------

#### 10.3.3 Configuration Parameters

The parameters for this attack are configured during the setup process via `setup_module2.py`. The script guides the user through a CLI questionnaire and stores all values inside the `.yaml` profile under `attack_overrides → data_poisoning → label_flipping`.

####  Suggested Defaults Based on Threat Profile

Before prompting the user, the framework suggests reasonable defaults based on the profile's threat model:

-   **Attack Goal (`attack_goal`)**
    
    -   `targeted`: attacker focuses on misclassifying specific classes.
        
        -   Suggests `one_to_one` if the data source is `user_generated` (indicating strong attacker control).
            
        -   Suggests `many_to_one` otherwise.
            
    -   `untargeted`: attacker aims for general degradation.
        
        -   Suggests `fully_random` strategy for broad disruption and low detectability.
            
-   **Training Data Source (`training_data_source`)**
    
    -   `user_generated`: lower flip rate (`0.05`) due to fewer poisoned samples.
        
    -   `external_public`: higher risk, so higher flip rate (`0.10`).
        
    -   Otherwise (e.g., `internal_clean`), uses a moderate rate (`0.08`).
        
-   **Class Selection**
    
    -   Random if not explicitly specified.
        
    -   For `many_to_one`, a single target class is selected.
        
    -   For `one_to_one`, both source and target classes are selected at random, but users can override this.
        

####  Manual Configuration Options

During the CLI interaction, users can override any of the above suggestions. The following options are configurable:

-   **Strategy:** One of `fully_random`, `many_to_one`, or `one_to_one`
    
-   **Flip Rate:** A value between `0.0` and `1.0`
    
-   **Source Class and Target Class:** Randomly selected or manually defined, with class labels shown
    

####  Example CLI Interaction

```
Choose flipping strategy:
[x] Random to fixed (many→one)
[ ] Fixed to fixed (one→one)
[ ] Fully random (random→random)

Flip rate (e.g., 0.08): 0.1

Pick target class randomly? No
Select target class (flip TO):
0 – airplane
1 – automobile
...

```

----------

#### 10.3.4 YAML Integration

Once configuration is finalized, the selected values are saved inside the YAML profile, ensuring full reproducibility:

```yaml
attack_overrides:
  data_poisoning:
    label_flipping:
      strategy: one_to_one
      flip_rate: 0.1
      source_class: 3
      target_class: 5

```

----------

#### 10.3.5 Reporting and Metrics

After executing the label flipping attack, the submodule generates a detailed JSON report and a corresponding Markdown file. These outputs help quantify the impact of the label manipulation on model performance and visualize the class flips.

####  Files Generated

```
results/data_poisoning/label_flipping/
├── label_flipping_metrics.json
├── label_flipping_report.md
└── examples/
    ├── flip_1234_cat_to_dog.png
    ├── flip_5678_airplane_to_dog.png
    └── ...

```

----------

####  `label_flipping_metrics.json`

This file includes all key metrics and attack parameters:

-   **Attack Metadata:**
    
    -   `attack_type`: always `"label_flipping"`
        
    -   `flipping_strategy`: selected strategy (e.g., `many_to_one`)
        
    -   `flip_rate`: percentage of samples that were flipped
        
    -   `source_class`, `target_class`: if applicable, based on strategy
        
-   **Performance Metrics:**
    
    -   `accuracy_after_attack`: model accuracy on the clean test set after being trained on the poisoned dataset
        
    -   `per_class_accuracy`: dictionary of class-wise accuracies post-attack
        
-   **Flip Summary:**
    
    -   `num_flipped`: total number of flipped samples
        
    -   `flipping_map`: dictionary showing the number of flips from one class to another
        
-   **Example Flips:**
    
    -   `example_flips`: list of flipped image samples, with index, labels, and label names
        

##### Example structure:

```json
{
  "attack_type": "label_flipping",
  "flipping_strategy": "many_to_one",
  "accuracy_after_attack": 0.402,
  "per_class_accuracy": {
    "airplane": 0.292,
    "automobile": 0.781,
    ...
  },
  "flip_rate": 0.08,
  "source_class": null,
  "target_class": 5,
  "num_flipped": 412,
  "flipping_map": {
    "cat→dog": 107,
    "airplane→dog": 96,
    ...
  },
  "example_flips": [
    {
      "index": 1234,
      "original_label": 0,
      "new_label": 5,
      "original_label_name": "airplane",
      "new_label_name": "dog"
    },
    ...
  ]
}

```

----------

####  `label_flipping_report.md`

The Markdown report presents the metrics in a human-readable format and includes:

-   Overview of the attack configuration (strategy, flip rate, classes)
    
-   Summary table of per-class accuracy
    
-   Flip map table showing flipped class pairs
    
-   Visual preview of 5 poisoned samples with class changes
    

This file is useful for qualitative assessment and comparisons across experiments.

----------

####  `examples/` Folder

Saved images are named as:

```
flip_<index>_<original>_to_<new>.png

```

They show a few of the poisoned training samples where the label was flipped. These are helpful for visually verifying the attack logic and identifying patterns or issues.

----------

#### 10.3.6 Design Considerations

-   **Modular Structure:** The attack is implemented independently and can be extended or modified easily.
    
-   **Dataset Compatibility:** Works with any dataset conforming to the expected interface (`dataset.targets`, etc.).
    
-   **Model Compatibility:** Compatible with built-in and user-defined models.
    
-   **Visual Examples:** Clear visualizations of label flips help in qualitative evaluation.
    
----------

### 10.4 Clean Label Poisoning

Clean label poisoning is a stealthy form of data poisoning where the attacker modifies input samples without changing their labels. These poisoned samples are crafted to appear benign but lead to misclassification at inference time. Unlike label flipping, clean-label attacks do not introduce inconsistencies in labels, making them significantly harder to detect using traditional data sanitization techniques.

This sub-attack module implements multiple perturbation-based strategies to simulate realistic clean-label scenarios under both targeted and untargeted threat models.

----------

#### 10.4.1 Objective

The goal of this attack is to simulate scenarios where training samples are subtly altered in a way that affects the learned model’s decision boundaries, without violating the semantic correctness of their labels.

This includes scenarios where:

-   Poisoned samples maintain plausible appearance
    
-   Labels are kept consistent, avoiding detection
    
-   The trained model becomes biased towards misclassifying similar clean inputs at test time
    

By evaluating a model's robustness against such subtle manipulations, we can assess its susceptibility to real-world threats that do not rely on overt data corruption.

----------

#### 10.4.2 Perturbation Methods

The clean label submodule supports three perturbation strategies:

-   **Overlay Patch (`overlay`)**  
    Adds a fixed opaque patch to a corner of the image. Simple and fast to compute. Controlled by an `epsilon` parameter which scales the patch intensity.
    
-   **Additive Noise (`noise`)**  
    Injects random Gaussian noise into the pixel values. This noise is controlled via the `epsilon` hyperparameter to maintain imperceptibility.
    
-   **Feature Collision (`feature_collision`)**  
    A more advanced method where poisoned samples are optimized to have similar internal feature representations to a target sample (from another class), according to the model’s intermediate layers. This causes feature-level confusion without changing labels.
    

Each strategy has different trade-offs in terms of stealthiness, attack success, and computational cost.

----------

#### 10.4.3 Configuration via Setup Script

The clean label attack is configured during the interactive execution of `setup_module2.py`. This setup process asks the user for all relevant parameters, but it also provides **automatic suggestions** based on the threat profile.

The configuration is stored under:

```yaml
attack_overrides:
  data_poisoning:
    clean_label:
      ...

```

#### Suggested Defaults Based on Threat Profile

Before prompting the user, the script suggests reasonable defaults based on the following logic:

-   **Attack Goal (`attack_goal`)**
    
    -   If the goal is `targeted`, the attack aims to affect a specific class:
        
        -   A random `target_class` is selected.
            
        -   A lower poison fraction is proposed (e.g., `0.05`) to remain stealthy.
            
    -   If the goal is `untargeted`, the goal is general degradation:
        
        -   `target_class` is left undefined.
            
        -   A higher poison fraction is suggested (e.g., `0.08` or more).
            
-   **Training Data Source (`training_data_source`)**
    
    -   If the data is from `user_generated` sources:
        
        -   Assumes stronger control → selects `overlay` (low-complexity attack).
            
    -   If from `external_public`:
        
        -   Assumes less trust → selects `feature_collision` or `noise` depending on access level.
            
-   **Default Values for Other Parameters**
    
    -   `epsilon`: set to `0.1`, controlling perturbation intensity.
        
    -   `max_iterations`: set to `100` for optimization-based methods.
        
    -   `source_selection`: set to `"random"` for selecting which clean images will be poisoned.
        

These defaults can be manually overridden via the CLI.

#### Manual Configuration Options

Users can modify the following parameters during setup:

-   **Poison Fraction:** The percentage of training samples to poison (`0.0` to `1.0`)
    
-   **Target Class (optional):** For targeted attacks; user can pick or randomize.
    
-   **Perturbation Method:** One of `overlay`, `noise`, or `feature_collision`
    
-   **Epsilon:** Perturbation magnitude
    
-   **Max Iterations:** Only applies to `feature_collision`
    
-   **Source Selection Strategy:**
    
    -   `random`: randomly pick clean images to poison.
        
    -   `most_confident` / `least_confident`: requires a trained model and influences source sample selection based on model confidence.
        

#### Example CLI Interaction

```
Poisoning fraction (e.g., 0.08): 0.15

Is this a targeted attack? Yes  
Select target class:  
0 – airplane  
1 – automobile  
...

Select perturbation method: overlay  
Epsilon: 0.1  
Max optimization iterations: 100  
Source selection: random  

```

Once completed, the values are saved into the profile, allowing the attack to run reproducibly later on.

----------

#### 10.4.4 YAML Integration

After the clean label configuration is finalized through the setup script, all selected values are saved into the `attack_overrides` section of the YAML profile. This ensures the attack setup can be re-used or shared without needing to reconfigure manually.

```yaml
attack_overrides:
  data_poisoning:
    clean_label:
      fraction_poison: 0.15
      target_class: 8
      perturbation_method: overlay
      max_iterations: 100
      epsilon: 0.1
      source_selection: random

```

This configuration is then used directly by the framework during the attack phase. If a `target_class` is omitted, the attack is considered untargeted and poisons across multiple classes.

----------

#### 10.4.5 Reporting and Metrics

After the clean label attack is executed, the submodule produces structured reports to analyze the effectiveness and behavior of the poisoning. These reports include numerical performance metrics, example logs, and visual samples of the poisoned data.

#### Files Generated

```
results/data_poisoning/clean_label/
├── clean_label_metrics.json
├── clean_label_report.md
└── examples/
    ├── poison_12345_ship.png
    ├── poison_67890_ship.png
    └── ...

```

----------

#### `clean_label_metrics.json`

This file logs all critical parameters and results of the attack. Key fields include:

-   **Attack Metadata:**
    
    -   `attack_type`: always `"clean_label"`
        
    -   `perturbation_method`: which method was used (e.g., `overlay`, `noise`, `feature_collision`)
        
    -   `fraction_poison`: portion of the training set that was poisoned
        
    -   `target_class`: only set for targeted attacks
        
    -   `epsilon`, `max_iterations`, `source_selection`: additional parameters used
        
-   **Performance Metrics:**
    
    -   `accuracy_after_attack`: final model accuracy on clean test data after training on poisoned set
        
    -   `per_class_accuracy`: dictionary with class-wise accuracies
        
-   **Poisoned Sample Log:**
    
    -   `num_poisoned_samples`: total number of poisoned examples
        
    -   `num_total_samples`: size of the training set
        
    -   `example_poisoned_samples`: list of up to 5 poisoned samples with:
        
        -   dataset index,
            
        -   original label,
            
        -   perturbation norm,
            
        -   image path for visualization
            

##### Example excerpt:

```json
{
  "attack_type": "clean_label",
  "perturbation_method": "overlay",
  "fraction_poison": 0.15,
  "target_class": 8,
  "max_iterations": 100,
  "epsilon": 0.1,
  "source_selection": "random",
  "num_poisoned_samples": 674,
  "num_total_samples": 45000,
  "accuracy_after_attack": 0.6067,
  "per_class_accuracy": {
    "airplane": 0.567,
    "automobile": 0.749,
    ...
  },
  "example_poisoned_samples": [
    {
      "index": 49924,
      "original_label": 8,
      "original_label_name": "ship",
      "perturbation_norm": 2.10,
      "example_image_path": "results/data_poisoning/clean_label/examples/poison_49924_ship.png"
    },
    ...
  ]
}

```

----------

#### `clean_label_report.md`

This Markdown file summarizes the attack in a human-readable format. It includes:

-   Overview of the parameters used
    
-   Final accuracy and per-class breakdown
    
-   Sample table of poisoned examples
    
-   Embedded visualizations for manual inspection
    

----------

#### `examples/` Folder

Images are saved in the format:

```
poison_<index>_<label>.png

```

These show the poisoned versions of real training images. Even though the label remains unchanged, the visual perturbation may influence how the model interprets the sample internally.

----------

#### 10.4.6 Design Considerations

The clean label attack implementation in Safe-DL was designed to be flexible, extensible, and realistic in terms of real-world threat modeling. Below are the key architectural and usability considerations:

-   **Label-Preserving Semantics:**  
    Unlike label flipping, clean label attacks do not alter the training labels. The poisoned samples appear legitimate and correctly labeled, making detection harder both for humans and automated validation pipelines.
    
-   **Multiple Perturbation Methods:**  
    The submodule supports a variety of perturbation strategies:
    
    -   `overlay`: visually adds a semi-transparent patch in a specific image region.
        
    -   `noise`: introduces random Gaussian noise controlled by an `epsilon` value.
        
    -   `feature_collision`: optimizes the poisoned sample so that its internal feature representation matches that of another class (requires feature access to the model).
        
    
    This diversity enables testing both simple and advanced clean-label attacks.
    
-   **Targeted and Untargeted Support:**  
    The attack can be configured to either mislead the model toward a specific class (targeted) or broadly degrade performance (untargeted), depending on the threat profile and user preference.
    
-   **Source Selection Strategy:**  
    If the attack uses `feature_collision`, the user may define how the poisoned base samples are selected:
    
    -   `random`: randomly chosen from the target class.
        
    -   `most_confident` or `least_confident`: based on prediction confidence from a pre-trained clean model (future extension).
        
    -   For overlay and noise, the strategy is mainly preserved for consistency but has limited practical effect.
        
-   **Model Feature Extraction Compatibility:**  
    When using `feature_collision`, the target model must implement a `features(x)` method that returns internal representations prior to the classification head. This is necessary for matching representations during optimization.
    
-   **Visualization for Transparency:**  
    The attack saves a sample of poisoned images to disk. These are used to validate the attack visually and support manual auditing and report generation.
    
-   **Modular Implementation:**  
    The submodule is structured to allow future integration of more complex clean label variants (e.g., adversarial patch optimization, stealth-aware perturbations, or poisoning during fine-tuning).