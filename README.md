
# Safe-DL — Modular Deep-Learning Security Framework

*A step-by-step pipeline that takes a neural-network project from threat modelling,  
through attack simulation and defence hardening, to a final audit-ready report.*

> **Developed by Tiago Barbosa** (MSc in Informatics Engineering, FEUP, 2025)

## Why Safe-DL?

- **End-to-End Coverage** – goes beyond isolated demos: starts with threat modelling, walks through realistic attack simulations, risk scoring, defence hardening, benchmarking, and ends with an audit-grade final report.

- **Modular by Design** – each stage is a self-contained folder (`moduleX_*`) with its own CLI; you can run the full pipeline or pick just the pieces you need.

- **Reproducible & YAML-Driven** – a single `profile.yaml` is incrementally enriched by every module, capturing *all* parameters, metrics, and decisions for perfect experiment replay.

- **Plug-and-Play Extensibility** – add your own dataset, model, attack, or defence by implementing a minimal Python interface; the framework auto-discovers and wires it in.

- **Rich Automated Reporting** – every module emits both machine-readable JSON and human-friendly Markdown (with images and plots), culminating in a one-click final dossier.


## Project Structure

The Safe-DL framework is organized into modular folders, each corresponding to a specific phase in the secure ML lifecycle. All modules operate independently via their own command-line interfaces and share a common YAML profile file.

```bash
src/
├── module1_threat_modeling/
│   └── threat_model_cli.py
├── module2_attack_simulation/
│   ├── attacks/
│   │   ├── data_poisoning/
│   │   ├── backdoor/
│   │   └── evasion/
│   ├── setup_module2.py
│   └── run_module2.py
├── module3_risk_analysis/
│   └── run_module3.py
├── module4_defense_application/
│   ├── defenses/
│   │   ├── activation_clustering/
│   │   ├── adversarial_training/
│   │   └── ....
│   ├── setup_module4.py
│   └── run_module4.py
├── module5_benchmarking/
│   └── run_module5.py
├── module6_final_report/
│   └── run_module6.py

```

-   `module1_threat_modeling/` — defines the attacker profile via a structured CLI and outputs `profile.yaml`.
    
-   `module2_attack_simulation/` — injects realistic adversarial threats based on the selected threat model.
    
-   `module3_risk_analysis/` — evaluates vulnerabilities by parsing attack metrics and visualizing impact.
    
-   `module4_defense_application/` — applies dynamic defenses tailored to the threats and attack outcomes.
    
-   `module5_benchmarking/` — reassesses the model post-defense and quantifies gains in robustness.
    
-   `module6_final_report/` — consolidates all results into a final, human-readable summary report.
    

> All modules use and update the same `profile.yaml`, which acts as the central configuration and logging artifact throughout the pipeline.


## Module Pipeline Overview

The framework is divided into six modular stages, each responsible for a specific phase in the secure development of a deep learning model:

-   **Module 1 — Threat Modeling**  
    Defines the attacker's capabilities, goals, and context. Generates the initial `profile.yaml`, which guides all future decisions.
    
-   **Module 2 — Attack Simulation**  
    Injects realistic adversarial threats based on the selected threat profile. Supports data poisoning, backdoors, evasion attacks, and more.
    
-   **Module 3 — Vulnerability Assessment**  
    Analyzes the impact of each attack on the model. Outputs structured metrics and visualizations to identify the most critical weaknesses.
    
-   **Module 4 — Defensive Strategies**  
    Applies tailored defenses based on previous vulnerabilities. Each defense is evaluated and documented for its effectiveness and side effects.
    
-   **Module 5 — Evaluation & Benchmarking**  
    Reassesses the model after defenses are applied. Compares performance before and after, and computes a defense score for each mitigation.
    
-   **Module 6 — Final Report**  
    Aggregates the results of all modules into a clean, comprehensive Markdown report that summarizes the full security lifecycle of the model.
    

> All modules are optional and can be run independently, but using the full pipeline ensures maximum coverage and insight.

## Getting Started

### Requirements

Before running Safe-DL, make sure you have:

-   Python 3.10+
    
-   PyTorch (recommended: latest stable version)
    
-   `questionary`, `matplotlib`, `numpy`, `scikit-learn`, and other standard ML libraries
    
-   (Optional) GPU support via CUDA for faster training and attacks
    

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

----------

### Setup

1.  **Clone the repository**
    

```bash
git clone https://github.com/your-username/safe-dl-framework.git
cd safe-dl-framework
```

2.  **Create a virtual environment (recommended)**
    

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
```

3.  **Install requirements**
    

```bash
pip install -r requirements.txt
```


## How to Run the Full Pipeline

Although each module can be executed independently, running the complete pipeline provides the full security lifecycle. Here's how to do it:

### Step-by-Step

1.  **Threat Modeling**
    

```bash
cd src/module1_threat_modeling/
python threat_model_cli.py

```

> Generates `profile.yaml` based on your model, data, and deployment context.

----------

2.  **Attack Simulation**
    

```bash
cd ../module2_attack_simulation/
python setup_module2.py        # Configure attacks
python run_module2.py          # Run selected attacks

```

> Applies data poisoning, backdoors, evasion, or other attacks based on the profile.

----------

3.  **Vulnerability Assessment**
    

```bash
cd ../module3_risk_analysis/
python run_module3.py

```

> Analyzes the impact of each attack and identifies the most vulnerable areas.

----------

4.  **Defense Configuration & Application**
    

```bash
cd ../module4_defense_application/
python setup_module4.py        # Select or auto-apply defenses
python run_module4.py          # Apply and evaluate selected defenses

```

> Applies and evaluates multiple defense strategies per attack type.

----------

5.  **Evaluation & Benchmarking**
    

```bash
cd ../module5_benchmarking/
python run_module5.py

```

> Computes clean vs adversarial accuracy deltas and assigns a defense score.

----------

6.  **Final Summary Report**
    

```bash
cd ../module6_final_summary/
python run_module6.py

```

> Aggregates all results into a single Markdown file (`final_summary.md`) ready for delivery or auditing.

----------

### Optional: Restarting from a Previous Step

You can edit `profile.yaml` at any time to rerun a module with different parameters.  
All logs and outputs are saved under the `results/` directory and are not overwritten unless explicitly removed.



## Customization & Extensibility

Safe-DL was built to be extended. You can easily plug in your own datasets, models, attacks, or defenses without touching the core logic.

----------

### Adding a Custom Dataset

Add your dataset script in:

```
src/module2_attack_simulation/datasets/
```

Implement the function:

```python
def get_dataset():
    # Must return (trainset, testset, valset)
    return train_dataset, test_dataset, val_dataset
```

> Each item should be a `(image_tensor, label_int)` tuple.  
> If you're using a `torch.utils.data.Subset`, make sure the base dataset has `targets` and `indices`.

----------

### Adding a Custom Model

Add your model in:

```
src/module2_attack_simulation/models/
```

Implement the function:

```python
def get_model(num_classes):
    return MyCustomModel(num_classes)
```

Your model must inherit from `torch.nn.Module` and support `.to(device)` and `.eval()`.

----------

### Adding a New Attack or Defense

Follow the structure under:

-   Attacks: `src/module2_attack_simulation/attacks/`
    
-   Defenses: `src/module4_defense_application/defenses/`
    

Each should define:

-   A `run_<name>.py` script with a `run()` or `apply_defense()` method
    
-   A Markdown/JSON report generator (see existing examples)
    
-   Parameters defined and parsed through the `profile.yaml`
    

> Your module will be auto-detected by the setup scripts as long as you respect the structure.

----------

### Shared Utilities

You can reuse utility functions from:

```
src/module2_attack_simulation/attacks/utils.py
```

Functions include:

-   `train_model()`
    
-   `evaluate_model()`
    
-   `get_class_labels()`
    
-   `load_model()`  
    ...and more.
    

> Safe-DL was built for research-grade security testing, so extensibility and modularity are core principles.

  
## Documentation  
  
- [Full Framework Overview](docs/framework.md)  
- [Module 1 – Threat Modeling](docs/module1.md)  
- [Module 2 – Attack Simulation](docs/module2.md)  
- [Module 3 – Risk Analysis](docs/module3.md)  
- [Module 4 - Defense Application](docs/module4.md)  
  

## Example Application

Imagine you're developing an image classification model for a traffic sign recognition system to be embedded in an autonomous vehicle. The system must be secure, robust, and auditable. Here's how Safe-DL can be applied end-to-end:

----------

### Threat Modeling

The user launches the CLI in `module1_threat_modeling/`, answering questions about the model, data, deployment context, and threat perception. The framework generates the initial `profile.yaml` with:

```yaml
threat_model:
  model_access: gray-box
  attack_goal: targeted
  deployment_scenario: physical_world
  data_sensitivity: medium
  threat_categories:
    - data_poisoning
    - backdoor_attacks
    - adversarial_examples

```

----------

### Attack Simulation

In `module2_attack_simulation/`, the user selects specific attacks relevant to the profile. The framework simulates:

-   **Label Flipping (Data Poisoning)**  
    → Flips 20% of labels in 3 classes → Accuracy drops from 89.5% to 71.2%
    
-   **Static Patch (Backdoor)**  
    → Trigger in bottom-right corner → ASR reaches 94%, clean accuracy drops slightly
    
-   **PGD (Evasion)**  
    → ε = 0.03 → Robust accuracy drops to 34.1%
    

----------

### Risk Analysis

In `module3_risk_analysis/`, the framework analyzes impact across attacks:

```yaml
vulnerabilities:
  data_poisoning:
    severity: high
    recommendation: apply data_cleaning and robust_loss
  backdoor:
    severity: critical
    recommendation: prune suspicious neurons + spectral signature
  evasion:
    severity: medium
    recommendation: adversarial_training (PGD)

```

Markdown and JSON reports are auto-generated, highlighting the most vulnerable classes and visualizing confidence degradation and trigger patterns.

----------

### Defense Application

In `module4_defense_application/`, the system applies selected countermeasures:

-   **Data Cleaning**  
    → Removes 11 suspicious training samples
    
-   **Spectral Signatures**  
    → Detects and filters poisoned samples
    
-   **Adversarial Training (PGD)**  
    → Improves robustness against white-box attacks
    
-   **Neuron Pruning**  
    → Removes neurons with high backdoor activation correlation
    

Each defense generates individual reports, and results are tracked in `defense_config` inside the `profile.yaml`.

----------

### Benchmarking

In `module5_benchmarking/`, the system aggregates defense metrics:

| Attack |  Accuracy Before|Accuracy After|ASR ↓  | Δ Clean Acc |
|--|--|--|--|--
|PGD (ε=0.03)|34.1%|69.8%|-35.7%|-2.4%
|Backdoor|94%|7%|-87%|-0.7%|
|Label Flipping|71.2%|85.4%|—|+14.2%

> Final defense scores combine per-class improvements and accuracy deltas, aiding final decision making.

----------

### Final Report

In `module6_final_summary/`, the framework consolidates:

-   All attack and defense outcomes
    
-   Diagrams, logs, and class-wise breakdowns
    
-   A one-click, audit-ready Markdown dossier summarizing the full process
    

----------

### Result

By the end of the pipeline, the team has:

-   Defined their threat landscape
    
-   Simulated realistic attacks
    
-   Assessed and quantified vulnerabilities
    
-   Applied targeted defenses
    
-   Evaluated effectiveness through standardized metrics
    
-   Produced a complete report ready for deployment, sharing, or certification
    

## Author

Created by **Tiago Barbosa**, M.Sc. in Informatics and Computing Engineering @ FEUP  
This framework is part of the master’s thesis:  
**“Safe-DL: A Modular Framework for Evaluating and Improving Security in Deep Neural Networks”**


## License

Copyright (c) 2025 Tiago Barbosa  
All rights reserved.

This software is made available for **viewing purposes only**.

Permission is granted to access and read the source code for **educational and non-commercial purposes**.  
However, any of the following actions are **strictly prohibited** without prior, explicit written permission from the author:

-   Use or execution of the software in any environment or application
    
-   Copying or redistributing any part of the codebase
    
-   Modification or creation of derivative works
    
-   Incorporation of any part of the code into other software projects
    
-   Commercial use of any kind
    

This software is provided “as is,” without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement.

If you wish to use this software beyond viewing, please contact the author for licensing inquiries.
