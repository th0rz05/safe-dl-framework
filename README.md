# safe-dl-framework

## **Modular Security Framework for Deep Learning Projects**

This framework provides a structured and practical approach to identifying, simulating, and defending against adversarial threats in deep learning models. It is especially suited for projects involving image classification, computer vision, and model deployment in potentially hostile environments.

## 📌 Features

- 🔍 Threat modeling with customizable profiles
- 🧪 Simulation of real-world attacks:
  - Data poisoning
  - Backdoor attacks
  - Adversarial examples
  - Model stealing and inference attacks
- 🛡️ Adaptive defense strategies
- 📊 Robustness benchmarking and evaluation tools
- 🚀 Secure deployment checklists
- 🧠 Optional real-time detection module for production environments

## ⚙️ Modules Overview

1. **Threat Modeling** – Define attacker assumptions and generate threat profiles
2. **Attack Simulation** – Apply selected attacks based on threat profile
3. **Vulnerability Assessment** – Quantify model weaknesses
4. **Defensive Strategies** – Apply and test appropriate defenses
5. **Benchmarking** – Compare model performance before and after defenses
6. **Deployment Guidelines** – Secure your model in the real world
7. *(Optional)* **Real-Time Monitoring** – Detect anomalies during inference

## 📦 Installation

```bash
git clone https://github.com/yourusername/safe-dl-framework.git
cd safe-dl-framework
pip install -r requirements.txt
```

## 🚀 Usage

Start with the threat modeling module:

```bash
python threat_model_cli.py
```

Then simulate attacks, apply defenses, and evaluate robustness:

```bash
python run_framework.py --profile traffic_signs_profile.yaml
```

## 📁 Project Structure

```bash
.
├── attacks/              # Attack scripts by category
├── defenses/             # Defensive techniques
├── framework/            # Core logic and pipeline
├── profiles/             # YAML threat profiles
├── results/              # Benchmark results and logs
├── threat_model_cli.py   # Threat modeling questionnaire (CLI)
├── run_framework.py      # Main execution script
└── requirements.txt
```

## 📖 License

MIT License

Designed by Tiago Barbosa – M.Sc. Informatics and Computing Engineering @ FEUP
Framework created as part of a master's thesis on enhancing the security of deep neural networks.
