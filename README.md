# Safe-DL-Framework

A modular and extensible security framework for deep learning systems, focused on **threat modeling**, **attack simulation**, **defensive strategies**, and **secure deployment**.  
Created as part of a master’s thesis at FEUP by Tiago Barbosa.

---

## Documentation

- [Full Framework Overview](docs/framework.md)
- [Module 1 – Threat Modeling](docs/module1.md)
- [Module 2 – Attack Simulation](docs/module2.md)
- [Module 3 – Vulnerability Assessment](docs/module3.md)

---

## Key Features

- Structured **threat modeling** based on attacker goals, data sensitivity, and system architecture
- Simulation of major attacks: **data poisoning**, **backdoors**, **adversarial examples**, **model stealing**, and more
- Automatic defense mapping and benchmarking tools
- Deployment-ready configurations with optional **real-time monitoring**
- YAML-based profiles for reproducibility and automation

---

## Modules Overview

1. **Threat Modeling** – Define attacker capabilities and generate a threat profile  
2. **Attack Simulation** – Launch realistic attacks based on selected threat categories  
3. **Vulnerability Assessment** – Quantify and visualize model weaknesses  
4. **Defensive Strategies** – Automatically recommend and apply countermeasures  
5. **Benchmarking** – Evaluate the impact of defenses in clean and adversarial settings  
6. **Deployment Guidelines** – Secure your final model with best practices  
7. *(Optional)* **Real-Time Monitoring** – Detect anomalies at runtime

---

## Getting Started

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/safe-dl-framework.git
cd safe-dl-framework
pip install -r requirements.txt
```

Start by creating a threat profile:

```bash
python threat_model_cli.py
```

Then run the main pipeline with your profile:

```bash
python run_framework.py --profile profiles/my_project.yaml
```

---

## Project Structure

```
.
├── attacks/               # Adversarial attack implementations
├── defenses/              # Defensive strategies
├── framework/             # Pipeline integration code
├── profiles/              # YAML threat profiles
├── results/               # Logs and evaluation reports
├── docs/                  # Documentation (markdown)
│   ├── framework.md
│   └── module1.md
├── threat_model_cli.py    # CLI-based threat modeling tool
├── run_framework.py       # Main execution script
└── requirements.txt

```

---

## Author

Created by **Tiago Barbosa**, M.Sc. Informatics and Computing Engineering @ FEUP  
This framework is part of the master's thesis *"Enhancing Security in Deep Neural Networks Against Adversarial Attacks."*

---

## License

This project is licensed under the MIT License.
