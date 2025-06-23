# Adversarial Training Defense Report

**Attack Evaluated:** pgd
**Defense Method:** adversarial_training

## Training Parameters
- **Base Attack Used for Training:** fgsm
- **Epsilon:** 0.03
- **Mixed with Clean Samples:** True

## Evaluation Results

- **Clean Test Accuracy:** 0.7655
- **Adversarial Test Accuracy:** 0.4188

### Per-Class Accuracy (Clean)
- **airplane**: 0.7980
- **automobile**: 0.8200
- **bird**: 0.6810
- **cat**: 0.5850
- **deer**: 0.7430
- **dog**: 0.6690
- **frog**: 0.8370
- **horse**: 0.8130
- **ship**: 0.8710
- **truck**: 0.8380

### Per-Class Accuracy (Adversarial)
- **3**: 0.2520
- **8**: 0.5770
- **0**: 0.5530
- **6**: 0.5060
- **1**: 0.4560
- **9**: 0.3900
- **5**: 0.3380
- **7**: 0.4860
- **4**: 0.3310
- **2**: 0.2990