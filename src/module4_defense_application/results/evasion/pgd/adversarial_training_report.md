# Adversarial Training Defense Report

**Attack Evaluated:** pgd
**Defense Method:** adversarial_training

## Training Parameters
- **Base Attack Used for Training:** fgsm
- **Epsilon:** 0.03
- **Mixed with Clean Samples:** True

## Evaluation Results

- **Clean Test Accuracy:** 0.4439
- **Adversarial Test Accuracy:** 0.1942

### Per-Class Accuracy (Clean)
- **airplane**: 0.6360
- **automobile**: 0.7710
- **bird**: 0.1970
- **cat**: 0.0160
- **deer**: 0.4290
- **dog**: 0.6000
- **frog**: 0.4760
- **horse**: 0.3890
- **ship**: 0.5350
- **truck**: 0.3900

### Per-Class Accuracy (Adversarial)
- **3**: 0.0050
- **8**: 0.2030
- **0**: 0.3670
- **6**: 0.1180
- **1**: 0.4820
- **9**: 0.1370
- **5**: 0.3030
- **7**: 0.1310
- **4**: 0.1540
- **2**: 0.0420