# Randomized Smoothing Defense Report

**Attack Evaluated:** pgd
**Defense Method:** randomized_smoothing

## Smoothing Parameters
- **Sigma (noise std):** 0.25
- **Number of Samples:** 25

## Evaluation Results
- **Smoothed Accuracy on Clean Test Set:** 0.3720
- **Smoothed Accuracy on Adversarial Test Set:** 0.1988

### Per-Class Accuracy (Clean + Smoothed)
- **airplane**: 0.3750
- **automobile**: 0.0260
- **bird**: 0.4540
- **cat**: 0.4130
- **deer**: 0.7440
- **dog**: 0.2500
- **frog**: 0.5500
- **horse**: 0.0960
- **ship**: 0.7280
- **truck**: 0.0840

### Per-Class Accuracy (Adversarial + Smoothed)
- **airplane**: 0.0410
- **automobile**: 0.0000
- **bird**: 0.2390
- **cat**: 0.1540
- **deer**: 0.8340
- **dog**: 0.0730
- **frog**: 0.4080
- **horse**: 0.0060
- **ship**: 0.2240
- **truck**: 0.0090