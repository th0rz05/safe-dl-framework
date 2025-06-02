# Randomized Smoothing Defense Report

**Attack Evaluated:** pgd
**Defense Method:** randomized_smoothing

## Smoothing Parameters
- **Sigma (noise std):** 0.25
- **Number of Samples:** 25

## Evaluation Results
- **Smoothed Accuracy on Clean Test Set:** 0.2472
- **Smoothed Accuracy on Adversarial Test Set:** 0.0476

### Per-Class Accuracy (Clean + Smoothed)
- **airplane**: 0.3480
- **automobile**: 0.8640
- **bird**: 0.0850
- **cat**: 0.0030
- **deer**: 0.0000
- **dog**: 0.0180
- **frog**: 0.5460
- **horse**: 0.0480
- **ship**: 0.1030
- **truck**: 0.4570

### Per-Class Accuracy (Adversarial + Smoothed)
- **airplane**: 0.0950
- **automobile**: 0.1670
- **bird**: 0.0050
- **cat**: 0.0000
- **deer**: 0.0000
- **dog**: 0.0010
- **frog**: 0.1400
- **horse**: 0.0030
- **ship**: 0.0090
- **truck**: 0.0560