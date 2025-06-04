# Fine-Pruning Defense Report

**Attack Type:** static_patch
**Defense Method:** Fine-Pruning

## Pruning Details

- **Pruning Ratio:** 0.2
- **Pruned Layer:** fc.3
- **Number of Neurons Pruned:** 2
- **Pruned Neuron Indices:** 5, 1

## Accuracy After Defense

- **Clean Accuracy:** 0.6000
- **Adversarial Accuracy:** 0.3533

### Per-Class Accuracy (Clean)
- **airplane**: 0.4240
- **automobile**: 0.7090
- **bird**: 0.4100
- **cat**: 0.5340
- **deer**: 0.4370
- **dog**: 0.7430
- **frog**: 0.5370
- **horse**: 0.8430
- **ship**: 0.5960
- **truck**: 0.7670

### Per-Class Accuracy (Adversarial)
- **airplane**: 0.1870
- **automobile**: 0.2950
- **bird**: 0.2380
- **cat**: 0.9350
- **deer**: 0.2260
- **dog**: 0.1870
- **frog**: 0.2120
- **horse**: 0.5470
- **ship**: 0.3130
- **truck**: 0.3930
