# Fine-Pruning Defense Report

**Attack Type:** static_patch
**Defense Method:** Fine-Pruning

## Pruning Details

- **Pruning Ratio:** 0.2
- **Pruned Layer:** fc.3
- **Number of Neurons Pruned:** 2
- **Pruned Neuron Indices:** 8, 1

## Accuracy After Defense

- **Clean Accuracy:** 0.6733
- **ASR After Defense:** `0.5161`

### Per-Class Accuracy (Clean)
- **airplane**: 0.7330
- **automobile**: 0.8920
- **bird**: 0.4220
- **cat**: 0.5610
- **deer**: 0.4640
- **dog**: 0.6980
- **frog**: 0.8260
- **horse**: 0.7090
- **ship**: 0.7820
- **truck**: 0.6460


### Per-Original-Class ASR
- **Original Class airplane**: `0.4210`
- **Original Class automobile**: `0.3650`
- **Original Class bird**: `0.6080`
- **Original Class cat**: `0.6080`
- **Original Class deer**: `0.6790`
- **Original Class dog**: `0.6540`
- **Original Class frog**: `0.3380`
- **Original Class ship**: `0.5180`
- **Original Class truck**: `0.4540`
