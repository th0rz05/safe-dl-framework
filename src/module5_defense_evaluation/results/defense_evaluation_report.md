# Defense Evaluation Report

**Profile**: `visitech.yaml`  
**Dataset**: `cifar10`  
**Model**: `resnet18`  
**Generated on**: 2025-06-23 12:44:41

## Overview

This report summarizes the evaluation of defenses applied to mitigate adversarial attacks on deep learning models. Each defense is scored based on its mitigation effectiveness, its impact on clean accuracy (CAD), and estimated computational cost. The final score reflects an overall balance between effectiveness and efficiency.

## Summary Table

| Attack | Defense | Mitigation | CAD | Cost | Final Score |
|--------|---------|------------|-----|------|--------------|
| static_patch | activation_clustering | -0.105 | 0.214 | 0.300 | -0.017 |
| static_patch | spectral_signatures | 0.027 | 0.000 | 0.500 | 0.000 |
| label_flipping | data_cleaning | 0.151 | 0.000 | 0.200 | 0.000 |
| pgd | adversarial_training | 0.357 | 0.362 | 0.800 | 0.072 |
| pgd | randomized_smoothing | 0.012 | 0.000 | 0.500 | 0.000 |
| spsa | gradient_masking | 0.007 | 0.987 | 0.400 | 0.005 |
| spsa | jpeg_preprocessing | 0.004 | 0.000 | 0.100 | 0.000 |

## Notes

- **Mitigation Score**: Effectiveness in recovering model performance after an attack.
- **CAD (Clean Accuracy Drop)**: Degree of performance degradation on clean data.
- **Cost Score**: Relative computational/resource impact of the defense.
- **Final Score**: Aggregated score combining all metrics.
