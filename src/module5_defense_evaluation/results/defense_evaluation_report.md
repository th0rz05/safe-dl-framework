# Defense Evaluation Report

**Profile**: `visitech.yaml`  
**Dataset**: `cifar10`  
**Model**: `resnet18`  
**Generated on**: 2025-06-23 16:22:40

## Overview

This report summarizes the evaluation of defenses applied to mitigate adversarial attacks on deep learning models. Each defense is scored based on its mitigation effectiveness, its impact on clean accuracy (CAD), and estimated computational cost. The final score reflects an overall balance between effectiveness and efficiency.

## Summary Table

| Attack | Defense | Mitigation | CAD | Cost | Final Score |
|--------|---------|------------|-----|------|--------------|
| static_patch | activation_clustering | 0.970 | 0.887 | 0.300 | 0.926 |
| static_patch | spectral_signatures | 0.000 | 0.743 | 0.500 | 0.142 |
| label_flipping | data_cleaning | 0.151 | 1.039 | 0.200 | 0.322 |
| pgd | adversarial_training | 0.282 | 0.923 | 0.800 | 0.380 |
| pgd | randomized_smoothing | 0.010 | 0.449 | 0.500 | 0.093 |
| spsa | gradient_masking | 0.006 | 0.998 | 0.400 | 0.196 |
| spsa | jpeg_preprocessing | 0.003 | 0.144 | 0.100 | 0.031 |

## Notes

- **Mitigation Score**: Effectiveness in recovering model performance after an attack.
- **CAD (Clean Accuracy Drop)**: Degree of performance degradation on clean data.
- **Cost Score**: Relative computational/resource impact of the defense.
- **Final Score**: Aggregated score combining all metrics.
