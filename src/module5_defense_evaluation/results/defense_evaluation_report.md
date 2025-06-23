# Defense Evaluation Report

**Profile**: `test.yaml`  
**Dataset**: `cifar10`  
**Model**: `cnn`  
**Generated on**: 2025-06-23 16:25:31

## Overview

This report summarizes the evaluation of defenses applied to mitigate adversarial attacks on deep learning models. Each defense is scored based on its mitigation effectiveness, its impact on clean accuracy (CAD), and estimated computational cost. The final score reflects an overall balance between effectiveness and efficiency.

## Summary Table

| Attack | Defense | Mitigation | CAD | Cost | Final Score |
|--------|---------|------------|-----|------|--------------|
| static_patch | activation_clustering | 0.022 | 0.658 | 0.300 | 0.145 |
| static_patch | spectral_signatures | 0.071 | 0.759 | 0.500 | 0.199 |
| static_patch | anomaly_detection | 0.003 | 0.963 | 0.300 | 0.189 |
| static_patch | pruning | 0.058 | 0.505 | 0.300 | 0.143 |
| static_patch | fine_pruning | 0.441 | 1.011 | 0.400 | 0.533 |
| static_patch | model_inspection | 0.008 | 0.967 | 0.200 | 0.196 |
| clean_label | provenance_tracking | 0.561 | 1.043 | 0.500 | 0.626 |
| clean_label | influence_functions | 0.573 | 1.044 | 0.500 | 0.636 |
| label_flipping | data_cleaning | 0.603 | 1.139 | 0.200 | 0.696 |
| label_flipping | per_class_monitoring | 0.000 | 1.000 | 0.200 | 0.196 |
| label_flipping | robust_loss | 0.675 | 1.156 | 0.500 | 0.734 |
| label_flipping | dp_training | -1.449 | 0.666 | 0.700 | -0.959 |
| pgd | adversarial_training | 0.194 | 0.657 | 0.800 | 0.266 |
| pgd | randomized_smoothing | 0.048 | 0.366 | 0.500 | 0.106 |
| spsa | gradient_masking | 0.013 | 0.993 | 0.400 | 0.201 |
| spsa | jpeg_preprocessing | 0.351 | 0.954 | 0.100 | 0.467 |

## Notes

- **Mitigation Score**: Effectiveness in recovering model performance after an attack.
- **CAD (Clean Accuracy Drop)**: Degree of performance degradation on clean data.
- **Cost Score**: Relative computational/resource impact of the defense.
- **Final Score**: Aggregated score combining all metrics.
