# Defense Evaluation Report

**Profile**: `test.yaml`  
**Dataset**: `cifar10`  
**Model**: `cnn`  
**Generated on**: 2025-06-09 18:36:31

## Overview

This report summarizes the evaluation of defenses applied to mitigate adversarial attacks on deep learning models. Each defense is scored based on its mitigation effectiveness, its impact on clean accuracy (CAD), and estimated computational cost. The final score reflects an overall balance between effectiveness and efficiency.

## Summary Table

| Attack | Defense | Mitigation | CAD | Cost | Final Score |
|--------|---------|------------|-----|------|--------------|
| static_patch | activation_clustering | -0.251 | 0.000 | 0.300 | -0.000 |
| static_patch | spectral_signatures | 3.501 | 0.507 | 0.400 | 1.268 |
| static_patch | anomaly_detection | 3.345 | 1.560 | 0.300 | 4.014 |
| static_patch | pruning | 3.823 | 0.000 | 0.300 | 0.000 |
| static_patch | fine_pruning | 2.450 | 0.942 | 0.400 | 1.649 |
| static_patch | model_inspection | 3.653 | 1.217 | 0.200 | 3.704 |
| clean_label | provenance_tracking | 0.561 | 0.790 | 0.500 | 0.295 |
| clean_label | influence_functions | 0.573 | 0.796 | 0.600 | 0.285 |
| label_flipping | data_cleaning | 0.603 | 0.497 | 0.200 | 0.250 |
| label_flipping | per_class_monitoring | 0.000 | 0.000 | 0.200 | 0.000 |
| label_flipping | robust_loss | 0.675 | 0.588 | 0.500 | 0.264 |
| label_flipping | dp_training | -1.449 | 0.000 | 0.700 | -0.000 |
| pgd | adversarial_training | 0.288 | 0.000 | 0.800 | 0.000 |
| pgd | randomized_smoothing | 0.070 | 0.000 | 0.500 | 0.000 |
| spsa | gradient_masking | 0.020 | 0.954 | 0.400 | 0.014 |
| spsa | jpeg_preprocessing | 0.525 | 0.691 | 0.100 | 0.330 |

## Notes

- **Mitigation Score**: Effectiveness in recovering model performance after an attack.
- **CAD (Clean Accuracy Drop)**: Degree of performance degradation on clean data.
- **Cost Score**: Relative computational/resource impact of the defense.
- **Final Score**: Aggregated score combining all metrics.
