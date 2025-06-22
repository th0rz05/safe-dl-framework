# Risk Analysis Report

**Profile**: `visitech.yaml`  
**Dataset**: `cifar10`  
**Model**: `resnet18`  
**Generated on**: 2025-06-22 17:41:35

## Overview

This report summarizes the risk associated with each attack simulated in Module 2 of the Safe-DL framework. Each attack is evaluated based on its impact (severity), likelihood of success (probability), and perceptibility (visibility). A final risk score is computed to help prioritize mitigation strategies.

## Summary Table

| Attack         | Type           |   Severity |   Probability |   Visibility |   Risk Score | Report                                                                                                   |
|----------------|----------------|------------|---------------|--------------|--------------|----------------------------------------------------------------------------------------------------------|
| label_flipping | data_poisoning |       0.57 |           1   |         0.62 |         0.79 | [Report](../../module2_attack_simulation/results/data_poisoning/label_flipping/label_flipping_report.md) |
| static_patch   | backdoor       |       0.33 |           1   |         0.6  |         0.46 | [Report](../../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md)           |
| pgd            | evasion        |       1    |           1   |         0.3  |         1.7  | [Report](../../module2_attack_simulation/results/evasion/pgd/pgd_report.md)                              |
| spsa           | evasion        |       1    |           0.8 |         0.2  |         1.44 | [Report](../../module2_attack_simulation/results/evasion/spsa/spsa_report.md)                            |

## Risk Matrix (Qualitative)

| Severity \ Probability   | Low   | Medium   | High                         |
|--------------------------|-------|----------|------------------------------|
| Low                      | -     | -        | -                            |
| Medium                   | -     | -        | label_flipping, static_patch |
| High                     | -     | -        | pgd, spsa                    |

## Risk Ranking

1. **pgd** — risk score: 1.70 → [Report](../../module2_attack_simulation/results/evasion/pgd/pgd_report.md)
2. **spsa** — risk score: 1.44 → [Report](../../module2_attack_simulation/results/evasion/spsa/spsa_report.md)
3. **label_flipping** — risk score: 0.79 → [Report](../../module2_attack_simulation/results/data_poisoning/label_flipping/label_flipping_report.md)
4. **static_patch** — risk score: 0.46 → [Report](../../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md)

## Recommendations

- **label_flipping**: Flip rate above 5%. Recommend data cleaning and per-class accuracy monitoring.
- **static_patch**: Fully blended backdoor trigger. Use activation clustering and spectral signature defenses.
- **pgd**: Very high-risk evasion attack. Recommend adversarial training and randomized smoothing
- **spsa**: Stealthy but strong evasion attack. Suggest gradient masking and input preprocessing (e.g., JPEG compression).