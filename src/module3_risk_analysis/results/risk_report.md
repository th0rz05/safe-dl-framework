# Risk Analysis Report

**Profile**: `test.yaml`  
**Dataset**: `mnist`  
**Model**: `cnn`  
**Generated on**: 2025-05-12 16:30:35

## Overview

This report summarizes the risk associated with each attack simulated in Module 2 of the Safe-DL framework. Each attack is evaluated based on its impact (severity), likelihood of success (probability), and perceptibility (visibility). A final risk score is computed to help prioritize mitigation strategies.

## Summary Table

| Attack          | Type           |   Severity |   Probability |   Visibility |   Risk Score | Report                                                                                                   |
|-----------------|----------------|------------|---------------|--------------|--------------|----------------------------------------------------------------------------------------------------------|
| label_flipping  | data_poisoning |       0.42 |          1    |          0.7 |         0.55 | [Report](../../module2_attack_simulation/results/data_poisoning/label_flipping/label_flipping_report.md) |
| clean_label     | data_poisoning |       0.16 |          0.9  |          0.3 |         0.24 | [Report](../../module2_attack_simulation/results/data_poisoning/clean_label/clean_label_report.md)       |
| static_patch    | backdoor       |       1    |          1    |          0.6 |         1.4  | [Report](../../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md)           |
| learned_trigger | backdoor       |       1    |          0.9  |          0.2 |         1.62 | [Report](../../module2_attack_simulation/results/backdoor/learned_trigger/learned_trigger_report.md)     |
| fgsm            | evasion        |       1    |          1    |          0.3 |         1.7  | [Report](../../module2_attack_simulation/results/evasion/fgsm/fgsm_report.md)                            |
| pgd             | evasion        |       1    |          1    |          0.3 |         1.7  | [Report](../../module2_attack_simulation/results/evasion/pgd/pgd_report.md)                              |
| cw              | evasion        |       1    |          0.9  |          0.3 |         1.53 | [Report](../../module2_attack_simulation/results/evasion/cw/cw_report.md)                                |
| deepfool        | evasion        |       1    |          1    |          0.1 |         1.9  | [Report](../../module2_attack_simulation/results/evasion/deepfool/deepfool_report.md)                    |
| nes             | evasion        |       1    |          0.8  |          0.2 |         1.44 | [Report](../../module2_attack_simulation/results/evasion/nes/nes_report.md)                              |
| spsa            | evasion        |       1    |          0.8  |          0.2 |         1.44 | [Report](../../module2_attack_simulation/results/evasion/spsa/spsa_report.md)                            |
| transfer        | evasion        |       1    |          0.85 |          0.2 |         1.53 | [Report](../../module2_attack_simulation/results/evasion/transfer/transfer_report.md)                    |

## Risk Matrix (Qualitative)

| Severity \ Probability   | Low   | Medium   | High                                                                        |
|--------------------------|-------|----------|-----------------------------------------------------------------------------|
| Low                      | -     | -        | clean_label                                                                 |
| Medium                   | -     | -        | label_flipping                                                              |
| High                     | -     | -        | static_patch, learned_trigger, fgsm, pgd, cw, deepfool, nes, spsa, transfer |

## Risk Ranking

1. **deepfool** — risk score: 1.90 → [Report](../../module2_attack_simulation/results/evasion/deepfool/deepfool_report.md)
2. **fgsm** — risk score: 1.70 → [Report](../../module2_attack_simulation/results/evasion/fgsm/fgsm_report.md)
3. **pgd** — risk score: 1.70 → [Report](../../module2_attack_simulation/results/evasion/pgd/pgd_report.md)
4. **learned_trigger** — risk score: 1.62 → [Report](../../module2_attack_simulation/results/backdoor/learned_trigger/learned_trigger_report.md)
5. **cw** — risk score: 1.53 → [Report](../../module2_attack_simulation/results/evasion/cw/cw_report.md)
6. **transfer** — risk score: 1.53 → [Report](../../module2_attack_simulation/results/evasion/transfer/transfer_report.md)
7. **nes** — risk score: 1.44 → [Report](../../module2_attack_simulation/results/evasion/nes/nes_report.md)
8. **spsa** — risk score: 1.44 → [Report](../../module2_attack_simulation/results/evasion/spsa/spsa_report.md)
9. **static_patch** — risk score: 1.40 → [Report](../../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md)
10. **label_flipping** — risk score: 0.55 → [Report](../../module2_attack_simulation/results/data_poisoning/label_flipping/label_flipping_report.md)
11. **clean_label** — risk score: 0.24 → [Report](../../module2_attack_simulation/results/data_poisoning/clean_label/clean_label_report.md)

## Recommendations

- **label_flipping**: Flip rate above 5%. Recommend data cleaning and per-class accuracy monitoring.
- **clean_label**: Large poisoned subset detected. Use data provenance tracking or influence functions.
- **clean_label**: Low overall risk. No immediate action required, but monitor for future drift or attack evolution.
- **learned_trigger**: Fully blended backdoor trigger. Use activation clustering and spectral signature defenses.
- **learned_trigger**: Backdoor with high ASR and high risk. Suggest fine-pruning or model inspection techniques.
- **fgsm**: Very high-risk evasion attack. Recommend adversarial training, randomized smoothing, or certified defenses.
- **pgd**: Very high-risk evasion attack. Recommend adversarial training, randomized smoothing, or certified defenses.
- **cw**: Very high-risk evasion attack. Recommend adversarial training, randomized smoothing, or certified defenses.
- **deepfool**: Very high-risk evasion attack. Recommend adversarial training, randomized smoothing, or certified defenses.
- **nes**: Stealthy but strong evasion attack. Suggest gradient masking and input preprocessing (e.g., JPEG compression).
- **spsa**: Stealthy but strong evasion attack. Suggest gradient masking and input preprocessing (e.g., JPEG compression).
- **transfer**: Very high-risk evasion attack. Recommend adversarial training, randomized smoothing, or certified defenses.