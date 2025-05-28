# Risk Analysis Report

**Profile**: `test.yaml`  
**Dataset**: `cifar10`  
**Model**: `cnn`  
**Generated on**: 2025-05-28 18:01:21

## Overview

This report summarizes the risk associated with each attack simulated in Module 2 of the Safe-DL framework. Each attack is evaluated based on its impact (severity), likelihood of success (probability), and perceptibility (visibility). A final risk score is computed to help prioritize mitigation strategies.

## Summary Table

| Attack          | Type           |   Severity |   Probability |   Visibility |   Risk Score | Report                                                                                               |
|-----------------|----------------|------------|---------------|--------------|--------------|------------------------------------------------------------------------------------------------------|
| clean_label     | data_poisoning |       0.16 |           0.9 |          0.3 |         0.24 | [Report](../../module2_attack_simulation/results/data_poisoning/clean_label/clean_label_report.md)   |
| static_patch    | backdoor       |       1    |           1   |          0.6 |         1.4  | [Report](../../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md)       |
| learned_trigger | backdoor       |       1    |           0.9 |          0.2 |         1.62 | [Report](../../module2_attack_simulation/results/backdoor/learned_trigger/learned_trigger_report.md) |

## Risk Matrix (Qualitative)

| Severity \ Probability   | Low   | Medium   | High                          |
|--------------------------|-------|----------|-------------------------------|
| Low                      | -     | -        | clean_label                   |
| Medium                   | -     | -        | -                             |
| High                     | -     | -        | static_patch, learned_trigger |

## Risk Ranking

1. **learned_trigger** — risk score: 1.62 → [Report](../../module2_attack_simulation/results/backdoor/learned_trigger/learned_trigger_report.md)
2. **static_patch** — risk score: 1.40 → [Report](../../module2_attack_simulation/results/backdoor/static_patch/static_patch_report.md)
3. **clean_label** — risk score: 0.24 → [Report](../../module2_attack_simulation/results/data_poisoning/clean_label/clean_label_report.md)

## Recommendations

- **clean_label**: Large poisoned subset detected. Use data provenance tracking or influence functions.
- **clean_label**: Low overall risk. No immediate action required, but monitor for future drift or attack evolution.
- **learned_trigger**: Fully blended backdoor trigger. Use activation clustering and spectral signature defenses.
- **learned_trigger**: Backdoor with high ASR and high risk. Suggest fine-pruning or model inspection techniques.