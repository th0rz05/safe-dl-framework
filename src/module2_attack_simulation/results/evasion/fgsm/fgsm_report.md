# Evasion Attack Report — FGSM

## Overview

- **Attack Type:** fgsm
- **Epsilon:** 0.03

## Performance Metrics

- **Accuracy on Clean Test Set (CDA):** 0.6754
- **Accuracy on Adversarial Test Set (ADA):** 0.0065

### Per‑Class Accuracy (Clean Test Set)

| Class | Accuracy |
|-------|----------|
| airplane | 0.6720 |
| automobile | 0.7830 |
| bird | 0.6380 |
| cat | 0.3570 |
| deer | 0.6360 |
| dog | 0.6080 |
| frog | 0.6940 |
| horse | 0.7840 |
| ship | 0.8620 |
| truck | 0.7200 |

### Per‑Class Accuracy (Adversarial Test Set)

| Class | Accuracy |
|-------|----------|
| airplane | 0.0040 |
| automobile | 0.0130 |
| bird | 0.0020 |
| cat | 0.0000 |
| deer | 0.0000 |
| dog | 0.0000 |
| frog | 0.0040 |
| horse | 0.0170 |
| ship | 0.0200 |
| truck | 0.0050 |

## Example Adversarial Samples

<div style="display: flex; gap: 10px;">
<div style="text-align:center;"><small>examples/fgsm_0_cat_8.png</small><br><img src="examples/fgsm_0_cat_8.png" style="width: 120px;"></div>
<div style="text-align:center;"><small>examples/fgsm_1_ship_1.png</small><br><img src="examples/fgsm_1_ship_1.png" style="width: 120px;"></div>
<div style="text-align:center;"><small>examples/fgsm_2_ship_0.png</small><br><img src="examples/fgsm_2_ship_0.png" style="width: 120px;"></div>
<div style="text-align:center;"><small>examples/fgsm_3_airplane_8.png</small><br><img src="examples/fgsm_3_airplane_8.png" style="width: 120px;"></div>
<div style="text-align:center;"><small>examples/fgsm_4_frog_4.png</small><br><img src="examples/fgsm_4_frog_4.png" style="width: 120px;"></div>
</div>