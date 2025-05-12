# Defense Report — Data Cleaning

## Overview

- **Defense:** data_cleaning
- **Attack Targeted:** label_flipping
- **Cleaning Method:** loss_filtering
- **Threshold:** 0.9

## Performance Metrics

- **Accuracy After Defense:** 0.6104

### Per‑Class Accuracy

| Class | Accuracy |
|-------|----------|
| airplane | 0.7160 |
| automobile | 0.6870 |
| bird | 0.4070 |
| cat | 0.1540 |
| deer | 0.6560 |
| dog | 0.4000 |
| frog | 0.7880 |
| horse | 0.7520 |
| ship | 0.6960 |
| truck | 0.8480 |

## Cleaning Summary

- **Total Samples Removed:** 20358

## Example Removed Samples

The following examples illustrate removed samples identified as outliers or noisy instances.
Each image is named using the format:

```
removed_<index>_<label>.png
```
- `<index>`: Sample index in the dataset.
- `<label>`: Original class label.

<div style="display: flex; gap: 10px; flex-wrap: wrap;">
<div style="text-align:center;"><small>cleaned_examples/removed_0_6.png</small><br><img src="cleaned_examples/removed_0_6.png" style="width: 120px;"></div>
<div style="text-align:center;"><small>cleaned_examples/removed_4_1.png</small><br><img src="cleaned_examples/removed_4_1.png" style="width: 120px;"></div>
<div style="text-align:center;"><small>cleaned_examples/removed_5_0.png</small><br><img src="cleaned_examples/removed_5_0.png" style="width: 120px;"></div>
<div style="text-align:center;"><small>cleaned_examples/removed_9_3.png</small><br><img src="cleaned_examples/removed_9_3.png" style="width: 120px;"></div>
<div style="text-align:center;"><small>cleaned_examples/removed_11_0.png</small><br><img src="cleaned_examples/removed_11_0.png" style="width: 120px;"></div>
</div>