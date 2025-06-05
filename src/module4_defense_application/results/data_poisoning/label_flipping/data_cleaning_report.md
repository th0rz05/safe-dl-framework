# Defense Report — Data Cleaning

## Overview

- **Defense:** data_cleaning
- **Attack Targeted:** label_flipping
- **Cleaning Method:** loss_filtering
- **Threshold:** 0.9

## Performance Metrics

- **Accuracy After Defense:** 0.6251

### Per‑Class Accuracy

| Class | Accuracy |
|-------|----------|
| airplane | 0.6530 |
| automobile | 0.7290 |
| bird | 0.3520 |
| cat | 0.3810 |
| deer | 0.6560 |
| dog | 0.3480 |
| frog | 0.8870 |
| horse | 0.7040 |
| ship | 0.7370 |
| truck | 0.8040 |

## Cleaning Summary

- **Total Samples Removed:** 18607

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
<div style="text-align:center;"><small>cleaned_examples/removed_2_9.png</small><br><img src="cleaned_examples/removed_2_9.png" style="width: 120px;"></div>
<div style="text-align:center;"><small>cleaned_examples/removed_32770_5.png</small><br><img src="cleaned_examples/removed_32770_5.png" style="width: 120px;"></div>
<div style="text-align:center;"><small>cleaned_examples/removed_32771_6.png</small><br><img src="cleaned_examples/removed_32771_6.png" style="width: 120px;"></div>
<div style="text-align:center;"><small>cleaned_examples/removed_6_2.png</small><br><img src="cleaned_examples/removed_6_2.png" style="width: 120px;"></div>
</div>