# Data Poisoning - Clean Label Attack Report

## Overview

- **Attack Type:** clean_label
- **Perturbation Method:** overlay
- **Poison Fraction:** 0.15
- **Target Class:** 1 (None)
- **Max Iterations:** 100
- **Epsilon:** 0.1
- **Source Selection:** random
- **Number of Poisoned Samples:** 673

## Performance Metrics

- **Accuracy After Attack:** 0.6235

### Per-Class Accuracy

| Class | Accuracy |
|--------|----------|
| airplane | 0.6270 |
| automobile | 0.6460 |
| bird | 0.4840 |
| cat | 0.4010 |
| deer | 0.7560 |
| dog | 0.4550 |
| frog | 0.6340 |
| horse | 0.7120 |
| ship | 0.7680 |
| truck | 0.7520 |

## Example Poisoned Samples

| Index | Original Label | Perturbation Norm |
|--------|----------------|-------------------|
| 40184 | automobile | 2.2323 |
| 17734 | automobile | 4.3825 |
| 38974 | automobile | 0.6147 |
| 21890 | automobile | 3.7684 |
| 19947 | automobile | 3.4391 |

## Visual Poisoned Examples (first 5)

<div style="display: flex; gap: 10px;">
<div style="text-align: center;"><small><strong>automobile</strong></small><br><img src="examples/poison_40184_automobile.png" alt="poisoned_example" style="width: 120px;"></div>
<div style="text-align: center;"><small><strong>automobile</strong></small><br><img src="examples/poison_17734_automobile.png" alt="poisoned_example" style="width: 120px;"></div>
<div style="text-align: center;"><small><strong>automobile</strong></small><br><img src="examples/poison_38974_automobile.png" alt="poisoned_example" style="width: 120px;"></div>
<div style="text-align: center;"><small><strong>automobile</strong></small><br><img src="examples/poison_21890_automobile.png" alt="poisoned_example" style="width: 120px;"></div>
<div style="text-align: center;"><small><strong>automobile</strong></small><br><img src="examples/poison_19947_automobile.png" alt="poisoned_example" style="width: 120px;"></div>
</div>
