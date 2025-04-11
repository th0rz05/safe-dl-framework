# Data Poisoning - Clean Label Attack Report

## Overview

- **Attack Type:** clean_label
- **Perturbation Method:** feature_collision
- **Poison Fraction:** 0.15
- **Target Class:** 9 (truck)
- **Max Iterations:** 100
- **Epsilon:** 0.1
- **Source Selection:** most_confident
- **Number of Poisoned Samples:** 675

## Performance Metrics

- **Accuracy After Attack:** 0.6276

### Per-Class Accuracy

| Class | Accuracy |
|--------|----------|
| airplane | 0.6760 |
| automobile | 0.7970 |
| bird | 0.5360 |
| cat | 0.4300 |
| deer | 0.5120 |
| dog | 0.5670 |
| frog | 0.6650 |
| horse | 0.6210 |
| ship | 0.7490 |
| truck | 0.7230 |

## Example Poisoned Samples

| Index | Original Label | Perturbation Norm |
|--------|----------------|-------------------|
| 5671 | truck | 0.0885 |
| 40734 | truck | 0.0737 |
| 8125 | truck | 0.0821 |
| 25708 | truck | 0.0737 |
| 35047 | truck | 0.0533 |

## Visual Poisoned Examples (first 5)

<div style="display: flex; gap: 10px;">
<div style="text-align: center;"><small><strong>truck</strong></small><br><img src="examples/poison_5671_truck.png" alt="poisoned_example" style="width: 120px;"></div>
<div style="text-align: center;"><small><strong>truck</strong></small><br><img src="examples/poison_40734_truck.png" alt="poisoned_example" style="width: 120px;"></div>
<div style="text-align: center;"><small><strong>truck</strong></small><br><img src="examples/poison_8125_truck.png" alt="poisoned_example" style="width: 120px;"></div>
<div style="text-align: center;"><small><strong>truck</strong></small><br><img src="examples/poison_25708_truck.png" alt="poisoned_example" style="width: 120px;"></div>
<div style="text-align: center;"><small><strong>truck</strong></small><br><img src="examples/poison_35047_truck.png" alt="poisoned_example" style="width: 120px;"></div>
</div>
