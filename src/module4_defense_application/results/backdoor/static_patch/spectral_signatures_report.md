# Spectral Signatures Defense Report

**Attack Type:** static_patch
**Defense Method:** Spectral Signatures
**Threshold Used:** 0.9
**Number of Removed Samples:** 4503

## Accuracy After Defense

- **Overall Accuracy:** 0.5983

### Per-Class Accuracy
- **airplane**: 0.4740
- **automobile**: 0.7280
- **bird**: 0.5910
- **cat**: 0.5590
- **deer**: 0.4400
- **dog**: 0.6280
- **frog**: 0.7980
- **horse**: 0.5410
- **ship**: 0.6530
- **truck**: 0.5710

## Spectral Histograms

The following histograms illustrate the spectral signature magnitudes for each class.

### Class 0 hist
![class_0_hist.png](spectral_histograms/class_0_hist.png)

### Class 1 hist
![class_1_hist.png](spectral_histograms/class_1_hist.png)

### Class 2 hist
![class_2_hist.png](spectral_histograms/class_2_hist.png)

### Class 3 hist
![class_3_hist.png](spectral_histograms/class_3_hist.png)

### Class 4 hist
![class_4_hist.png](spectral_histograms/class_4_hist.png)

### Class 5 hist
![class_5_hist.png](spectral_histograms/class_5_hist.png)

### Class 6 hist
![class_6_hist.png](spectral_histograms/class_6_hist.png)

### Class 7 hist
![class_7_hist.png](spectral_histograms/class_7_hist.png)

### Class 8 hist
![class_8_hist.png](spectral_histograms/class_8_hist.png)

### Class 9 hist
![class_9_hist.png](spectral_histograms/class_9_hist.png)


## Removed Examples

The following examples were identified as suspicious and removed from the training set.

**Label:** airplane — **Index:** 29
![Removed Example](spectral_removed/removed_29_0.png)

**Label:** truck — **Index:** 208
![Removed Example](spectral_removed/removed_208_9.png)

**Label:** airplane — **Index:** 215
![Removed Example](spectral_removed/removed_215_0.png)

**Label:** dog — **Index:** 297
![Removed Example](spectral_removed/removed_297_5.png)

**Label:** truck — **Index:** 306
![Removed Example](spectral_removed/removed_306_9.png)
