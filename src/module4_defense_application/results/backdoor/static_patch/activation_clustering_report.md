# Activation Clustering Report – Static Patch

## 1. Overview
- **Defense:** activation_clustering
- **Attack Type:** static_patch
- **Number of Removed Samples:** 17448
- **Defense Parameters:**
  - `num_clusters`: 2

## 2. Accuracy After Defense
- **Clean Test Set Accuracy:** `0.3735`
- **Adversarial Test Set Accuracy:** _Not available_

### Per-Class Accuracy (Clean)
- **airplane**: `0.5140`
- **automobile**: `0.0000`
- **bird**: `0.8140`
- **cat**: `0.2740`
- **deer**: `0.3810`
- **dog**: `0.3550`
- **frog**: `0.8060`
- **horse**: `0.5530`
- **ship**: `0.0260`
- **truck**: `0.0120`

## 3. Removed Sample Examples (Cluster-based)

**Removed Sample — Class: automobile**

![Removed](activation_removed/removed_5_1.png)

**Removed Sample — Class: horse**

![Removed](activation_removed/removed_12_7.png)

**Removed Sample — Class: truck**

![Removed](activation_removed/removed_14_9.png)

**Removed Sample — Class: horse**

![Removed](activation_removed/removed_16_7.png)

**Removed Sample — Class: bird**

![Removed](activation_removed/removed_18_2.png)
