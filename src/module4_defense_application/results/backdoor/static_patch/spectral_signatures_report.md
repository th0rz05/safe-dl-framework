# Spectral Signatures Defense Report

**Attack Type:** static_patch

**Defense:** spectral_signatures



## Defense Parameters

- `threshold`: 0.9

## Accuracy After Defense

- **Clean Accuracy:** 0.5565
- **Adversarial Accuracy:** 0.1704

## Per-Class Accuracy (Clean)

- airplane: 0.7670
- automobile: 0.2340
- bird: 0.4910
- cat: 0.2780
- deer: 0.7400
- dog: 0.3500
- frog: 0.6050
- horse: 0.7640
- ship: 0.7170
- truck: 0.6190

## Per-Class Accuracy (Adversarial)

- airplane: 0.1780
- automobile: 0.0200
- bird: 0.0650
- cat: 0.9660
- deer: 0.0510
- dog: 0.0240
- frog: 0.0770
- horse: 0.0930
- ship: 0.1040
- truck: 0.1260

## Removed Samples Summary

- **Total Removed:** 20877

## Spectral Signature Histogram

![Spectral Histogram](./spectral_histogram.png)


## Examples of Removed Samples

- **Index**: 0, **Label**: frog

  ![Removed](./spectral_removed/removed_0_6.png)

- **Index**: 6, **Label**: bird

  ![Removed](./spectral_removed/removed_6_2.png)

- **Index**: 7, **Label**: horse

  ![Removed](./spectral_removed/removed_7_7.png)

- **Index**: 14, **Label**: truck

  ![Removed](./spectral_removed/removed_14_9.png)

- **Index**: 16, **Label**: truck

  ![Removed](./spectral_removed/removed_16_9.png)
