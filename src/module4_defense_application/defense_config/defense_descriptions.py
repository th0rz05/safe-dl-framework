# defense_config/defense_descriptions.py

DEFENSE_DESCRIPTIONS = {

    # Data Poisoning
    "data_cleaning": "Remove corrupted or mislabeled samples using heuristics, clustering, or manual review.",
    "per_class_monitoring": "Track class-specific accuracy to catch attacks that only affect certain classes.",
    "robust_loss": "Use robust loss functions (e.g., MAE, TRIM) to reduce influence of poisoned samples.",
    "dp_training": "Train with differential privacy to mitigate the effect of individual malicious samples.",
    "provenance_tracking": "Track data sources to identify and filter potentially malicious origins.",
    "influence_functions": "Estimate how much each sample affects the model to detect high-influence poisons.",

    # Evasion
    "adversarial_training": "Include adversarial examples in training to increase robustness.",
    "randomized_smoothing": "Add noise to inputs and average predictions to gain certified robustness.",
    "certified_defense": "Use provable guarantees against attacks within certain norms.",
    "gradient_masking": "Obfuscate gradients to prevent attackers from computing effective perturbations.",
    "jpeg_preprocessing": "Apply JPEG compression to remove high-frequency adversarial noise.",

    # Backdoor
    "activation_clustering": "Cluster neuron activations to separate clean and poisoned data.",
    "spectral_signatures": "Use spectral analysis of activations to detect outlier backdoor signals.",
    "anomaly_detection": "Use statistical or machine learning methods to detect poisoned inputs.",
    "pruning": "Remove neurons that react strongly to trigger patterns.",
    "fine_pruning": "Finetune and prune the model selectively to remove triggers while preserving accuracy.",
    "model_inspection": "Analyze model parameters for suspicious patterns introduced during training."
}
