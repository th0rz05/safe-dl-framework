# defense_config/defense_tags.py

DEFENSE_TAGS = {
    "data_poisoning": {
        "label_flipping": [
            "data_cleaning",
            "per_class_monitoring",
            "robust_loss",
            "dp_training"
        ],
        "clean_label": [
            "provenance_tracking",
            "influence_functions",
            "robust_loss",
            "dp_training"
        ]
    },
    "evasion_attacks": {
        "fgsm": ["adversarial_training", "randomized_smoothing", "certified_defense"],
        "pgd": ["adversarial_training", "randomized_smoothing", "certified_defense"],
        "nes": ["gradient_masking", "jpeg_preprocessing"],
        "spsa": ["gradient_masking", "jpeg_preprocessing"],
        "cw": ["adversarial_training", "randomized_smoothing", "certified_defense"],
        "deepfool": ["adversarial_training", "randomized_smoothing", "certified_defense"],
        "transfer": ["adversarial_training", "randomized_smoothing", "certified_defense"]
    },
    "backdoor": {
        "static_patch": [
            "activation_clustering",
            "spectral_signatures",
            "anomaly_detection",
            "pruning",
            "fine_pruning",
            "model_inspection"
        ],
        "learned_trigger": [
            "activation_clustering",
            "spectral_signatures",
            "anomaly_detection",
            "pruning",
            "fine_pruning",
            "model_inspection"
        ]
    }
}
