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
    "evasion": {
        "fgsm": ["adversarial_training", "randomized_smoothing"],
        "pgd": ["adversarial_training", "randomized_smoothing"],
        "nes": ["gradient_masking", "jpeg_preprocessing"],
        "spsa": ["gradient_masking", "jpeg_preprocessing"],
        "cw": ["adversarial_training", "randomized_smoothing"],
        "deepfool": ["adversarial_training", "randomized_smoothing"],
        "transfer": ["adversarial_training", "randomized_smoothing"]
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
