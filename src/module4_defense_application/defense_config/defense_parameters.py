# defense_config/defense_parameters.py

from questionary import text, select, confirm


def configure_dp_training():
    epsilon = float(text("DP ε (epsilon):", default="2.0").ask())
    delta = float(text("DP δ (delta):", default="1e-5").ask())
    clip = float(text("Gradient clipping norm:", default="1.0").ask())
    return {"epsilon": epsilon, "delta": delta, "clip_norm": clip}


def configure_robust_loss():
    loss_type = select(
        "Choose robust loss type:",
        choices=["gce", "symmetric_cross_entropy", "label_smoothing"]
    ).ask()
    return {"type": loss_type}


def configure_influence_functions():
    sample_size = int(text("Number of samples for influence estimation:", default="500").ask())
    method = select("Influence method:", choices=["grad_influence", "hessian_inverse"]).ask()
    return {"method": method, "sample_size": sample_size}


def configure_provenance_tracking():
    granularity = select("Tracking granularity:", choices=["sample", "batch"]).ask()
    return {"granularity": granularity}


def configure_data_cleaning():
    method = select("Cleaning strategy:", choices=["loss_filtering","outlier_detection"]).ask()
    threshold = float(text("Anomaly threshold (e.g., 0.9):", default="0.9").ask())
    return {"method": method, "threshold": threshold}


def configure_per_class_monitoring():
    std_threshold = float(text("Standard deviation threshold:", default="2.0").ask())
    return {"std_threshold": std_threshold}


def configure_activation_clustering():
    num_clusters = int(text("Number of clusters:", default="2").ask())
    return {"num_clusters": num_clusters}


def configure_spectral_signatures():
    threshold = float(text("SVD threshold (0–1):", default="0.9").ask())
    return {"threshold": threshold}


def configure_fine_pruning():
    pruning_ratio = float(text("Neurons to prune (0–1):", default="0.2").ask())
    return {"pruning_ratio": pruning_ratio}


def configure_model_inspection():
    layers_to_inspect = text("Comma-separated layer names to inspect:", default="fc1,fc2").ask()
    return {"layers": [l.strip() for l in layers_to_inspect.split(",")]}


def configure_anomaly_detection():
    detector_type = select("Anomaly detection type:", choices=["autoencoder", "isolation_forest", "lof"]).ask()
    return {"type": detector_type}

def configure_pruning():
    pruning_ratio = float(text("Fraction of neurons to prune (0–1):", default="0.2").ask())
    layer_scope = select("Scope of pruning:", choices=["all_layers", "last_layer_only"]).ask()
    return {"pruning_ratio": pruning_ratio, "scope": layer_scope}



def configure_gradient_masking():
    masking_strength = float(text("Gradient masking strength (0.0–1.0):", default="0.5").ask())
    return {"strength": masking_strength}


def configure_jpeg_preprocessing():
    quality = int(text("JPEG compression quality (1–100):", default="75").ask())
    return {"quality": quality}


def configure_adversarial_training():
    attack_type = select("Base attack for adversarial training:", choices=["fgsm", "pgd"]).ask()
    epsilon = float(text("Training epsilon:", default="0.03").ask())
    return {"attack_type": attack_type, "epsilon": epsilon}


def configure_randomized_smoothing():
    sigma = float(text("Noise level σ (e.g. 0.25):", default="0.25").ask())
    return {"sigma": sigma}


def configure_certified_defense():
    method = select("Certified defense method:", choices=["interval_bound", "convex_relaxation", "lipschitz_bound"]).ask()
    return {"method": method}

def configure_perturbation_detection():
    method = select("Detection method:", choices=["l2_thresholding", "statistical_test", "entropy_based"]).ask()
    threshold = float(text("Detection threshold:", default="0.1").ask())
    return {"method": method, "threshold": threshold}


# Main dispatcher
DEFENSE_PARAMETER_FUNCTIONS = {
    "dp_training": configure_dp_training,
    "robust_loss": configure_robust_loss,
    "influence_functions": configure_influence_functions,
    "provenance_tracking": configure_provenance_tracking,
    "data_cleaning": configure_data_cleaning,
    "per_class_monitoring": configure_per_class_monitoring,
    "activation_clustering": configure_activation_clustering,
    "spectral_signatures": configure_spectral_signatures,
    "fine_pruning": configure_fine_pruning,
    "model_inspection": configure_model_inspection,
    "anomaly_detection": configure_anomaly_detection,
    "gradient_masking": configure_gradient_masking,
    "pruning": configure_pruning,
    "perturbation_detection": configure_perturbation_detection,
    "jpeg_preprocessing": configure_jpeg_preprocessing,
    "adversarial_training": configure_adversarial_training,
    "randomized_smoothing": configure_randomized_smoothing,
    "certified_defense": configure_certified_defense
}
