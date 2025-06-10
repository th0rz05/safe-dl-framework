# Safe-DL Framework - Final Security Report
**Profile Selected**: `test.yaml`
**Report Generated On**: 2025-06-10 19:58:33

---

## 1. Introduction and Overview
This comprehensive report aggregates the findings from the Safe-DL framework's security assessment, covering threat modeling, attack simulation, risk analysis, defense application, and defense evaluation. Its purpose is to provide a unified overview of the deep learning model's security posture against adversarial threats, document the mitigation strategies applied, and quantify their effectiveness. This dossier serves as a critical resource for decision-making regarding model deployment and continuous security improvement.

## 2. System Under Evaluation Details
This section details the core components of the system analyzed in this report, as defined in the selected threat profile.

### 2.1 Model Details
- **Name**: `cnn`
- **Type**: `builtin`
- **Input Shape**: `[3, 32, 32]`
- **Number of Classes**: `10`
- **Parameters**: `{'conv_filters': 32, 'hidden_size': 128}`


### 2.2 Dataset Details
- **Name**: `cifar10`
- **Type**: `builtin`
