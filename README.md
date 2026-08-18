# Deepfake Image Detection Using Deep Learning Approaches

This repository contains the code notebooks, datasets, and frameworks for evaluating and improving the generalization of deepfake detection models. The study assesses six state-of-the-art (SOTA) models from the ForensicHub framework on the Robust Deepfake Detection Challenge (RDDC) dataset. To improve robustness, this project implements targeted dataset expansion, soft-voting ensembles, and a custom TruFor-guided observation reweighting approach within the loss function for the Effort model. It also includes a replication study of the winning RDDC 2026 solution, ShallowReal DINO-MAC.

## Repository Structure

* **`Code Notebooks/`**: Contains the primary scripts and notebooks used for model training, evaluation, and data processing.
* **`Database/`**: Stores the datasets utilized for training and testing the models.
    * **`RDDC_Dataset/`**: The original Robust Deepfake Detection Challenge dataset used as the baseline for evaluation.
    * **`Dataset Expansion Resources/`**: Additional benchmark data sourced from DF40, HiDF, and WhichFaceIsReal used to address specific misclassification patterns.
    * **`3440_Train_Dataset/`**: The fully expanded and balanced training dataset, consisting of 1,720 real images and 1,720 manipulated images.
* **`Frameworks/`**: Houses the specific detection frameworks and model implementations evaluated in this study.
    * **`ForensicHub/`**: The baseline framework used to evaluate the Effort, TruFor, ResNet, Capsule-Net, SPSL, and RECCE models.
    * **`DeepfakeBench/`**: A supplementary framework required as a dependency by ForensicHub.
    * **`IMDLBenCo/`**: A supplementary framework for image manipulation detection and localization, also utilized as a dependency by ForensicHub.
    * **`DinoV3/`**: Contains the architecture and training pipeline for the DINO-MAC replication study.
* **`Deepfake.pdf`**: The complete research paper detailing the methodology, experiments, and full results of this study.

## Methodology

* **Dataset Expansion**: The RDDC training set was expanded using actively degraded images and additional datasets to simulate severe real-world post-processing conditions like blurring, compression, and noise.
* **Soft-Voting Ensembles**: The methodology pairs the Effort model with other detectors (TruFor, SPSL, and Capsule-Net) to average predicted probabilities and improve detection rates.
* **TruFor-Guided Loss Reweighting**: This approach utilizes a frozen TruFor model to generate manipulation-evidence scores, which are then used to reweight the cross-entropy loss of the Effort model during training.
* **DINO-MAC Replication**: An implementation of the DINO-MAC architecture utilizing a DINOv3-Base backbone and Multi-Aspect Classification head, adapted for single-GPU constraints.

## Key Results

* The baseline Effort model achieved 71% accuracy and an AUC of 0.76 on the public test set.
* The TruFor-guided Effort loss reweighting approach achieved the best overall performance, improving upon the baseline with 73% accuracy, 0.78 AUC, and a 0.69 F1 score.
* The DINO-MAC small replication achieved a 0.66 F1 score and 0.73 AUC, demonstrating competitive performance under reduced hardware settings.

## Authors
* Hripsime Soghomonyan
* Aram Butavyan