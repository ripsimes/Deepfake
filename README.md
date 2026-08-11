\# Deepfake Image Detection Using Deep Learning Approaches



This repository contains the source code and experimental framework for the research study: \*\*"Deepfake Image Detection Using Deep Learning Approaches"\*\* (authored by Hripsime Soghomonyan and Aram Butavyan). 



This work evaluates the generalization of state-of-the-art (SOTA) deepfake detection models on the \*\*Robust Deepfake Detection Challenge (RDDC) 2026\*\* dataset, proposing targeted strategies to improve detection robustness under constrained computational settings.



\---



\## 🔍 Research Highlights

\* \*\*Baseline Evaluation:\*\* Evaluated six representative models from the ForensicHub framework: \*Effort, TruFor, ResNet, Capsule-Net, SPSL, and RECCE\*.

\* \*\*Dataset Expansion:\*\* Targeted expansion of the training set from 1,000 to 3,440 images, incorporating curated samples from DF40 and HiDF to improve robustness against blur, compression, and noise.

\* \*\*TruFor-Guided Reweighting:\*\* Developed a novel "Effort Adjusted Loss" approach that uses frozen TruFor forensic evidence scores to reweight the training loss of the \*Effort\* detector, achieving the best performance (73% accuracy, 0.78 AUC).

\* \*\*Replication Study:\*\* Conducted a small-scale replication of the RDDC 2026 winning solution, \*ShallowReal DINO-MAC\*, optimized for single NVIDIA T4 GPU constraints.



\---



\## 📂 Repository Structure



```text

├── Database/                 # Dataset expansion resources, configurations, and scripts

├── Frameworks/               

│   ├── DinoV3/               # DINO-MAC replication code, LoRA configs, and training logs

│   ├── ForensicHub/          # ForensicHub pipeline implementations, tests, and logs

│   └── DeepfakeBench/        # DeepfakeBench integration modules

├── Paper/                    # Research paper (Deepfake.pdf)

└── requirements.txt          # Python dependencies

