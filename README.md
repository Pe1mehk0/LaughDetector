# LaughDetector: Acoustic Laughter Detection using LSTM

This project implements a sequence-labeling system to detect laughter in complex audio environments. Using Long Short-Term Memory (LSTM) network, the model analyzes acoustic features to identify laughter.

# 🚀 Performance Highlights
Validation F1-Score: 0.9252

Recall: 95.83% (Optimized to ensure minimal missed detections)

Accuracy: 96.07%

Precision: 89.88%

# 🛠 Project Architecture
Laughter is a context-dependent acoustic event. A standard feed-forward network treats audio frames in isolation, but this project utilizes a Bi-LSTM to capture temporal context:

Forward Pass: Captures the onset and rhythmic buildup of laughter.

Backward Pass: Captures the decaying resonance and trailing breath patterns.

Feature Engineering: Raw audio is transformed into 40 Mel-Frequency Cepstral Coefficients (MFCCs) on a Decibel (dB) scale to mimic human auditory perception.

# 📁 Repository Structure
laughdetector/: Core package containing the model architecture and data loaders.

train.py: The training pipeline with class-weighted Cross-Entropy loss.

inference.py: Script for processing new audio files using a sliding-window approach.

annotations.json: Contains the example of how annotated samples looked like.

environment.yml: Full Conda environment specification for reproducibility.

# 📊 Dataset & Annotations
To comply with privacy and storage constraints, the raw audio files are not hosted in this repository. However, the annotations.json file is provided as documentation of the data labeling process:

Format: Each entry maps a filename to a list of [start_seconds, end_seconds] laughter segments.

Data Diversity: Annotations cover various speakers and acoustic environments to ensure model robustness.

# 🧠 Future Improvments
Incorporate Data Augmentation (pitch shifting, background noise injection) to further improve generalization in noisy environments.
