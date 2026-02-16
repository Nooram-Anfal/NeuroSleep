# NeuroSleep – EEG Sleep Stage Classification with Deep Learning

NeuroSleep is an end-to-end deep learning project for automatic sleep stage classification using raw EEG signals from the Sleep-EDF dataset.

The project implements a complete pipeline: EEG preprocessing, epoch segmentation, model training, evaluation, and interactive visualization through a Streamlit dashboard.

---

## Overview

Manual sleep stage scoring is time-consuming and subject to inter-rater variability. This project automates the process using machine learning and deep learning models trained on physiological EEG data.

Sleep stages classified:

* Wake
* N1
* N2
* N3
* REM

---

## Dataset

Dataset used: Sleep-EDF Expanded Sleep Database

Data includes:

* Raw EEG recordings (.rec files)
* Expert annotations (.hyp files)
* 30-second epoch-based sleep stage labels

The dataset contains overnight polysomnography recordings used for research in sleep analysis.

---

## Project Pipeline

### 1. Data Loading

* EEG signals loaded from .rec files
* Annotations parsed from .hyp files
* Signals aligned and trimmed for consistency

### 2. Preprocessing

* Channel selection (EEG only)
* Signal normalization
* 30-second epoch segmentation
* Label encoding (Wake=0, N1=1, N2=2, N3=3, REM=4)
* Storage as NumPy arrays for efficient training

### 3. Modeling

#### Deep Learning Model – 1D CNN (PyTorch)

* 1D Convolutional layers for temporal feature extraction
* Batch Normalization
* Max Pooling layers
* Fully connected classifier
* Softmax output for multi-class prediction
* Loss: CrossEntropyLoss
* Optimizer: Adam

#### Baseline Model – Random Forest

* Statistical feature extraction
* scikit-learn implementation
* Used for performance comparison

---

## Model Performance

* CNN achieves strong multi-class classification accuracy
* Improved performance over classical ML baseline
* Evaluated using:

  * Accuracy
  * F1-score
  * Confusion Matrix

The CNN captures temporal EEG dynamics more effectively than handcrafted feature-based models.

---

## Streamlit Dashboard

An interactive web interface was built using Streamlit to:

* Visualize raw EEG signals
* Display sleep stage timelines
* Show processed data summaries
* Compare true vs predicted labels
* Display confusion matrix results

Run locally with:

```
streamlit run app.py
```

---

## Tech Stack

* Python 3.11
* PyTorch
* MNE-Python
* NumPy
* pyEDFlib
* scikit-learn
* Matplotlib
* Seaborn
* Streamlit

---

## Project Structure

```
NeuroSleep/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── cnn_model.py
│   └── random_forest.py
│
├── preprocessing/
│   └── data_loader.py
│
├── app.py
├── train.py
├── evaluate.py
└── README.md
```

---

## Key Learnings

* EEG is high-dimensional time-series data requiring careful preprocessing.
* Deep learning models (CNNs) outperform traditional ML models on temporal physiological signals.
* Biomedical ML projects require strict alignment between signals and annotations.
* Visualization improves interpretability and debugging of model predictions.

---

## Future Improvements

* Attention mechanisms for interpretable EEG segment importance
* Subject-wise cross-validation for better generalization
* Learning rate scheduling and early stopping
* Artifact removal (eye blinks, muscle noise) using advanced MNE preprocessing
* Transformer-based time-series models

---

## Installation

Clone the repository:

```
git clone <your-repo-link>
cd NeuroSleep
```

Install dependencies:

```
pip install -r requirements.txt
```

Run training:

```
python train.py
```

Launch dashboard:

```
streamlit run app.py
```

---

## Applications

* Automated sleep disorder screening
* Clinical sleep research
* Biomedical signal processing research
* Time-series deep learning experimentation

---
