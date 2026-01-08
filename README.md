# Phishing Detection System (Email & URL)

## Overview

This repository implements a Transformer-based phishing detection system capable of identifying phishing attempts in both **emails** (subject and body) and **URLs**.
The system is built using **DistilBERT** models and provides an interactive **Gradio** interface for real-time inference.

> **Important Notice**
> Trained models are intentionally **not included** in this repository.
> The project is designed to:
>
> * Train models locally or in Google Colab
> * Save trained models to **Google Drive**
> * Load models from Drive for inference without retraining

---

## Features

* Phishing detection for email subject and body
* Phishing detection for URLs
* Transformer-based NLP models (DistilBERT)
* Support for training, saving, loading, and inference
* Model reuse without retraining
* Interactive Gradio web interface
* CPU and GPU support

---

## Models

### Email Phishing Detection Model

* **Base model:** `distilbert-base-uncased`
* **Architecture:** DistilBERT with custom classification head
* **Input:**

  * Email subject
  * Email body
* **Output:**

  * Phishing
  * Legitimate

### URL Phishing Detection Model

* **Base model:** `distilbert-base-uncased`
* **Architecture:** `AutoModelForSequenceClassification`
* **Input:**

  * URL text
* **Output:**

  * Phishing
  * Legitimate

---

## Repository Structure

```text
.
├── app.py                         # Gradio interface (loads models from Google Drive)
├── Interface.ipynb                # Notebook version of the interface
├── Model_Training_&_Saving.ipynb  # End-to-end training and saving workflow
├── TrainEmailModel.py             # Email model training and saving script
├── TrainURLModel.py               # URL model training and saving script
```

> Trained models are stored in **Google Drive**, not in this repository.

---

## Datasets

* **Email dataset:** `CEAS_08.csv`
* **URL dataset:** `new_data_urls.csv`

### Dataset Preparation

* Cleaned
* Deduplicated
* Balanced
* Preprocessed to avoid label leakage

---

## Leakage Prevention

To prevent models from learning explicit label indicators, the following terms are removed during preprocessing:

* `spam`
* `phish`
* `phishing`
* `ham`

This ensures the models learn **semantic patterns** rather than keyword-based shortcuts.

---

## Training, Saving, and Loading Workflow

### 1. Training

Models are trained using the provided scripts:

```bash
python TrainEmailModel.py
python TrainURLModel.py
```

Training includes:

* Tokenization
* DistilBERT fine-tuning
* Evaluation using accuracy, precision, recall, and F1-score

---

### 2. Saving Models to Google Drive

After training, models and tokenizers are automatically saved to:

```text
/content/drive/MyDrive/phishing_models/
├── email/
│   ├── email_model.pt
│   └── tokenizer files
└── url/
    ├── model files
    └── tokenizer files
```

* Models are not pushed to GitHub due to size constraints

---

### 3. Loading Models for Inference

The interface loads models directly from Google Drive:

```python
BASE_PATH = "/content/drive/MyDrive/phishing_models"
```

* No retraining is required
* A warning is displayed if models are missing
* Enables fast reuse of trained models

---

## Running the Interface

After training and saving the models:

```bash
python app.py
```

The Gradio interface allows users to:

* Enter an email subject
* Enter an email body
* Enter a URL (optional)

The system outputs:

* Email phishing prediction with confidence score
* URL phishing prediction with confidence score

---

## Installation

Install the required dependencies:

```bash
pip install torch transformers datasets scikit-learn pandas gradio tqdm
```

---

## Evaluation Metrics

Model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

Metrics are displayed during the training process.

---

## Notes

* Trained models are intentionally excluded from the repository
* Google Drive is required for model storage
* The interface depends on Drive-stored models
* Supports both CPU and GPU execution
* Designed for reuse, experimentation, and deployment
