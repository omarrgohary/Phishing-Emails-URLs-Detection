🛡️ Phishing Detection System (Email & URL)

This repository contains a Transformer-based phishing detection system capable of identifying phishing attempts in emails (subject and body) and URLs.
The system is built using DistilBERT models and provides an interactive Gradio interface for real-time predictions.

⚠️ Important:
Trained models are not included in this repository.
The codebase is intentionally designed to train models, save them to Google Drive, and later load them from Drive for inference in the interface.

📌 Features

Detects phishing emails using subject and body text

Detects phishing URLs

Uses Transformer-based NLP models (DistilBERT)

Supports training, saving, loading, and inference

Avoids retraining by reusing saved models

Includes a user-friendly Gradio web interface

Runs on CPU or GPU

🧠 Models
Email Phishing Model

Base model: distilbert-base-uncased

Architecture: DistilBERT + custom classification head

Input:

Email subject

Email body

Output:

Phishing / Legitimate

URL Phishing Model

Base model: distilbert-base-uncased

Architecture: AutoModelForSequenceClassification

Input:

URL text

Output:

Phishing / Legitimate

📂 Repository Structure
.
├── app.py                         # Gradio interface (loads models from Drive)
├── Interface.ipynb                # Notebook version of the interface
├── Model_Training_&_Saving.ipynb  # Full training & saving workflow
├── TrainEmailModel.py             # Email model training & saving
├── TrainURLModel.py               # URL model training & saving


📌 The trained models are stored in Google Drive, not in this repository.

📊 Datasets

Email dataset: CEAS_08.csv

URL dataset: new_data_urls.csv

Datasets are:

Cleaned

Deduplicated

Balanced

Preprocessed to avoid label leakage

🔐 Leakage Prevention

To prevent the models from learning explicit label cues, the following words are removed from text before training:

spam

phish

phishing

ham

This ensures the models learn semantic patterns rather than keyword hints.

📦 Training, Saving & Loading Workflow

This project follows a three-stage workflow.

1️⃣ Training

Models are trained using the provided scripts:

python TrainEmailModel.py
python TrainURLModel.py


Training includes:

Tokenization

Fine-tuning DistilBERT

Evaluation using accuracy, precision, recall, and F1-score

2️⃣ Saving Models to Google Drive

After training, models and tokenizers are automatically saved to:

/content/drive/MyDrive/phishing_models/
├── email/
│   ├── email_model.pt
│   └── tokenizer files
└── url/
    ├── model files
    └── tokenizer files


⚠️ Models are not pushed to GitHub due to size limitations.

3️⃣ Loading Models for Inference

The interface loads models directly from Google Drive:

BASE_PATH = "/content/drive/MyDrive/phishing_models"


No retraining is required

If models are missing, the interface will show a warning

This allows fast reuse of trained models

🖥️ Running the Interface

After training and saving the models:

python app.py


This launches a Gradio interface where users can:

Enter an email subject

Enter an email body

Enter a URL (optional)

The system returns:

Email prediction with confidence score

URL prediction with confidence score

⚙️ Installation

Install required dependencies:

pip install torch transformers datasets scikit-learn pandas gradio tqdm

🧪 Evaluation Metrics

Model performance is evaluated using:

Accuracy

Precision

Recall

F1-score

Metrics are displayed during training.

📌 Notes

Models are intentionally excluded from the repository

Google Drive is required for model storage

The interface depends on Drive-stored models

Supports CPU and GPU execution

Designed for reuse, experimentation, and deployment
