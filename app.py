
# To run this locally u need to download the models from drive after u save them or export the model locally after training it

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BASE_PATH = "/content/drive/MyDrive/phishing_models"


class BERTEmailClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        return self.fc(cls)

try:
    print(f"Loading email model from: {BASE_PATH}/email")
    tokenizer = DistilBertTokenizer.from_pretrained(
        f"{BASE_PATH}/email",
        local_files_only=True
    )

    email_model = BERTEmailClassifier().to(device)

    email_model.load_state_dict(
        torch.load(
            f"{BASE_PATH}/email/email_model.pt",
            map_location=device
        )
    )

    email_model.eval()
    print("Email model loaded")
except Exception as e:
    print(f"Error loading email model: {e}")
    tokenizer = None
    email_model = None


try:
    print(f"Loading URL model from: {BASE_PATH}/url")
    url_tokenizer = AutoTokenizer.from_pretrained(
        f"{BASE_PATH}/url",
        local_files_only=True
    )

    url_model = AutoModelForSequenceClassification.from_pretrained(
        f"{BASE_PATH}/url",
        local_files_only=True
    )

    url_model.to(device)
    url_model.eval()
    print("URL model loaded")
except Exception as e:
    print(f"Error loading URL model: {e}")
    url_tokenizer = None
    url_model = None

LEAKAGE_PATTERNS = [
    r"\bspam\b",
    r"\bphish\b",
    r"\bphishing\b",
    r"\bham\b"
]

def remove_leakage(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for p in LEAKAGE_PATTERNS:
        text = re.sub(p, "", text)
    return text

def predict(subject, body, url):
    outputs = []

    if (subject.strip() or body.strip()) and email_model is not None:
        text = remove_leakage(subject) + " [SEP] " + remove_leakage(body)

        enc = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = email_model(enc["input_ids"], enc["attention_mask"])
            probs = F.softmax(logits, dim=1)[0]

        label = "Phishing" if probs[1] > 0.5 else "Legitimate"
        confidence = float(probs[1] if label == "Phishing" else probs[0])

        outputs.append(f"Email Prediction: {label} ({confidence*100:.2f}%)")
    else:
        if email_model is None:
             outputs.append("Email Prediction: Model not loaded (Check paths)")
        else:
             outputs.append("Email Prediction: No email provided")

    if url.strip() and url_model is not None:
        inputs = url_tokenizer(
            [url],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = url_model(**inputs).logits
            probs = F.softmax(logits, dim=1)[0]

        label = "Phishing" if probs[1] > 0.5 else "Legitimate"
        confidence = float(probs[1] if label == "Phishing" else probs[0])

        outputs.append(f"URL Prediction: {label} ({confidence*100:.2f}%)")
    else:
        if url_model is None:
            outputs.append("URL Prediction: Model not loaded (Check paths)") 

    return "\n".join(outputs)

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Email Subject"),
        gr.Textbox(lines=8, label="Email Body"),
        gr.Textbox(label="URL (optional)")
    ],
    outputs=gr.Textbox(label="Predictions"),
    title="Phishing Detection System",
    description="Enter an email subject & body and/or a URL to check if it is phishing."
)

if __name__ == "__main__":
    interface.launch(share=True, inbrowser=True)
