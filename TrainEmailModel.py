!pip install gradio
!pip install -q transformers datasets torch scikit-learn pandas torchtext tqdm

import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm.auto import tqdm
import os
from google.colab import drive

drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

email_df = pd.read_csv("/content/CEAS_08.csv", engine="python", on_bad_lines="skip")
print("Email shape:", email_df.shape)

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

email_df["subject"] = email_df["subject"].apply(remove_leakage)
email_df["body"] = email_df["body"].apply(remove_leakage)
email_df["sender"] = email_df["sender"].astype(str)

email_df["text_raw"] = (
    email_df["subject"].fillna("") + " " +
    email_df["body"].fillna("")
)

before = len(email_df)
email_df = email_df.drop_duplicates(subset="text_raw")
after = len(email_df)

print(f"Removed {before - after} duplicate emails")

email_df["text"] = (
    email_df["subject"].fillna("") + " [SEP] " +
    email_df["body"].fillna("")
)

email_texts = email_df["text"].tolist()
email_labels = email_df["label"].astype(int).tolist()

X_train_e, X_val_e, y_train_e, y_val_e = train_test_split(
    email_texts,
    email_labels,
    test_size=0.2,
    random_state=42,
    stratify=email_labels
)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_emails(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

train_enc_e = tokenize_emails(X_train_e)
val_enc_e   = tokenize_emails(X_val_e)

class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

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

train_ds_e = EmailDataset(train_enc_e, y_train_e)
val_ds_e   = EmailDataset(val_enc_e, y_val_e)

train_loader_e = DataLoader(train_ds_e, batch_size=16, shuffle=True)
val_loader_e   = DataLoader(val_ds_e, batch_size=16)

email_model = BERTEmailClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(email_model.parameters(), lr=2e-5)

def train_email_epoch(model, loader, epoch, total_epochs):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Training]", leave=False)
    for batch in pbar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def eval_email(model, loader, epoch, total_epochs):
    model.eval()
    preds, labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Evaluating]", leave=False)
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            out = model(input_ids, mask)
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    return {
        "acc": accuracy_score(labels, preds),
        "prec": precision_score(labels, preds),
        "rec": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

EPOCHS = 2

for epoch in range(1, EPOCHS + 1):
    loss = train_email_epoch(email_model, train_loader_e, epoch, EPOCHS)
    metrics = eval_email(email_model, val_loader_e, epoch, EPOCHS)
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {loss:.4f} | Acc: {metrics['acc']:.4f}")

BASE_PATH = "/content/drive/MyDrive/phishing_models"
os.makedirs(f"{BASE_PATH}/email", exist_ok=True)
torch.save(email_model.state_dict(), f"{BASE_PATH}/email/email_model.pt")
tokenizer.save_pretrained(f"{BASE_PATH}/email")
