!pip install gradio
!pip install -q transformers datasets torch scikit-learn pandas torchtext tqdm

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import os
from google.colab import drive

drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv("/content/new_data_urls.csv")
df = df.rename(columns={"url": "text", "status": "label"})

df['text'] = df['text'].astype(str)
df = df.dropna()

df['label'] = 1 - df['label']

N_PER_CLASS = 50000

df_0 = df[df['label'] == 0].sample(N_PER_CLASS, random_state=42)
df_1 = df[df['label'] == 1].sample(N_PER_CLASS, random_state=42)

df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

MODEL_NAME = "distilbert-base-uncased"
url_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return url_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64
    )

train_encodings = tokenize(train_texts.tolist())
test_encodings = tokenize(test_texts.tolist())

class URLDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = URLDataset(train_encodings, train_labels)
test_dataset = URLDataset(test_encodings, test_labels)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    do_train=True,
    do_eval=True
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=url_tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)

print(classification_report(test_labels, y_pred, digits=4))

BASE_PATH = "/content/drive/MyDrive/phishing_models"
os.makedirs(f"{BASE_PATH}/url", exist_ok=True)
model.save_pretrained(f"{BASE_PATH}/url")
url_tokenizer.save_pretrained(f"{BASE_PATH}/url")
