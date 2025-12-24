🔐 Phishing Detection System using Deep Learning

Email & URL Phishing Detection with BERT and Transformer Models

----------------------------------------------------------------------------------------------------------------------------------------------------------------
📌 Project Overview:

Phishing attacks remain one of the most common cybersecurity threats, targeting users through malicious emails and fraudulent URLs.
This project presents a dual-model phishing detection system that identifies phishing attempts using deep learning and NLP techniques.

The system consists of:
  Email Phishing Detection using DistilBERT
  URL Phishing Detection using a Hybrid Transformer + Feature-Based Model
  Both models are trained, evaluated, and tested independently using real-world datasets.
  
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Key Features:
  Uses state-of-the-art NLP models (DistilBERT) for email classification
  Character-level Transformer encoder for URL analysis
  Combines engineered URL features with learned representations
  Prevents data leakage by removing label-related keywords
  Supports real-world inference on unseen emails and URLs
  GPU-accelerated (runs on CUDA if available)

----------------------------------------------------------------------------------------------------------------------------------------------------------------
Datasets Used
  A.Email Dataset
    CEAS 2008 Email Dataset
    Fields used: subject, body, sender, label
    Labels:
      0 → Legitimate
      1 → Phishing

  B.URL Dataset
    PhiUSIIL Phishing URL Dataset
    Includes:
      Raw URLs
      18 engineered features (lengths, ratios, HTTPS flag, obfuscation metrics)
      
----------------------------------------------------------------------------------------------------------------------------------------------------------------
System Architecture
  1.Email Phishing Model (NLP-Based)
    Tokenizer: DistilBertTokenizer
    Backbone: DistilBertModel
    Classifier: Fully Connected + Dropout

    A.Input:
      Subject [SEP] Body
      Loss: Cross-Entropy
      Optimizer: AdamW
      Max Sequence Length: 256

    B.Evaluation Metrics:
      Accuracy
      Precision
      Recall
      F1-Score


  2️.URL Phishing Model (Hybrid Approach)
    This model combines two information sources:

  A.Character-Level Encoding
    URLs encoded character-by-character
    Learned embeddings + positional encoding
    Transformer Encoder (2 layers)

  B.Engineered Features

  Examples:
    URL length
    Domain length
    HTTPS usage
    Digit & special character ratios
    Obfuscation indicators

  Final Architecture:
    [Transformer Output] + [Feature MLP] → Classifier


  Evaluation Metrics:
    Accuracy
    Precision
    Recall
    F1-Score

----------------------------------------------------------------------------------------------------------------------------------------------------------------

Data Leakage Prevention:
To ensure fair evaluation, label-leaking keywords are removed during preprocessing:
spam, phish, phishing, ham
This prevents the model from learning trivial shortcuts.

----------------------------------------------------------------------------------------------------------------------------------------------------------------
Installation & Setup
  A.Clone the Repository
    git clone https://github.com/your-username/phishing-detection-system.git
    cd phishing-detection-system

  B.Install Dependencies
    pip install torch transformers torchtext tqdm pandas numpy scikit-learn

The project was developed and tested on Google Colab with GPU support.

----------------------------------------------------------------------------------------------------------------------------------------------------------------
How to Run:
  Upload the datasets when prompted (Google Colab)
  Run all cells sequentially

The script will:
  Train both models
  Evaluate performance
  Test on unseen CSV samples
  Run real-world email predictions

Sample Output:
  Email Prediction:
    Subject: Urgent: Verify your bank account immediately
    Prediction: Phishing

  URL Prediction:
    URL: http://secure-login-paypal.verify-user.com
    Prediction: Phishing
    
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Results Summary:
Both models achieve strong performance across all evaluation metrics, demonstrating the effectiveness of:
Transformer-based NLP for email security
Hybrid deep learning + feature engineering for URL analysis

----------------------------------------------------------------------------------------------------------------------------------------------------------------
Use Cases:
  Email security gateways
  SOC phishing analysis tools
  Academic research in cybersecurity & NLP
  ML-based fraud detection systems

----------------------------------------------------------------------------------------------------------------------------------------------------------------
Technologies Used:

  Python 3
  PyTorch
  HuggingFace Transformers
  Scikit-learn
  Google Colab (CUDA)

----------------------------------------------------------------------------------------------------------------------------------------------------------------
Author:

  Omar EL Gohary | Linkedin: https://www.linkedin.com/in/omarelgohary2003/ | Penetration Testing | Network Security | AI for Security
  
  Youssef Mohamed Azmy | Linkedin: https://www.linkedin.com/in/youssef-azmy/ |  Machine Learning & Deep Learning Enthusiast
