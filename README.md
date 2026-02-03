# 🚀 Telecom AI Customer Support Copilot

An AI-powered customer support copilot for a virtual telecom company that automatically detects customer intent, analyzes sentiment, retrieves relevant policy information using Retrieval-Augmented Generation (RAG), and provides AI-suggested responses.

---

## 📌 Project Overview

Telecom companies receive thousands of customer queries related to billing, refunds, technical issues, complaints, and product inquiries. Manual handling is slow and inconsistent.

This project builds an end-to-end AI system that:
- Understands customer messages
- Predicts intent and sentiment
- Retrieves relevant company policies
- Suggests accurate responses for support agents

---

## 🧠 Features

- Intent Classification using DistilBERT  
- Sentiment Analysis using HuggingFace pipeline  
- Retrieval-Augmented Generation (RAG) with FAISS  
- Streamlit Web Application  
- Synthetic Telecom Dataset  

---

## 🗂️ Folder Structure

Telecom-ai-copilot/
│
├── app.py
├── train_intent.py
├── data/
│ └── zends_customer_queries_.csv
├── rag/
│ └── company_docs.txt
├── models/ (generated after training)
├── venv/
└── .gitignore


---

## 🛠️ Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- HuggingFace Transformers  
- Sentence-Transformers  
- FAISS  
- Streamlit  

---

## ⚙️ Installation

Create virtual environment:

python -m venv venv
venv\Scripts\activate


Install dependencies:

pip install pandas numpy scikit-learn torch transformers sentence-transformers faiss-cpu streamlit accelerate


---

## 🏋️ Train Intent Model

python train_intent.py


This will create:

models/intent_model


---

## ▶ Run Application

streamlit run app.py


Open browser and enter customer message.

---

## 🧪 Sample Inputs

My internet is not working
I want refund for my plan
Why my bill is high
What plans do you offer


---

## 📊 Output

- Predicted Intent  
- Detected Sentiment  
- Retrieved Policy Text  
- AI Suggested Reply  

---

## 🏆 Learning Outcomes

- NLP Pipeline Design  
- Transformer Model Fine-Tuning  
- RAG Architecture  
- Vector Search  
- Streamlit Deployment  
- GitHub Version Control  

---

## 📌 Note

Trained models are not pushed to GitHub due to large file size.  
Run `train_intent.py` to regenerate models.

---

## 👤 Author

Rajapandi  
