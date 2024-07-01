import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from imblearn.ensemble import EasyEnsembleClassifier
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset
nltk.download('stopwords')

st.title("Research Paper Classifier")
def load_data():
    # Replace the URL with the path to your data file
    url = 'train.csv'  # Replace with the actual path or URL
    data = pd.read_csv(url)
    return data
data=load_data()

y = data['Categories']
mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(y.apply(eval))

def load_model():
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('scibert')  # Ensure 'scibert_final' directory contains the model files
    return tokenizer, model

tokenizer, model = load_model()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])       
    return text

def tokenize_text(text):
    return tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

title = st.text_input("Title", "Enter the title of the research paper")
abstract = st.text_area("Abstract", "Enter the abstract of the research paper")

if st.button("Predict Categories"):
    if title and abstract:
        # Preprocess the input
        processed_text = preprocess_text(title + " " + abstract)
        inputs = tokenize_text(processed_text)

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Convert logits to probabilities and then to predicted categories
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        threshold = 0.5
        predicted_labels = (probabilities > threshold).astype(int)
        predicted_labels = predicted_labels.reshape(1, -1)

        # Get the predicted category names
        predicted_categories = mlb.inverse_transform(predicted_labels)

        # Display predicted categories
        st.write("Predicted Categories:")
        st.write(predicted_categories)

    else:
        st.write("Please enter both the title and abstract.")
