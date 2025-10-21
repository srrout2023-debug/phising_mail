import joblib
import numpy as np
import re
from scipy.sparse import hstack

# Load the saved model and transformers
model = joblib.load('model/best_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')
scaler = joblib.load('model/scaler.pkl')

def prepare_features_single(text):
    # numeric features (same as training)
    text_len = len(text)
    word_count = len(text.split())
    num_urls = len(re.findall(r'https?://[^\s]+', str(text)))
    suspicious_words = sum(str(text).lower().count(w) for w in ['verify','password','login','bank','account','urgent','click','update','confirm','ssn','verify'])
    upper_words = sum(1 for w in str(text).split() if w.isupper() and len(w)>1)
    num_arr = np.array([[text_len, word_count, num_urls, suspicious_words, upper_words]])
    num_s = scaler.transform(num_arr)
    text_tfidf = tfidf.transform([text])
    return hstack([text_tfidf, num_s])

# Test with sample emails
sample1 = "Your bank account is locked. Click https://fake-bank.com to verify your password."
sample2 = "Dear team, please find attached the monthly report."

for sample in [sample1, sample2]:
    X_sample = prepare_features_single(sample)
    prob = model.predict_proba(X_sample)[0,1]
    label = "Phishing" if prob > 0.5 else "Safe"
    print(f"Email: {sample[:50]}...")
    print(f"Phishing Probability: {prob:.4f}")
    print(f"Prediction: {label}")
    print()
