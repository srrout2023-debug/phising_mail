import streamlit as st
import joblib
import numpy as np
import re
import pandas as pd
from scipy.sparse import hstack

# Set page config
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light mode
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: white;
        color: black;
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        color: black;
    }
    .stTextArea textarea {
        background-color: white;
        color: black;
    }
    .stButton button {
        background-color: #007bff;
        color: white;
    }
    .stMetric {
        background-color: #f8f9fa;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Load the saved model and transformers
model = joblib.load('../model/best_model.pkl')
tfidf = joblib.load('../model/tfidf_vectorizer.pkl')
scaler = joblib.load('../model/scaler.pkl')

def prepare_features_single(text):
    # numeric features (same as training)
    text_len = len(text)
    word_count = len(text.split())
    num_urls = len(re.findall(r'https?://[^\s]+', str(text)))
    suspicious_words = sum(str(text).lower().count(w) for w in ['verify','password','login','bank','account','urgent','click','update','confirm'])
    upper_words = sum(1 for w in str(text).split() if w.isupper() and len(w)>1)
    num_arr = np.array([[text_len, word_count, num_urls, suspicious_words, upper_words]])
    num_s = scaler.transform(num_arr)
    text_tfidf = tfidf.transform([text])
    return hstack([text_tfidf, num_s]), text_len, word_count, num_urls, suspicious_words, upper_words

# Sidebar
st.sidebar.title("🛡️ Phishing Detector")
st.sidebar.write("Detect phishing emails using ML.")

# Load metrics
try:
    with open('../model/metrics.txt', 'r') as f:
        metrics = f.read()
    st.sidebar.write("**Model Performance:**")
    st.sidebar.code(metrics)
except FileNotFoundError:
    st.sidebar.write("Model: Random Forest")

st.sidebar.write("Features: TF-IDF + Numeric")

# Examples
st.sidebar.subheader("Quick Examples")
if st.sidebar.button("Load Phishing Example"):
    st.session_state.email_text = "Your bank account is locked. Click https://fake-bank.com to verify your password."
if st.sidebar.button("Load Safe Example"):
    st.session_state.email_text = "Dear team, please find attached the monthly report."

st.header("Phishing Email Detection")
st.write("Enter email text to analyze.")

email_text = st.text_area("Email Text", value=st.session_state.get('email_text', ''), height=200, key="single_text")

col1, col2 = st.columns(2)
with col1:
    predict_button = st.button("Predict", key="predict_single")
with col2:
    clear_button = st.button("Clear", key="clear_single")

if clear_button:
    st.session_state.email_text = ''
    st.rerun()

if predict_button:
    if email_text.strip():
        X_sample, text_len, word_count, num_urls, suspicious_words, upper_words = prepare_features_single(email_text)
        prob = model.predict_proba(X_sample)[0,1] if hasattr(model, "predict_proba") else model.predict(X_sample)[0]
        label = "Phishing" if prob > 0.5 else "Safe"

        st.success("Analysis Complete!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", label)
            st.metric("Probability", f"{prob:.4f}")
        with col2:
            st.progress(prob)
            st.write("Confidence Level")

        with st.expander("Extracted Features"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Text Length", text_len)
                st.metric("Word Count", word_count)
            with col2:
                st.metric("URLs", num_urls)
                st.metric("Suspicious Words", suspicious_words)
            with col3:
                st.metric("Uppercase Words", upper_words)
    else:
        st.error("Please enter email text.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Protect yourself from phishing!")
