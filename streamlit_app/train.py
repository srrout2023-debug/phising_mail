import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import joblib
import os

# Load data
df = pd.read_csv('Phishing_Email.csv')

# Standardize column names
df = df.rename(columns=lambda c: c.strip())

# Map labels
label_map = {'Phishing Email': 1, 'Safe Email': 0}
df['label'] = df['Email Type'].map(label_map)
df = df.dropna(subset=['label'])

# Fill NaNs
df['Email Text'] = df['Email Text'].fillna('')

# Feature engineering
def count_urls(text):
    return len(re.findall(r'https?://[^\s]+', str(text)))

def count_suspicious_words(text):
    suspicious = ['verify','password','login','bank','account','urgent','click','update','confirm','ssn','verify']
    t = str(text).lower()
    return sum(t.count(w) for w in suspicious)

def count_upper_words(text):
    return sum(1 for w in str(text).split() if w.isupper() and len(w)>1)

df['text_len'] = df['Email Text'].apply(len)
df['word_count'] = df['Email Text'].apply(lambda t: len(str(t).split()))
df['num_urls'] = df['Email Text'].apply(count_urls)
df['suspicious_words'] = df['Email Text'].apply(count_suspicious_words)
df['upper_words'] = df['Email Text'].apply(count_upper_words)

feature_cols = ['text_len','word_count','num_urls','suspicious_words','upper_words']

# Split
X_text = df['Email Text']
X_num = df[feature_cols].values
y = df['label'].values

X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_text_train)
X_test_tfidf = tfidf.transform(X_text_test)

# Scale numeric
scaler = StandardScaler()
X_num_train_s = scaler.fit_transform(X_num_train)
X_num_test_s = scaler.transform(X_num_test)

# Combine
X_train = hstack([X_train_tfidf, X_num_train_s])
X_test = hstack([X_test_tfidf, X_num_test_s])

# Train models
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
f1_lr = f1_score(y_test, lr.predict(X_test))
f1_rf = f1_score(y_test, rf.predict(X_test))

best_model = rf if f1_rf >= f1_lr else lr
best_name = 'random_forest' if best_model is rf else 'logistic_regression'

print("Selected:", best_name)

# Compute train accuracy
train_pred = best_model.predict(X_train)
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, train_pred)
print(f"Train Accuracy: {train_accuracy:.4f}")

# Save
os.makedirs('model', exist_ok=True)
joblib.dump(best_model, 'model/best_model.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Save metrics
with open('model/metrics.txt', 'w') as f:
    f.write(f"Model: {best_name}\n")
    f.write(f"Train Accuracy: {train_accuracy:.4f}\n")

print("Model retrained and saved.")
