import pandas as pd
import numpy as np
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
import neattext.functions as nt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("emotion_sentimen_dataset.csv")
df.head()

def clean_text(text):
    text = nt.remove_emojis(text)
    text = nt.remove_urls(text)
    text = nt.remove_puncts(text)
    text = nt.remove_numbers(text)
    text = text.lower().strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)
df.head()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Emotion", hue="Emotion", order=df["Emotion"].value_counts().index, palette="viridis", legend=False)
plt.title("Emotion Distribution")
plt.xticks(rotation=45)
plt.show()
top_emotions = df["Emotion"].value_counts().index  
min_count = df["Emotion"].value_counts().min()
df_balanced = pd.concat([df[df["Emotion"] == emo].sample(min_count, random_state=42) for emo in top_emotions])

plt.figure(figsize=(8, 5))
sns.countplot(data=df_balanced, x="Emotion", hue="Emotion", order=top_emotions, palette="coolwarm", legend=False)
plt.title("Balanced Emotion Distribution")
plt.xticks(rotation=45)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["clean_text"], df_balanced["Emotion"], test_size=0.2, stratify=df_balanced["Emotion"], random_state=42
)

vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1,2), stop_words='english', sublinear_tf=True, smooth_idf=True)  
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=120, max_depth=20, min_samples_split=3, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced")  
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "emotion_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

def predict_emotion(user_input):
    model = joblib.load("emotion_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    user_input_clean = clean_text(user_input)
    user_vec = vectorizer.transform([user_input_clean])
    return model.predict(user_vec)[0]

while True:
    user_input = input("Enter a sentence (or '1' to stop): ").strip()
    if user_input.lower() == "1":
        break
    print(f"Predicted Emotion: {predict_emotion(user_input)}\n")
