import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("intents.csv")

X = df["text"]
y = df["intent"]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=2000, C=2.0)
model.fit(X_vec, y)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("intent_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Multi‑domain intent model trained successfully")
