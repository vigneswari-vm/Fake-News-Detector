# train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Load dataset
df = pd.read_csv('news.csv')  # Make sure this file exists

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)

# Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# Save the models
pickle.dump(vectorizer, open('tfidf_model.pkl', 'wb'))
pickle.dump(model, open('classifier.pkl', 'wb'))

print("Model and Vectorizer saved successfully.")
