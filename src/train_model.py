from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
import time
model_out = "./out/sentiment_model.pkl"

t1 = time.perf_counter()

# Reading the dataset
df = pd.read_csv("datasets/processed/processed_data.csv", encoding="utf-8")

texts = df["Comment"]
labels = df["Sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Convert the text data into numerical features using TF-IDF
print("Vectorization")
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train a Decission Tree model
model = DecisionTreeClassifier()

print("Training model...")
model.fit(X_train_tfidf, y_train)
    
# Save model
pipeline = make_pipeline(vectorizer, model)
joblib.dump(pipeline, model_out)
print("Saved model at: ", model_out)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

t2 = time.perf_counter()

print(f"Elapsed time: {t2-t1}s")
