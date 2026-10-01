import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# 1. Load datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# 2. Add labels
fake["label"] = 0
real["label"] = 1

# 3. Combine both datasets
data = pd.concat([fake, real], ignore_index=True)

# 4. Remove rows with missing text/title
data = data.dropna(subset=["title", "text"])

# 5. Combine title and article text
data["content"] = data["title"] + " " + data["text"]

# 6. Select input and output
X = data["content"]
y = data["label"]

# 7. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_df=0.7
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 9. Train the model
print("Training model...")
model.fit(X_train, y_train)

# 10. Make predictions
y_pred = model.predict(X_test)

# 11. Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 12. Create model folder
os.makedirs("model", exist_ok=True)

# 13. Save trained model
joblib.dump(model, "model/fake_news_model.pkl")

print("\nModel saved successfully!")
print("Location: model/fake_news_model.pkl")