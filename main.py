

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

# Load dataset
df = pd.read_csv("archive/emotions.csv")  

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM with grid search
param_grid = {
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
    "C": [0.1, 1, 10]
}
grid = GridSearchCV(SVC(), param_grid, cv=3)
grid.fit(X_train, y_train)

# Save best model and vectorizer
joblib.dump(grid.best_estimator_, "svm_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Training complete. Best kernel:", grid.best_params_)
