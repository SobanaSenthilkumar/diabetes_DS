import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load dataset
df = pd.read_csv('../data/diabetes.csv')
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "xgboost": XGBClassifier( eval_metric="logloss"),
    "svm": SVC(probability=True)
}

results = {}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, preds)
    joblib.dump(model, f"models/{name}.pkl")
    results[name] = score

# SHAP explainability for best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)

# Save SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("reports/shap_summary.png")

# Save AUC scores
pd.Series(results).to_csv("reports/model_scores.csv")
