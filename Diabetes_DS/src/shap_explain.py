import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load data and model
df = pd.read_csv('data/diabetes.csv')
X = df.drop("Outcome", axis=1)

model = joblib.load("models/xgboost.pkl")  # or logistic.pkl, etc.

# SHAP explainability
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Save SHAP plot
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("reports/shap_summary_standalone.png")
