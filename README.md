#  Diabetes Risk Prediction Dashboard

An interactive **Streamlit** dashboard that predicts diabetes risk using machine learning models.
Built end-to-end in **Python**, it supports both **batch CSV uploads** and **single patient input**, with model performance comparison and **SHAP-based explainability**.

---

##  Features

*  **Batch Prediction** from uploaded CSV files
*  **4 Trained ML Models**:

  * Logistic Regression
  * Random Forest
  * XGBoost
  * SVM
*  **Model Evaluation**:

  * Average risk per model
  * Automatically identifies **Best**, **Average**, and **Worst** models
*  **Single Patient Prediction** via form input
*  **SHAP Summary Plot** to explain feature importance

---

## 🗂️ Project Structure

```
DiabetesInsight-ML/
├── data/                 # diabetes.csv dataset
├── src/
│   ├── train_models.py   # Trains models, saves SHAP plot and scores
│   ├── models/           # Saved ML models (.pkl)
│   └── reports/          # SHAP summary + model scores
├── dashboard/
│   └── app.py            # Streamlit frontend
├── requirements.txt
└── README.md
```

---

##  Getting Started

###  Clone the Repository

```bash
git clone https://github.com/your-username/diabetes-dashboard.git
cd diabetes-dashboard
```

###  Install Dependencies

```bash
pip install -r requirements.txt
```

###  Train the Models

```bash
python src/train_models.py
```

###  Run the Dashboard

```bash
streamlit run dashboard/app.py
```

---

##  Example Use Cases

* Upload a CSV file to evaluate diabetes risk for multiple patients.
* Compare average risk across **4 models**.
* Input values for a **single patient** to get instant prediction.
* Visualize **which features contribute most** to the prediction using SHAP.

---

##  Tech Stack

* **Python**, **Pandas**, **scikit-learn**, **XGBoost**
* **SHAP** for model explainability
* **Streamlit** for UI
* **Matplotlib** for visualizations
---

##  Contributions

Contributions are welcome!
Feel free to fork this repo, open issues, or submit pull requests.
