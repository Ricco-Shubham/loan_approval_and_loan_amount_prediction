# 🏦 Loan Approval & Loan Amount Prediction App

An end-to-end Machine Learning project that predicts:

✅ Loan Approval (Yes/No)
✅ Eligible Loan Amount (₹)

# Built using a two-stage ML pipeline and deployed as an interactive Streamlit web application.

🚀 Live Demo

👉 [Add your Streamlit/GitHub Pages link here]

📌 Project Overview

This project simulates a real-world bank loan decision system using Machine Learning.
It uses a two-stage prediction approach:

🔹 Stage 1 — Loan Approval (Classification)

Predicts whether a loan should be approved or rejected.

🔹 Stage 2 — Loan Amount (Regression)

If approved, predicts the eligible loan amount for the applicant.
This mirrors how financial institutions often separate risk assessment and amount eligibility.

# 🧠 ML Workflow
# 1️⃣ Data Processing

Cleaned and standardized dataset
Handled categorical & numerical features
Removed ID columns and irrelevant fields
Managed missing values

# 2️⃣ Feature Engineering
Column stripping & formatting
Target encoding for loan status
Separation of categorical & numerical pipelines

# 3️⃣ Preprocessing Pipelines
Numerical:
Median imputation
Standard scaling

Categorical:
Most-frequent imputation
One-Hot Encoding

# 4️⃣ Model Training

Two independent pipelines:
# ✅ Classification Model
RandomForestClassifier
Hyperparameter tuning with GridSearchCV
Stratified train-test split
F1-score optimization

Performance:
Accuracy ≈ 98%
F1 Score ≈ 0.98 

# 💰 Regression Model

RandomForestRegressor
GridSearchCV tuning
R² optimization

Performance:
R² Score ≈ 0.87
RMSE ≈ 3.27M 

# 🖥️ Streamlit App Features

✨ Clean UI with sidebar inputs
✨ Real-time predictions
✨ Currency-formatted loan output
✨ Two-stage prediction logic
✨ Modular class-based design

# 📂 Project Structure
📦 Loan-Prediction-App
│
├── models/
│   ├── stage_1_rf_classifier_pipeline.pkl
│   ├── stage_2_rf_regression_pipeline.pkl
│
├── app.py
├── requirements.txt
└── README.md

# ⚙️ Tech Stack

# Languages & Libraries:
Python
Pandas
NumPy
Scikit-learn
Joblib
ML Techniques
Pipeline architecture
ColumnTransformer
Random Forest
GridSearchCV
Feature scaling & encoding
Deployment
Streamlit


# ▶️ How to Run Locally
# 1️⃣ Clone the repo
git clone https://github.com/yourusername/loan-prediction-app.git
cd loan-prediction-app

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run Streamlit
streamlit run app.py

# 🎯 Use Cases

✔️ Banking risk assessment
✔️ FinTech credit scoring
✔️ Loan eligibility simulation
✔️ ML portfolio demonstration
✔️ Educational ML deployment example

# 📈 Future Improvements

Add SHAP explainability
Connect to real financial datasets
Deploy on AWS/Azure
Add user authentication
Build REST API version

🙋‍♂️ Author

Shubham Singh
Aspiring Data Scientist / ML Engineer

LinkedIn: https://www.linkedin.com/in/riccoshubham/

GitHub: https://github.com/Ricco-Shubham/

⭐ If you like this project

Give it a ⭐ on GitHub and feel free to fork!
