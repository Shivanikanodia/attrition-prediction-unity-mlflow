# 🧠 Employee Attrition Prediction Pipeline with MLflow, Unity Catalog & SHAP

## 📌 Problem Statement

Employee retention is critical for organizational success. This project aims to understand why and when employees are most likely to leave an organization. By identifying the key factors driving employee attrition, organizations can implement targeted strategies to improve retention and plan new hiring in advance.

This project demonstrates an end-to-end machine learning pipeline using **Databricks**. It includes data exploration with **Unity Catalog**, preprocessing, feature selection via A/B testing, model training, evaluation, and explainability using **SHAP**. **MLflow** is used extensively to track experiments, log metrics, and store artifacts for reproducibility.

---

## 🎯 Objectives

- Build an accurate attrition prediction model using clean, preprocessed data.
- Use Unity Catalog for secure and centralized data access.
- Interpret model behavior using SHAP and identify key drivers of employee churn.
- Create an MLflow pipeline for reproducibility, version control, metrics logging, and artifact tracking.

---

## 📊 Dataset Overview

- **Total records**: 1,470 employees
- **Attributes**: Demographics, job role, satisfaction levels, performance metrics, and more.

---

## 🗂️ Unity Catalog Integration

Created a Unity Catalog (`ml_catalog`) and associated schema (`ml_schema`) under a managed volume to store both the complete dataset and the training subset in Delta format.

This structure enables governed, scalable, and versioned access to data for the ML workflow.

![Unity Catalog](https://github.com/user-attachments/assets/00c377d8-3a01-47ab-9969-d4719ce93242)

<img width="1022" alt="Unity Screenshot" src="https://github.com/user-attachments/assets/741696ff-5a02-4574-b761-50424fcd0900" />

---

## 📊 Data Visualization

### 1. Years Since Last Promotion By Job Level and Attrition

<img width="1075" alt="Years Since Last Promotion" src="https://github.com/user-attachments/assets/ff088eb7-efc3-4874-8d5f-8f1d66e45913" />

- Median years since promotion is ≈1 year for both groups at entry-level.
- For job level 5, employees who left tend to have higher "Years Since Last Promotion."
- **Insight**: Delayed promotions at senior levels may increase attrition risk.

---

### 2. Daily Rate By Job Level and Attrition

<img width="1068" alt="Daily Rate" src="https://github.com/user-attachments/assets/4d0b6e6c-d0b5-4e9c-bdf4-b104623a0ef0" />

- **Sales Representatives** who left had lower hourly rates.
- **Human Resources**: Lower daily rates among those who left.
- **Insight**: Compensation disparities can drive attrition.

---

## 🧪 Feature Selection Techniques

To ensure only statistically significant features contribute to the model, univariate feature selection was applied:

### Categorical Features (vs. Categorical Target)

- **Method**: Chi-Square Test
- **Criteria**: `p < 0.05` and `Chi-Square > 15`

<img width="888" alt="Chi-Square Stats" src="https://github.com/user-attachments/assets/e46a5a90-3bb6-4f56-9998-c83d4c28120d" />

---

### Numerical Features (vs. Categorical Target)

- **Method**: ANOVA F-Test
- **Criteria**: `p < 0.05` and `F-statistic > 5`

<img width="901" alt="ANOVA Stats" src="https://github.com/user-attachments/assets/2e256d34-f5a0-4786-b2c7-137a7b2a9c38" />

---

## 🔢 Label Encoding

Transformed categorical features into numeric format using label encoding.

![Label Encoding](https://github.com/user-attachments/assets/697e7881-331c-47fc-bda3-26e926e8ae46)

---

## 🚀 Model Experimentation

<img width="1039" alt="Experimentation" src="https://github.com/user-attachments/assets/a927d8d0-5ea7-4e53-8365-fdb842b5bd62" />

---

## 📈 Model Training & Evaluation using MLflow

![MLflow Training](https://github.com/user-attachments/assets/051b5b34-ebd0-42c0-9339-789a19b74836)

---

### MLflow Metrics and Summary

<img width="1267" alt="MLflow Metrics 1" src="https://github.com/user-attachments/assets/3aa46be3-58a5-4961-91f8-b138bc220283" />


### MLFlow Tracking Dashboard

<img width="1233" alt="MLflow Metrics 2" src="https://github.com/user-attachments/assets/5c3c49d6-c90c-41f4-93d4-c038d72e7e35" />

We tracked and compared multiple models — Logistic Regression, Random Forest, and XGBoost — using MLflow. 

Each run recorded: adjusted_f1, adjusted_precision, adjusted_recall, precision, recall, f1, optimized_metric and hyperparameters

📌 This centralized tracking ensured experiment reproducibility, hyperparameter versioning, and performance benchmarking

<img width="922" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/0831eec3-0b56-492c-a56f-0f47f1ccb666" />

<img width="914" alt="Precision-Recall" src="https://github.com/user-attachments/assets/e655b1c0-9b12-438c-bb14-dda791281ab1" />

<img width="699" alt="ROC Curve" src="https://github.com/user-attachments/assets/24b77f97-4a3a-4238-81b4-8a30194d95b5" />


### Confusion Matrix

<img width="618" alt="SHAP Summary" src="https://github.com/user-attachments/assets/062074f1-5a47-4028-9b3e-767ee474cc7c" />

This matrix helps visualize how many predictions were correct (True Positives and True Negatives) versus incorrect (False Positives and False Negatives). 
The model Correctly predicted 205 no-attritions and 32 attritions and Misclassified 42 as no-attrition and 15 as attrition

🎯 This helped assess class-wise performance and diagnose bias.

---

## 📌 Conclusion

- **Promotion delays** and **low compensation** are key attrition drivers.
- Unity Catalog ensures secure, governed, and scalable data access.
- SHAP explainability provides transparency into model decisions.
- MLflow ensures experiment reproducibility and tracking for deployment.

---

## 🧰 Tech Stack

- **Languages**: Python, SQL  
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `mlflow`, `pyspark`, `shap`  
- **Platform**: Databricks  
- **Version Control & Tracking**: MLflow, Unity Catalog

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/employee-attrition-mlpipeline.git
cd employee-attrition-mlpipeline

# Install dependencies
pip install -r requirements.txt

