# Predictive Modeling with Explainability, logging using MLflow and Unity Catalog for managing and securing metadata. 

## Problem Statement
Employee retention is critical for organizational success. This project aims to understand why and when employees are most likely to leave an organization. By identifying the key factors driving employee attrition, organizations can implement targeted strategies to improve retention and plan new hiring in advance

This project demonstrates an end-to-end machine learning pipeline using Databricks. It includes data exploration with Unity Catalog, preprocessing, feature selection via A/B testing, model training, evaluation, and explainability using SHAP. MLflow is used extensively to track experiments, log metrics, and store artifacts for reproducibility.

## Objectives
Build an accurate attrition prediction model using clean, preprocessed data.

Use Unity Catalog for secure and centralized data access.

Interpret model behavior using SHAP explainability and Identify key drivers of employee churn.

Create Mlflow Pipeline for Model Reproducibility, version control and Management, logging metrics and artifacts. 

## Dataset Overview

The dataset used in this project is sourced includes data for 1,470 employees, with various attributes such as demographics, job role, satisfaction levels, and performance metrics.

## Key Steps Involved

### Unity Catalog Integration: 

Created a Unity Catalog (ml_catalog) and associated schema (ml_schema) under a managed volume to store both the complete dataset and the training subset in Delta format. This structure enables governed, scalable, and versioned access to data for machine learning workflow 

![image](https://github.com/user-attachments/assets/00c377d8-3a01-47ab-9969-d4719ce93242)

## Data Visualization:

#### 1. Years Since Last Promotion By Job Level and Attrition. 

<img width="1075" alt="Screenshot 2025-06-26 at 19 48 37" src="https://github.com/user-attachments/assets/ff088eb7-efc3-4874-8d5f-8f1d66e45913" />


- Median years since promotion is low for ≈1 year for both groups and no significant difference in promotion wait time between those who left and stayed. This suggests promotion delay isn't a strong attrition factor at entry-level positions.

- For Job level 5 & 5 Those who left (red) tend to have higher "Years Since Last Promotion" and Those who stayed (blue) have lower medians and narrower IQRs. This strongly suggests that delayed promotions at senior levels may increase attrition risk.

- Focus on promotion pathways for senior-level employees as part of your attrition mitigation strategies. 


#### 2. Daily Rate By Job Level and Attrition. 

<img width="1068" alt="Screenshot 2025-06-26 at 19 48 49" src="https://github.com/user-attachments/assets/4d0b6e6c-d0b5-4e9c-bdf4-b104623a0ef0" />

Sales Representative has Significant difference between attrition groups: those who left had noticeably lower hourly rates. This suggests undercompensation might be a driver of attrition in this role.

Human Resources with Lower daily rate range for employees who left compared to those who stayed. Potential area for pay structure revision or deeper investigation into job satisfaction. 

## To ensure that only statistically significant features contribute to the model, I applied univariate feature selection techniques tailored to the data types:

**1.For categorical features (vs. categorical target), I applied the Chi-Square test. Features with a p-value < 0.05 and a Chi-Square statistic > 15 were retained. This threshold indicates strong dependence between the feature and target variable, confirming their predictive relevance.**
 
<img width="888" alt="Screenshot 2025-06-26 at 19 58 12" src="https://github.com/user-attachments/assets/e46a5a90-3bb6-4f56-9998-c83d4c28120d" />

**2.  For numerical features (vs. categorical target), I used the ANOVA F-test. I selected features with a p-value < 0.05 (indicating statistical significance) and an F-statistic > 5, ensuring that selected features have a meaningful variance between groups and contribute to class separation.**

<img width="901" alt="Screenshot 2025-06-26 at 19 58 19" src="https://github.com/user-attachments/assets/2e256d34-f5a0-4786-b2c7-137a7b2a9c38" />

These thresholds help strike a balance between statistical rigor and practical model performance, reducing noise and enhancing model interpretability.

--

### Model Experimentation: 

<img width="1039" alt="Screenshot 2025-06-26 at 20 10 52" src="https://github.com/user-attachments/assets/a927d8d0-5ea7-4e53-8365-fdb842b5bd62" />


<img width="1267" alt="Screenshot 2025-06-26 at 19 31 20" src="https://github.com/user-attachments/assets/3aa46be3-58a5-4961-91f8-b138bc220283" />


<img width="1233" alt="Screenshot 2025-06-26 at 19 23 06" src="https://github.com/user-attachments/assets/5c3c49d6-c90c-41f4-93d4-c038d72e7e35" />

<img width="922" alt="Screenshot 2025-06-26 at 19 30 00" src="https://github.com/user-attachments/assets/0831eec3-0b56-492c-a56f-0f47f1ccb666" />

<img width="914" alt="Screenshot 2025-06-26 at 19 30 06" src="https://github.com/user-attachments/assets/e655b1c0-9b12-438c-bb14-dda791281ab1" />

<img width="699" alt="Screenshot 2025-06-26 at 18 27 03" src="https://github.com/user-attachments/assets/24b77f97-4a3a-4238-81b4-8a30194d95b5" />


<img width="618" alt="Screenshot 2025-06-26 at 18 27 13" src="https://github.com/user-attachments/assets/062074f1-5a47-4028-9b3e-767ee474cc7c" />




9. Conclusion and Sources
Summarizing insights and actionable recommendations

## Languages: 
Python, SQL
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn , mlflow, pyspark 
Version Control: Databricks Mlflow

# ⚙️ Installation

Install the required dependencies:
pip install -r requirements.txt
