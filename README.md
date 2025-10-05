## Workforce Attrition Prediction with Explainable and Reproducible ML

Predicting employee churn and identifying key drivers of attrition using scalable machine learning practices. Built with  classification and booster algorithms, SHAP for explainability and trustworthy insights, Unity Catalog to store models and feature, and MLflow for experiment tracking, model reproducibility and transparency.

### Problem Statement:

**Employee attrition poses a significant risk to organizational stability and workforce planning. This project focuses on predicting which employees are most likely to leave and uncovering the core reasons behind employees leaving organisation.**

**By leveraging machine learning models, MLflow, and Unity Catalog, the goal is to empower HR teams with proactive insights—helping them improve retention, reduce the high cost of unplanned exits, and recommend policies or strategies based on results to improve employee retention.**

*Given that we have data on former employees, this is a standard supervised classification problem where the label is a binary variable, 0 (likely to stay), 1 (likely to leave). In this study, our target variable Y is the probability of an employee leaving the company.*

**This is a complete, production-ready ML pipeline built on Databricks, covering:**

- Data Management with Unity Catalog (secure, versioned Delta tables)

- Exploratory Analysis to uncover attrition trends by Job Role, Income and Daily Rate, and Promotion History

- Feature Selection using statistical tests (Chi-Square, ANOVA)

- Model Development with tuned XGBoost

- Explainability using SHAP for transparent insights

- Model Tracking with MLflow (parameters, metrics, artifacts, comparison)

  Each stage of the pipeline was built for reproducibility, scalability, and interpretability.

---

### Dataset Overview:

- **Total records**: 1,470 employees
- **Attributes**: Demographics, Job Role, Satisfaction levels, Performance metrics etc.

---

### Unity Catalog for Data Storage and Access Control:

Created a Catalog (`ml_catalog`) and associated schema (`ml_schema`) under a managed volume to store both the complete dataset and the training subset in Delta format.

This structure enables governed, scalable, and versioned access to data for the ML workflow.

![image](https://github.com/user-attachments/assets/dd4dcbdf-1cbb-420d-bd80-874eba5f51f8)



![Unity Catalog](https://github.com/user-attachments/assets/00c377d8-3a01-47ab-9969-d4719ce93242)



Below code describes all the tables stored in unity catalog, which is reliable data tracking for ML pipeline versioning.


<img width="1022" alt="Unity Screenshot" src="https://github.com/user-attachments/assets/741696ff-5a02-4574-b761-50424fcd0900" />

---

### Data Visualization:



### Feature Selection Techniques:


To ensure only statistically significant features contribute to the model, univariate feature selection was applied, both the tables are stored in **Unity Catalog**

**ml_catalog.ml_schema.anova_results**
**ml_catalog.ml_schema.chisquare_results**


### Categorical Features vs Categorical Target

- Used Chi-Square Test to test whether there's an association between our  two categorical variables. (Eg., Attrition AND Job Roles (Sales, HR, IT))
- Features with `p < 0.05` and `Chi-Square > 15` were selected from the testing results to further add into our analysis.

- If the observed vs. expected frequencies differ significantly, the feature is likely influencing the target.

<img width="888" alt="Chi-Square Stats" src="https://github.com/user-attachments/assets/e46a5a90-3bb6-4f56-9998-c83d4c28120d" />


---


### Numerical Features vs Categorical Target

- Used ANOVA F-Test to measure whether the means of a our numerical feature differ significantly across the classes (Yes/No) of the target - Attrition.
- Features with p values < 0.05` states that the difference is statistically significant and `F-statistic > 5` were selected from the test.


<img width="901" alt="ANOVA Stats" src="https://github.com/user-attachments/assets/2e256d34-f5a0-4786-b2c7-137a7b2a9c38" />


---




---


## 🚀 Model Experimentation:


Setting the Experimentation inside Databricks Notebook. 

<img width="1039" alt="Experimentation" src="https://github.com/user-attachments/assets/a927d8d0-5ea7-4e53-8365-fdb842b5bd62" />


---


## 📈 Model Training & Evaluation using MLflow


This centralized tracking ensured experiment reproducibility, hyperparameter versioning, and performance benchmarking

Logged key hyperparameters, evaluation metrics, trained model and visual artifacts like confusion matrix for every run — making it easy to reproduce or explain later.

![MLflow Training](https://github.com/user-attachments/assets/051b5b34-ebd0-42c0-9339-789a19b74836)


---


### MLflow Metrics and Summary: 


As we can see screenshot below from Databricks MLFlow UI with Run Name, Duration of each Run and metrics logged. 

Used the MLflow UI in Databricks to compare multiple runs of Logistic Regression, Random Forest, and XGBoost. Selected the best model based on precision-recall trade-off and registered it using Model Resgistry in databricks to serve it later for deployment and making production level predictions.

We used precision-recall curve evaluation and selected an optimal threshold to minimize false negatives while avoiding unnecessary false alarms of employee leaving.


<img width="1267" alt="MLflow Metrics 1" src="https://github.com/user-attachments/assets/3aa46be3-58a5-4961-91f8-b138bc220283" />


<img width="1233" alt="MLflow Metrics 2" src="https://github.com/user-attachments/assets/5c3c49d6-c90c-41f4-93d4-c038d72e7e35" />



### MLFlow Tracking Dashboard:


I tracked and compared multiple models — Logistic Regression, Random Forest, and XGBoost — using MLflow. 

Each dashboard recorded: adjusted_f1, adjusted_precision, adjusted_recall, precision, recall and f1 score. 


<img width="922" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/0831eec3-0b56-492c-a56f-0f47f1ccb666" />


--


<img width="914" alt="Precision-Recall" src="https://github.com/user-attachments/assets/e655b1c0-9b12-438c-bb14-dda791281ab1" />


--

### Model Serving and Registry: 

I've provided signatures examples to be used by registered and served model for making predictions at endpoint. Below is snapshot for same. 

<img width="1784" height="1412" alt="image" src="https://github.com/user-attachments/assets/0fcedbef-b014-419f-9714-b29c7d71a4d1" />



### SHAP Explainability:


SHAP analysis identified the most influential features in predicting attrition, with DailyRate, MonthlyIncome, TotalWorkingYears, and YearsAtCompany leading the list. DailyRate had the strongest impact, followed by tenure-related features.
High values (pink) and low values (blue) were assessed for their influence on predictions, offering clear and transparent insights into model behavior.


<img width="699" alt="ROC Curve" src="https://github.com/user-attachments/assets/24b77f97-4a3a-4238-81b4-8a30194d95b5" />


--


### Confusion Matrix:

The confusion matrix provides a clear view of the model’s performance by showing how many predictions were correct (True Positives and True Negatives) versus incorrect (False Positives and False Negatives).

In our case, the model:

 - Correctly predicted 205 non-attritions (True Negatives) and 32 attritions (True Positives)
 - Misclassified 42 instances as attrition when it was not (False Positives) and 15 instances as non-attrition when attrition actually     occurred (False Negatives)

This model significantly outperformed others like Random Forest and Logistic Regression, both of which produced a much higher number of False Negatives—a critical metric in our use case, as missing true attrition cases could lead to substantial business impact. Reducing False Negatives was a top priority, and this model demonstrated a clear advantage in that regard.

**True Negatives (TN)**: 205 — Correctly predicted “no attrition”
**True Positives (TP)**: 32 — Correctly predicted “attrition”
**False Positives (FP)**: 42 — Predicted “attrition” but employee stayed
**False Negatives (FN)**: 15 — Predicted “stay” but employee left (high risk)

- Lower false negatives improve recall — crucial for catching real churn cases.


![image](https://github.com/user-attachments/assets/57be8037-3c30-4c01-ac9e-fd2fa3967109)


---

### Model Registry:


<img width="878" alt="Screenshot 2025-07-01 at 15 35 05" src="https://github.com/user-attachments/assets/8c874b9d-2f28-4a8b-97b8-a2c591cf2c37" />


## Conclusion:

- After evaluating multiple models—including Logistic Regression, Random Forest, and XGBoost—XGBoost emerged as the best-performing model for our attrition prediction task. With a tuned threshold of 0.21, it struck an effective balance between interpretability, performance, and generalization.

- The model achieved a recall of 68% and precision of 43% on the minority class (attrition), which is crucial for early risk detection while minimizing false positives that may harm employee trust. Compared to Random Forest, which achieved 36% recall at threshold 0.5 and 70% recall with 40% precision at threshold 0.39, XGBoost provided a more reliable balance—particularly avoiding over-flagging false attrition cases while retaining strong detection capability.

- Key insights revealed that Laboratory Technicians, Healthcare Representatives, and Sales Executives—often with 3–4 years of tenure and no promotions or salary increases—are at higher risk. This indicates that the Sales department may require focused development around promotions and compensation.

Additionally, employees who have spent extended periods under the same manager or in the same role showed higher attrition probability, pointing to career stagnation and lack of growth opportunities as significant drivers.


- Unity Catalog ensures secure, governed, and scalable data access.
- SHAP explainability provides transparency into model decisions.
- MLflow ensures experiment reproducibility and tracking for deployment.
  
---


### Future Scope:

- Automating retraining via Databricks Jobs and retraining with 50,000 row making inference on Production level data. 

- AI Agent for Attrition Predictions, giving alerts on employee sentiments and recommending strategies. 


### Tech Stack:

- **Languages**: Python, SQL  
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `mlflow`, `pyspark`, `shap`  
- **Platform**: Databricks  
- **Version Control & Tracking**: MLflow, Unity Catalog
  
---

### Business Value:

This model enables HR departments to:

- Proactively intervene with at-risk employees (based on high attrition scores)
- Optimize pay structure in high-risk roles like Sales or HR
- Justify budget allocation for promotions and training at key senior levels
- Reduce churn cost by retaining skilled senior professionals


### 📁 Clone the repository
git clone https://github.com/<ShivaniKanodia>/employee-attrition-mlpipeline.git
cd employee-attrition-mlpipeline


### 📦 Install Python dependencies
pip install -r requirements.txt





