## Workforce Attrition Prediction with Explainable and Reproducible ML

Predicting employee churn and identifying key drivers of attrition using scalable machine learning practices. Built with  classification and booster algorithms, SHAP for explainability and trustworthy insights, Unity Catalog to store models and feature, and MLflow for experiment tracking, model reproducibility and transparency.

### Problem Statement:

**Employee attrition poses a significant risk to organizational stability and workforce planning. This project focuses on predicting which employees are most likely to leave and uncovering the core reasons behind employees leaving organisation.**

**By leveraging machine learning models, MLflow, SHAP and Unity Catalog, the goal is to empower HR teams with proactive insights—helping them improve retention, reduce the high cost of unplanned exits, and recommend policies or strategies based on results to improve employee retention.**

**This is a complete, production-ready ML pipeline built on Databricks, covering:**

- Data Management with Unity Catalog (secure, versioned Delta tables)

- Exploratory Analysis to uncover attrition trends by Job Role, Income and Daily Rate, and Promotion History

- Feature Selection using 

- Model Development with Logistic Regression, Random Forest XGBoost. 

- Explainability using SHAP for transparent and trustworthy decisions. 

- Model Tracking with MLflow (parameters, metrics, artifacts, comparison)

 - Each stage of the pipeline was built for reproducibility, scalability, and interpretability.

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

#### Visualizing Data Distribution by Department, Job Role and Job level:

<img width="578" height="455" alt="Screenshot 2025-10-07 at 16 19 49" src="https://github.com/user-attachments/assets/cb57e4bb-2196-430d-a544-9110b76b954f" />

<img width="569" height="401" alt="Screenshot 2025-10-07 at 16 19 57" src="https://github.com/user-attachments/assets/1baadcaf-4d71-4041-a9f1-b832d7659f22" />

<img width="583" height="344" alt="Screenshot 2025-10-07 at 16 20 13" src="https://github.com/user-attachments/assets/bd07642a-17d8-4b89-aa9a-a2eb284a6d36" />

### Checking Skewness and Outliers:

<img width="512" height="625" alt="Screenshot 2025-10-07 at 16 20 34" src="https://github.com/user-attachments/assets/1b17bdc2-a6a7-4ce0-b992-dda22b0a0bcf" />

<img width="527" height="628" alt="Screenshot 2025-10-07 at 16 20 49" src="https://github.com/user-attachments/assets/1ad8d5cb-d2d1-4a50-b48d-bec83261439e" />

<img width="487" height="320" alt="Screenshot 2025-10-07 at 16 21 00" src="https://github.com/user-attachments/assets/48b74cab-051f-43b2-b3f6-8d4cda8e127d" />


### Feature Selection Techniques:

**Chi- Square Testing:** 

<img width="467" height="332" alt="Screenshot 2025-10-07 at 16 21 23" src="https://github.com/user-attachments/assets/037310da-d9a0-41fa-8fc4-e888bc511620" />

<img width="456" height="311" alt="Screenshot 2025-10-07 at 16 21 37" src="https://github.com/user-attachments/assets/6e12b45b-4d18-4116-93dc-07e8f57f9bad" />

**T-Test:**

<img width="445" height="315" alt="Screenshot 2025-10-07 at 16 21 44" src="https://github.com/user-attachments/assets/6deb6d5d-e261-4cf7-8be8-3d75ac1c8883" />

The t-test and chi-square analyses were conducted to examine relationships between various features and employee attrition.

Features such as Years Since Last Promotion, Gender, Monthly Rate, Number of Companies Worked, Percent Salary Hike, Performance Rating, Employee Number, Hourly Rate, Education, and Relationship Satisfaction showed no statistically significant relationship with attrition (all p-values > 0.05).

This indicates that these variables, when considered individually, do not have a strong influence on whether an employee leaves or stays. However, they may still contribute in combination with other factors when included in a predictive model.


To ensure only statistically significant features contribute to the model, univariate feature selection was applied, both the tables are stored in **Unity Catalog**.

**ml_catalog.ml_schema.anova_results**
**ml_catalog.ml_schema.chisquare_results**

---

## 🚀 Model Experimentation:


Setting the Experimentation inside Databricks Notebook. 

<img width="1039" alt="Experimentation" src="https://github.com/user-attachments/assets/a927d8d0-5ea7-4e53-8365-fdb842b5bd62" />


---


## 📈 Model Training & Evaluation using MLflow


This centralized tracking ensured experiment reproducibility, hyperparameter versioning, and performance benchmarking

Logged key hyperparameters, evaluation metrics, trained model and visual artifacts like confusion matrix for every run — making it easy to reproduce or explain later.

<img width="1264" height="440" alt="Screenshot 2025-10-07 at 11 25 24" src="https://github.com/user-attachments/assets/f693697b-e089-41cc-ac34-b011333a2f3e" />


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

<img width="928" height="362" alt="Screenshot 2025-10-07 at 07 31 20" src="https://github.com/user-attachments/assets/5f1499cb-25e5-4f8c-bb85-1005b888d5ca" />

<img width="908" height="363" alt="Screenshot 2025-10-07 at 07 30 50" src="https://github.com/user-attachments/assets/deb087ec-132e-446a-a847-430f7634a4f3" />

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





