## Workforce Attrition Prediction with Explainable and Reproducible ML

Predicting employee churn and identifying key drivers of attrition using scalable machine learning practices. Built with  classification and booster algorithms, SHAP for explainability and trustworthy insights, Unity Catalog to store models and feature, and MLflow for experiment tracking, model reproducibility and transparency.

### Problem Statement:

Employee attrition poses a significant risk to organizational stability and workforce planning. This project focuses on predicting which employees are most likely to leave and uncovering the core reasons behind employees leaving organisation.**

By leveraging machine learning models, MLflow, SHAP and Unity Catalog, the goal is to empower HR teams with proactive insights—helping them improve retention, reduce the high cost of unplanned exits, and recommend policies or strategies based on results to improve employee retention.**

**This is a complete, production-ready ML pipeline built on Databricks, covering:**

- Data Management with Unity Catalog (secure and versioned Delta tables)

- Exploratory Data Analysis to uncover attrition trends by Job Role, Job Level, Income, Career Trajectory and Satisfaction level. 

- Feature Selection using Chi Square and T - testing, Feature Importance from Random Forest. 
  
- Model Development with Logistic Regression, Random Forest XGBoost and Model Tracking with MLflow (parameters, metrics, artifacts, comparison)

- Each stage of the pipeline was built for reproducibility, scalability, and used SHAP for interpretability.

---

### Dataset Overview:

- **Total records**: 1,470 employees
- **Attributes**: Demographics, Job Role, Satisfaction levels, Performance metrics etc.

---

### Unity Catalog for Data Storage and Access Control:

Created a Catalog (`ml_catalog`) and associated schema (`ml_schema`) under a managed volume to store both the complete dataset and the training subset in Delta format.

This structure enables governed, scalable, and versioned access to data for the ML workflow.


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

**T-Test:**


<img width="456" height="311" alt="Screenshot 2025-10-07 at 16 21 37" src="https://github.com/user-attachments/assets/6e12b45b-4d18-4116-93dc-07e8f57f9bad" />


<img width="445" height="315" alt="Screenshot 2025-10-07 at 16 21 44" src="https://github.com/user-attachments/assets/6deb6d5d-e261-4cf7-8be8-3d75ac1c8883" />


The t-test and chi-square analyses were conducted to examine relationships between various features and employee attrition.

Features such as Years Since Last Promotion, Gender, Monthly Rate, Number of Companies Worked, Percent Salary Hike, Performance Rating, Employee Number, Hourly Rate, Education, and Relationship Satisfaction showed no statistically significant relationship with attrition (all p-values > 0.05).

This indicates that these variables, when considered individually, do not have a strong influence on whether an employee leaves or stays. However, they may still contribute in combination with other factors when included in a predictive model.


To ensure only statistically significant features contribute to the model, univariate feature selection was applied, both the tables are stored in **Unity Catalog**.

---

## 🚀 Model Experimentation:


Setting the Experimentation inside Databricks Notebook. 

<img width="1039" alt="Experimentation" src="https://github.com/user-attachments/assets/a927d8d0-5ea7-4e53-8365-fdb842b5bd62" />


---


## 📈 Model Training & Evaluation using MLflow


This centralized tracking ensured experiment reproducibility, hyperparameter versioning, and performance benchmarking.

Logged key hyperparameters, evaluation metrics, trained model and visual artifacts like confusion matrix for every run — making it easy to reproduce or explain later.


---


### MLflow Metrics and Summary: 

As we can see screenshot below from Databricks MLFlow UI with Run Name, Duration of each Run and metrics logged. 

Used the MLflow UI in Databricks to compare multiple runs of Logistic Regression, Random Forest, and XGBoost. Selected the best model based on precision-recall trade-off and registered it using Model Registry in databricks to serve it later for deployment and making production level predictions.

We used precision-recall curve evaluation and selected an optimal threshold to minimize false negatives while avoiding unnecessary false alarms of employee leaving.

<img width="1264" height="440" alt="Screenshot 2025-10-07 at 11 25 24" src="https://github.com/user-attachments/assets/1ef7910e-319a-4d11-9ffd-5691308bcfb8" />


### MLFlow Tracking Dashboard:


I tracked and compared multiple models — Logistic Regression, Random Forest, and XGBoost — using MLflow. 

Each dashboard recorded: adjusted_f1, adjusted_precision, adjusted_recall, precision, recall and f1 score. 


<img width="908" height="363" alt="Screenshot 2025-10-07 at 07 30 50" src="https://github.com/user-attachments/assets/fcd14556-35a6-40a7-a289-db8e17fb9729" />

<img width="928" height="362" alt="Screenshot 2025-10-07 at 07 31 20" src="https://github.com/user-attachments/assets/f5446022-0a39-4c09-98ee-10de40d85eda" />

--

### Model Serving and Registry: 

I've provided signatures examples to be used by registered and served model for making predictions at endpoint. Below is snapshot for same. 

<img width="1784" height="1412" alt="image" src="https://github.com/user-attachments/assets/0fcedbef-b014-419f-9714-b29c7d71a4d1" />

--


### Confusion Matrix:

The confusion matrix provides a clear view of the model’s performance by showing how many predictions were correct (True Positives and True Negatives) versus incorrect (False Positives and False Negatives).

In our case, the model:

This model significantly outperformed others like Random Forest and Logistic Regression, both of which produced a much higher number of False Negatives—a critical metric in our use case, as missing true attrition cases could lead to substantial business impact. Reducing False Negatives was a top priority, and this model demonstrated a clear advantage in that regard.

- Lower false negatives improve recall — crucial for catching real churn cases.

---

### Model Registry:

<img width="878" alt="Screenshot 2025-07-01 at 15 35 05" src="https://github.com/user-attachments/assets/8c874b9d-2f28-4a8b-97b8-a2c591cf2c37" />


## Conclusion:

- After evaluating multiple models—including Logistic Regression, Xgboost and Random Forest (with maximum depth of 6 and n_estimators of 200) emerged as the best-performing model for our attrition prediction task. With a tuned threshold of 0.33, it struck an effective balance between interpretability, performance, and generalization. 

- The model achieved a recall of 82% and precision of 44% on detecting attrition, which is crucial for early risk detection while minimizing false positives that may harm employee trust.


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





