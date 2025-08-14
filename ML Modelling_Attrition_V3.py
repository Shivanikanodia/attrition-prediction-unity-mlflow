# Databricks notebook source
# MAGIC %md
# MAGIC ### Problem Statement: 
# MAGIC
# MAGIC *Understanding your employees, their sentiments and when are they likely to leave organisation and why is crucial for Business. As employee churn and their dissatisfaction with the organisation can lead to hamper the productivity and also cost and effort it takes to hire new employee. Thus, building an inclusive work place by understanding key drivers of employee sentiment and attrition is significant.* 
# MAGIC
# MAGIC 1. What is the likelihood of an active employee leaving the company?
# MAGIC 2. What are the key indicators of an employee leaving the company?
# MAGIC 3. What policies or strategies can be adopted based on the results to improve employee retention?
# MAGIC    
# MAGIC Given that we have data on former employees, this is a standard supervised classification problem where the label is a binary variable, 0 (likely to stay), 1 (likely to leave). In this study, our target variable Y is the probability of an employee leaving the company.
# MAGIC
# MAGIC ### Project Structure:
# MAGIC
# MAGIC - **Data Exploration and Saving to Unity Catalog**
# MAGIC
# MAGIC - **Exploratory Data Analysis** (Distribution, Summary, Skewness and Multicollinearity)
# MAGIC
# MAGIC - **Feature Selection using A/B Testing:** ANOVA and Chi-Square
# MAGIC
# MAGIC - **Data Preprocessing:** Scaling and Encoding
# MAGIC
# MAGIC - **Model Training and logging:**
# MAGIC
# MAGIC - *Logging parameters, metrics, metadata, input_examples, model.pkl, artifacts and plots using Databricks Mlflow for reproducibility and versioning*
# MAGIC
# MAGIC - **Model Evaluation using Classification Report & Confusion Matrix**
# MAGIC
# MAGIC - **Model Explainability using SHAP Values to build trust and provide transparent insights**
# MAGIC
# MAGIC - **Deploying Best Model to Production using model serving**
# MAGIC
# MAGIC ### - **Conclusion and Further Recommendations** 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Exploration:
# MAGIC
# MAGIC In this case study, a HR dataset was sourced for Employee Attrition & Sentiment Analysis which contains employee data for 10,000 employees with information on monetary benefits, careeer progression, worklife balance and etc. I will use this dataset to predict when employees are going to quit by understanding the main drivers of employee churn.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataset Source:
# MAGIC
# MAGIC The Employee Attrition & Performance dataset is used for this project. It is saved in delta tables using databricks unity catalog for tracking and security.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Importing Libraries/Modules
# MAGIC We will import all the required libraries and modules required for our project as follows:  

# COMMAND ----------

# Data handling
import pandas as pd
import numpy as np

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import plotly.express as px

# Warnings
import warnings
warnings.filterwarnings('ignore')  # Optional: to suppress warnings


# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Importing:
# MAGIC
# MAGIC Let us import the dataset using spark to know what the data contains.
# MAGIC
# MAGIC  Spark is an open-source, distributed big data processing framework designed for fast computation and scalable analytics.

# COMMAND ----------

df1 = spark.read.csv("/Volumes/ml_catalog/ml_schema/version_2_traindata/train.csv", header=True, inferSchema=True)

df2 = spark.read.csv("/Volumes/ml_catalog/ml_schema/version_2_testdata/test.csv", header=True, inferSchema=True)

display(df1)
display(df2) 

# COMMAND ----------

# number of rows and number of columns in employee dataset 
num_rows = df1.count()
num_columns = len(df1.columns)

# Print results
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")


# COMMAND ----------

# Convert to pandas
pandas_df1 = df1.toPandas()
pandas_df2 = df2.toPandas()

# COMMAND ----------

df1.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC The dataset has object (string) and numeric (int) datatypes, showing mix of sets.  

# COMMAND ----------


#Looking at Null values in dataset to check if we need to treat them using mean/median or any other techniques. 

pandas_df1.isnull().sum()




# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary Statistics of Integer Numeric coulmns to see median, data range, outliers and distribution.  

# COMMAND ----------


summary_df1 = df1.select([c for c, t in df1.dtypes if t in ('int', 'double', 'float')]).describe()

summary_df1.display()


# COMMAND ----------

# MAGIC %md
# MAGIC **Key Observations & Insights:**
# MAGIC
# MAGIC - Age Feature Indicates a mix of young and experienced professionals.
# MAGIC
# MAGIC -  Long average tenure "Years at Company", suggesting good employee retention. The distribution may be right-skewed due to some very long tenures (51 years).
# MAGIC
# MAGIC - High income variance "Monthly Income";—possibly management or executive roles. A histogram could reveal salary bands or inequality. 
# MAGIC
# MAGIC - Most employees have had 0–1 promotions, while others have 3-4 promotions which may raise questions about career progression paths in organisation stickely tied to employee staying or leaving company.
# MAGIC
# MAGIC - Commute distances are fairly large, with wide variability, Could be a concern for work-life balance or attrition.
# MAGIC
# MAGIC - Indicates moderate family responsibilities, potentially useful for analyzing benefits, insurance plans, or flexible work policies. 
# MAGIC
# MAGIC As we already have years at company coulmn, Company tenure seems to be redundant and miscalculated. lets drop that coulmn for clarity and to ignore errors. 
# MAGIC
# MAGIC These are some powerful insights to measure potential reasons behind employee attrition. lets visualize the feaures for better visibiltiy and explainability. 

# COMMAND ----------

from pyspark.sql.functions import col

# Find the actual column name (case-insensitive match)
columns_to_drop = [c for c in df1.columns if c.lower() == "Company Tenure".lower()]
df = df1.drop(*columns_to_drop)


# COMMAND ----------

# MAGIC %md
# MAGIC Lets see frequency of employees falling under different categories of our categorical variables to see how data distribution is, It will help us also in spotting dominant values and identifying if there are potential class imbalances. 

# COMMAND ----------

# Step 1: Get all string (object) columns
string_columns = pandas_df1.select_dtypes(include=['object', 'string']).columns

# Step 2: Loop through each string column and show top 10 frequent values
for column in string_columns:
    print(f"\nFrequent values in column: '{column}'")
    print(pandas_df1[column].value_counts(dropna=False).head(10))


# COMMAND ----------

import pandas as pd

# Step 1: Select all string (object or string) columns
string_columns = pandas_df1.select_dtypes(include=['object', 'string']).columns

# Step 2: Count and print unique (non-null) values in each string column
for col_name in string_columns:
    unique_count = pandas_df1[col_name].dropna().nunique()
    print(f"{col_name}: {unique_count} unique values")




# COMMAND ----------

# MAGIC %md
# MAGIC Unique value will help us understanding how we can apply transformation techniques based on number of unique values avaiable, their behaviour and type of categories. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### ANOVA-TEST AND CHI SQUARE TEST FOR NUMERICAL AND CATEGORICAL FEATURE SELECTION:

# COMMAND ----------

from scipy.stats import f_oneway


# Step 1: Get only numeric columns excluding Attrition
num_cols = pandas_df1.select_dtypes(include=['number']).columns.tolist()
if 'Attrition' in num_cols:
    num_cols.remove('Attrition')

# Step 2: Filter out numeric columns with ≤10 unique values (likely categorical)
filtered_cols = [col for col in num_cols if pandas_df1[col].nunique() > 10]

# Step 3: Run ANOVA
anova = []
for c in filtered_cols:
    groups = [g[c].dropna() for _, g in pandas_df1.groupby('Attrition') if len(g) > 1]
    if len(groups) > 1:
        f_stat, p_val = f_oneway(*groups)
        anova.append((c, f_stat, p_val))

ANOVA_train_df = pd.DataFrame(anova, columns=['Column', 'F_Stat', 'pValue']).sort_values(by='pValue')
print(ANOVA_train_df)


# COMMAND ----------

# MAGIC %md
# MAGIC **SAVING STASTICALLY SIGNIFICANCE FEATURES FROM ANOVA_TEST AND CHI_SQUARE TO UNITY CATALOG TO ENSURE REPRODUCIBILITY:** 

# COMMAND ----------


ANOVA_train_df_results = spark.createDataFrame(ANOVA_train_df)

# Saving to Unity Catalog Delta Table correctly
ANOVA_train_df_results.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("ml_catalog.ml_schema.ANOVA_train_df_results")


# COMMAND ----------

# MAGIC %pip install scipy
# MAGIC
# MAGIC from scipy.stats import chi2_contingency
# MAGIC
# MAGIC cat_cols = pandas_df1.select_dtypes(include=['object', 'category']).columns.tolist()
# MAGIC chi_results = []
# MAGIC
# MAGIC for col in cat_cols:
# MAGIC     if col == 'Attrition': continue
# MAGIC     table = pd.crosstab(pandas_df1[col], pandas_df1['Attrition'])
# MAGIC     if table.shape[0] > 1 and table.shape[1] > 1:
# MAGIC         chi2, p, dof, _ = chi2_contingency(table)
# MAGIC         chi_results.append((col, chi2, p, dof))
# MAGIC
# MAGIC result_df = pd.DataFrame(chi_results, columns=['Column', 'Chi2_Stat', 'pValue', 'DegreesOfFreedom']) \
# MAGIC              .sort_values(by='pValue')
# MAGIC
# MAGIC print(result_df)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC From above analysis we can choose coulmns with high **Chi Square Statistic** and **P value < 0.05**
# MAGIC
# MAGIC All of our features have p values less then 0.05 except employee recognization. Lets drop it from our analysis for more robust results.   
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Visualization Phase:

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Define color palette
my_palette = sns.color_palette('rainbow')

# Set font size
sns.set(font_scale=3)

# Plot histogram
pandas_df1.hist(bins=20, figsize=(40, 40), color=my_palette[5], alpha=0.6)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The distribution is bimodal — there are two prominent peaks.
# MAGIC
# MAGIC - For Monthly Income, most values fall between **$4,000 and $10,000,** with a few extending towards **$14,000**. The curve is smoothly bell-shaped, but the two humps indicate distinct clusters.
# MAGIC
# MAGIC - The distribution for years at company is highly right-skewed. The majority of employees have been with the company for **less than 10 years** and very few employees have stayed for more than 30–40 years. The sharp drop-off indicates either High attrition after a certain period. 
# MAGIC
# MAGIC - The organization seems to have a well-balanced age structure and farily distributed distance from home — not overly concentrated in one groups.
# MAGIC
# MAGIC

# COMMAND ----------


#df = df.toPandas()

fig, axes = plt.subplots(5, 1, figsize=(30, 30))

# Years at Company
sns.boxplot(x=pandas_df1['Years at Company'], ax=axes[0],  palette='Set3')
axes[0].set_title('Years at Company')

# Monthly Income
sns.boxplot(x=pandas_df1['Monthly Income'], ax=axes[1], palette='Set3')
axes[1].set_title('Monthly Income')

# Age
sns.boxplot(x=pandas_df1['Age'], ax=axes[2])
axes[2].set_title('Age')

# Distance from Home 
sns.boxplot(x=pandas_df1['Distance from Home'], ax=axes[3])
axes[3].set_title('Distance from Home')

# Distance from Home 
sns.boxplot(x=pandas_df1['Number of Promotions'], ax=axes[4])
axes[4].set_title('Number of Promotions')

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The **Years at Company** feature shows a right-skewed distribution, with the median falling between 12–14 years. The boxplot is shifted to the left, confirming the skew observed in the histogram.
# MAGIC
# MAGIC Similarly, **Monthly Income** also exhibits right skewness, with a median around $7,000–$8,000. The left-shifted box in the boxplot aligns with the histogram findings, reinforcing the presence of skewness.
# MAGIC
# MAGIC The **Number of Promotions** displays a pronounced right-skewed distribution, with a large majority of employees (~7,500) having zero promotions. This highlights that promotions are rare events in this dataset. Such rarity may indicate:
# MAGIC
# MAGIC A high turnover rate, with employees leaving before receiving a promotion, or
# MAGIC
# MAGIC Promotions being strongly associated with long tenure or specific roles.
# MAGIC
# MAGIC To address skewness and improve model performance, we will apply log transformations to these features during model training. 

# COMMAND ----------

# MAGIC %md
# MAGIC **Lets view attrition by different categorical coulmns we have and see how each one impacts attrition.**   
# MAGIC

# COMMAND ----------


#Setting up the visual style
sns.set(style="dark")

# 1. Job Level vs Attrition
job_attr = pandas_df1.groupby(['Job Level', 'Attrition']).size().unstack(fill_value=0)
job_attr.plot(kind='bar', stacked=True, figsize=(6, 5))
plt.title('Attrition and Job Level')
plt.ylabel('Employee Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#2. Attrition vs Job Role
job_attr = pandas_df1.groupby(['Job Role', 'Attrition']).size().unstack(fill_value=0)
job_attr.plot(kind='bar', stacked=True, figsize=(6, 5))
plt.title('Attrition by Job Role')
plt.ylabel('Employee Count')
#plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Employee Recognition vs Attrition
rec_attr = pandas_df1.groupby(['Employee Recognition', 'Attrition']).size().unstack(fill_value=0)
rec_attr.plot(kind='bar', stacked=True, figsize=(6, 5))
plt.title('Attrition by Recognition Level')
plt.ylabel('Employee Count')
plt.show()

# 4. Overtime vs Attrition
ot_attr = pandas_df1.groupby(['Overtime', 'Attrition']).size().unstack(fill_value=0)
ot_attr.plot(kind='bar', stacked=True, figsize=(6, 5))
plt.title('Attrition by Overtime')
plt.ylabel('Employee Count')
plt.show()

# 5. Work-Life Balance vs Attrition
wlb_attr = pandas_df1.groupby(['Work-Life Balance', 'Attrition']).size().unstack(fill_value=0)
wlb_attr.plot(kind='bar', stacked=True, figsize=(6, 5))
plt.title('Attrition by Work-Life Balance')
plt.ylabel('Employee Count')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Employees who rated their work-life balance as "Fair" or "Good" exhibit higher attrition counts. 
# MAGIC
# MAGIC Interestingly, even those reporting an "Excellent" work-life balance are not immune to attrition, though the absolute numbers are comparatively lower. Additionally, while "Poor" work-life balance accounts for fewer total employees, it shows a high attrition rate relative to its small base, indicating strong dissatisfaction within that group. 

# COMMAND ----------


# 2. Performance Rating vs Attrition
ot_attr = pandas_df1.groupby(['Performance Rating', 'Attrition']).size().unstack(fill_value=0)
ot_attr.plot(kind='bar', stacked=True, figsize=(6, 5))
plt.title('Attrition by Performance Rating')
plt.ylabel('Employee Count')
plt.show()

# 3. Leadership Opportunities vs Attrition
wlb_attr = pandas_df1.groupby(['Leadership Opportunities', 'Attrition']).size().unstack(fill_value=0)
wlb_attr.plot(kind='bar', stacked=True, figsize=(4, 4))
plt.title('Attrition by Leadership Opportunities')
plt.ylabel('Employee Count')
plt.show()

# 4. Innovation Opportunities vs Attrition
wlb_attr = pandas_df1.groupby(['Innovation Opportunities', 'Attrition']).size().unstack(fill_value=0)
wlb_attr.plot(kind='bar', stacked=True, figsize=(4, 4))
plt.title('Attrition by Innovation Opportunities')
plt.ylabel('Employee Count')
plt.show()

# 5. Remote Work vs Attrition
wlb_attr = pandas_df1.groupby(['Remote Work', 'Attrition']).size().unstack(fill_value=0)
wlb_attr.plot(kind='bar', stacked=True, figsize=(4, 4))
plt.title('Attrition by Remote Work')
plt.ylabel('Employee Count')
plt.show() 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC From the bar chart above, it is evident that employees without remote work options and below-average performance ratings are more likely to leave the organization. This trend suggests a lack of satisfaction, possibly due to limited flexibility in work arrangements and lower job performance perceptions.
# MAGIC

# COMMAND ----------

sns.catplot(data=pandas_df1, kind='count',
            x='Performance Rating', hue='Attrition',
            col='Job Satisfaction')
plt.subplots_adjust(top=0.8)
plt.suptitle('Attrition Count by Performance Rating and Job Satisfaction')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Based on the visualization above , Employees with low and average Performance Rating with High & low Job Satisfacion tend to leave more quickly than the one who stayed, this confirms our above insight we received from employee count bar charts.   

# COMMAND ----------

# MAGIC %md
# MAGIC Bar charts above shows that technology and finance job roles show higher median values for monthly Income than other Job roles like Healthcare and Media. 
# MAGIC
# MAGIC While the median values for Monthly Income and Years at Company for employee who left is lower then median values for the one who stayed, indicating that employees are satisfied with monetary benefits provided . 

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming `df` is your Pandas DataFrame
categorical_cols = ['Job Role', 'Gender', 'Marital Status', 'Number of Dependents']

# Set style
sns.set(style="whitegrid")
plt.figure(figsize=(16, 12))

# Loop through each categorical column
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 2, i)
    sns.countplot(data=pandas_df1, x=col, hue='Attrition')
    plt.title(f'Attrition by {col}')
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel("Count")

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The plots indicate that female employees, single individuals, and those with 2–3 dependents are more likely to leave compared to their counterparts who stay. This suggests potential underlying factors such as support systems, flexibility needs, or work-life challenges that may disproportionately affect these groups and warrant further attention. 

# COMMAND ----------


type(pandas_df1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### MODEL TRAINING: 

# COMMAND ----------

target = 'Attrition'

X_train = pandas_df1.drop(columns=[target])
y_train = pandas_df1[target]

X_test = pandas_df2.drop(columns=[target])
y_test = pandas_df2[target]


numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()


# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import mlflow

# Set the registry URI to databricks
mlflow.set_registry_uri("databricks")

experiment_name = "Attrition_Prediction_Version3"

# Try to create experiment only if it doesn’t exist
from mlflow.exceptions import MlflowException

try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
    print(f"Created new experiment: {experiment_name}")
except MlflowException as e:
    print(f"Experiment may already exist. Error: {e}")

mlflow.set_experiment("/Users/kanodiashivani27@gmail.com/Attrition_Prediction_Version3")


# COMMAND ----------

# MAGIC %pip install numpy==1.26.4
# MAGIC
# MAGIC %pip install shap
# MAGIC from mlflow.models.signature import infer_signature
# MAGIC from sklearn.model_selection import GridSearchCV
# MAGIC import shap
# MAGIC
# MAGIC # plot & log confusion matrix
# MAGIC def log_confusion_matrix(y_true, y_pred, name="conf_matrix.png"):
# MAGIC     cm = confusion_matrix(y_true, y_pred)
# MAGIC     plt.figure(figsize=(6, 4))
# MAGIC     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# MAGIC     plt.xlabel("Predicted")
# MAGIC     plt.ylabel("Actual")
# MAGIC     plt.title("Confusion Matrix")
# MAGIC     plt.tight_layout()
# MAGIC     plt.savefig(name)
# MAGIC     mlflow.log_artifact(name)
# MAGIC     plt.close()
# MAGIC
# MAGIC # 🎨 plot & log ROC-AUC
# MAGIC def log_roc_auc(y_true, y_scores, name="roc_auc.png"):
# MAGIC     RocCurveDisplay.from_predictions(y_true, y_scores)
# MAGIC     plt.title("ROC Curve")
# MAGIC     plt.tight_layout()
# MAGIC     plt.savefig(name)
# MAGIC     mlflow.log_artifact(name)
# MAGIC     plt.close()
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %pip install xgboost
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC from sklearn.compose import ColumnTransformer
# MAGIC from sklearn.impute import SimpleImputer
# MAGIC from sklearn.preprocessing import StandardScaler, OneHotEncoder
# MAGIC from sklearn.linear_model import LogisticRegression
# MAGIC from sklearn.ensemble import RandomForestClassifier
# MAGIC from xgboost import XGBClassifier
# MAGIC
# MAGIC
# MAGIC numeric_transformer = Pipeline([
# MAGIC     ('imputer', SimpleImputer(strategy='mean')),
# MAGIC     ('scaler', StandardScaler())
# MAGIC ])
# MAGIC
# MAGIC categorical_transformer = Pipeline([
# MAGIC     ('imputer', SimpleImputer(strategy='most_frequent')),
# MAGIC     ('encoder', OneHotEncoder(handle_unknown='ignore'))
# MAGIC ])
# MAGIC
# MAGIC preprocessor = ColumnTransformer([
# MAGIC     ('num', numeric_transformer, numerical_features),
# MAGIC     ('cat', categorical_transformer, categorical_features)
# MAGIC ])
# MAGIC
# MAGIC                 

# COMMAND ----------

# MAGIC %pip install imbalanced-learn
# MAGIC from imblearn.pipeline import Pipeline as ImbPipeline
# MAGIC from imblearn.over_sampling import SMOTE
# MAGIC
# MAGIC log_model = LogisticRegression(solver="liblinear", max_iter=1000)
# MAGIC
# MAGIC log_pipeline = ImbPipeline([
# MAGIC     ('preprocessor', preprocessor),
# MAGIC     ('smote', SMOTE(random_state=42)),
# MAGIC     ('clf', log_model)
# MAGIC ])
# MAGIC
# MAGIC log_param_grid = {
# MAGIC     "clf__C": [0.1, 1, 10],
# MAGIC     "clf__penalty": ["l1", "l2"]
# MAGIC }
# MAGIC
# MAGIC

# COMMAND ----------

print(log_pipeline.named_steps['preprocessor'])


# COMMAND ----------


rf_model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    bootstrap=True  # required for oob_score
)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', rf_model)
])

rf_param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [4, 6, 8],
    "clf__min_samples_split": [2, 5],
    "clf__max_features": [0.3, 0.8],
    "clf__oob_score": [True]
}

# COMMAND ----------

xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', xgb_model)
])

xgb_param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [3, 4, 5, 6],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__scale_pos_weight": [1, 3, 5, 10],
    "clf__gamma": [0, 1, 3, 5]
}


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Lets create a pipeline for performing data preprocessing which is crucial step before we implement our models. 
# MAGIC
# MAGIC **We will apply log transformation for features that exhibit skewness, as evident from the histograms above. Specifically, Years at Company and Number of Promotions shows wide value ranges and right-skewed distributions.** 
# MAGIC
# MAGIC **Applying log transformation to these features helps make the data more normally distributed, reduces skewness, and handles large value ranges, thereby improving model robustness and performance.** 
# MAGIC
# MAGIC Also, we will perform standard scaling on Monthly Income and Age ensuring that these large scale values are compressed to their normal scale before ingesting in the model for interpretabiltiy. 
# MAGIC
# MAGIC Moreover, we will do label encoding and one_hot encoding for our categorical variables. 
# MAGIC

# COMMAND ----------

def train_and_log(pipeline, param_grid, model_name):
    with mlflow.start_run(run_name=model_name):

        # Step 2: Perform Grid Search
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # Step 3: Predictions
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Step 6: Evaluate with Default Threshold (0.5)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, pos_label='Left'),
            "precision": precision_score(y_test, y_pred, pos_label='Left'),
            "recall": recall_score(y_test, y_pred, pos_label='Left'),
            "roc_auc": roc_auc_score((y_test == 'Left').astype(int), y_proba)
        }

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)

        # Step 6.1: Find Optimal Threshold (F1-Optimized)
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba,)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_idx = f1.argmax()
        best_threshold = thresholds[best_idx]

        y_pred_opt = (y_proba >= best_threshold).astype(int) 

        positive_class = 'Left'
        negative_class = 'Stayed'
        y_pred_opt = np.where((y_proba >= best_threshold), positive_class, negative_class)

        mlflow.log_param("optimal_threshold", best_threshold)
        mlflow.log_metrics({
            "adjusted_f1": f1[best_idx],
            "adjusted_precision": precision[best_idx],
            "adjusted_recall": recall[best_idx]
        })

        # Step 7: Log the Model with Signature
        input_example = X_test.iloc[[0]]
        signature = infer_signature(input_example, best_model.predict(input_example))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name=model_name,
            input_example=input_example,
            signature=signature
        )

        # Step 8: Confusion Matrix and ROC Curve
        log_confusion_matrix(y_test, y_pred)
        log_confusion_matrix(y_test, y_pred_opt)
        log_roc_auc(y_test, y_proba)

        print(f"\n{model_name} classification report (0.5 threshold):\n", classification_report(y_test, y_pred))
        print(f"\n{model_name} classification report (tuned threshold = {best_threshold:.2f}):\n",
              classification_report(y_test, y_pred_opt))

        # Step 9: SHAP Interpretation
        print("\n📊 SHAP: Starting SHAP interpretation using classifier trained on SMOTE-scaled data.")

        # 1. Re-train classifier on scaled training data
        clf_for_shap = type(pipeline.named_steps['clf'])(**grid.best_params_['clf']) if 'clf' in grid.best_params_ else model
        clf_for_shap.fit(X_resampled_scaled, y_resampled)

        # 2. Transform test data (scaled version for SHAP input)
        X_test_scaled = best_model.named_steps["scaler"].transform(X_test)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        # 3. Sample 100 test points from scaled data for SHAP input
        X_sample_scaled = shap.utils.sample(X_test_scaled_df, 100, random_state=42)

        # 4. KernelExplainer on scaled data
        explainer = shap.KernelExplainer(clf_for_shap.predict, X_sample_scaled)
        shap_values = explainer.shap_values(X_test_scaled_df)

        # 5. 🎯 For user-friendly display: use inverse_transform for readable feature values (NOT for SHAP input!)
        X_test_original_df = pd.DataFrame(X_test, columns=X_test.columns)

        # 6. SHAP summary plot with readable values
        plt.figure()
        shap.summary_plot(shap_values, features=X_test_original_df, feature_names=X_test.columns, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")
        print("✅ SHAP summary plot saved and logged to MLflow.")

        print("\n📊 SHAP: Completed SHAP interpretation with user-friendly feature display.")
 

 

# COMMAND ----------

import numpy as np 
print("Unique labels in y_test:", np.unique(y_test))


# COMMAND ----------

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
    confusion_matrix, RocCurveDisplay, precision_recall_curve
)
import sys
import os

train_and_log(log_pipeline, log_param_grid, "LogisticRegression")
train_and_log(rf_pipeline, rf_param_grid, "RandomForest")
train_and_log(xgb_pipeline, xgb_param_grid, "XGBoost")

