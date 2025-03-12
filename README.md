# HR Analytics: IBM Employee Attrition & Performance

## Problem Statement

Employee retention is critical for organizational success. This project aims to understand why and when employees are most likely to leave an organization. By identifying the key factors driving employee attrition, organizations can implement targeted strategies to improve retention and plan new hiring in advance.

### In this case study, we address the following objectives:

Identify key drivers of employee churn.
Predict when employees are likely to quit.
Provide actionable insights to improve employee retention.

## Dataset Overview

The dataset used in this project is sourced from IBM HR Analytics Employee Attrition & Performance. It includes data for 1,470 employees, with various attributes such as demographics, job role, satisfaction levels, and performance metrics.

## Source: 
IBM HR Analytics Employee Attrition & Performance (fictional dataset created by IBM data scientists).
## Purpose: 
The dataset was designed to demonstrate the IBM Watson Analytics tool for analyzing employee attrition.

## Key Steps Involved

1. Data Exploration
Gaining an initial understanding of the dataset.
Identifying key features and data types.
Checking for missing or inconsistent data.
2. Data Preprocessing
Handling missing values.
Encoding categorical variables.
Normalizing numerical features for modeling.
3. Exploratory Data Analysis (EDA)
Analyzing trends, patterns, and distributions using Python and SQL.
Identifying relationships between features and employee attrition.
4. Data Visualization
Visualizing trends and key findings using:
Python (Matplotlib, Seaborn).
Power BI/Tableau for interactive dashboards.
5. Feature Selection
Identifying the most important predictors of employee attrition using:
Correlation matrices.
Feature importance from machine learning models.
6. Model Selection and Training
Using machine learning algorithms to predict employee attrition:
Logistic Regression.
Random Forest.
7. Model Evaluation
Evaluating model performance using:
Accuracy.
Precision, Recall, and F1-Score.
ROC-AUC Curve.
8. Classification Report and Accuracy
Generating a detailed classification report.
Explaining the accuracy and implications of the model's predictions.
9. Conclusion and Sources
Summarizing insights and actionable recommendations for reducing employee attrition.
Listing references and tools used in the project.
Tech Stack

## Languages: 
Python, SQL
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn 
Version Control: Git/GitHub

## HR-Analytics-IBM-Attrition

A comprehensive analysis of IBM HR Analytics Employee Attrition & Performance dataset to identify key drivers of attrition, build predictive models, and provide actionable insights through interactive dashboards.

# 📁 Project Structure

HR-Analytics-IBM-Attrition/
├── data/                       # Dataset files
├── notebooks/                  # Jupyter notebooks for analysis
├── src/                        # Python scripts for preprocessing, modeling, and evaluation
├── visualizations/             # Power BI/Tableau dashboards
├── README.md                   # Project overview and instructions
├── requirements.txt            # Dependencies for the project
└── .gitignore                  # Ignored files and folders
⚙️ Installation

# Clone the repository:
git clone https://github.com/yourusername/HR-Analytics-IBM-Attrition.git
cd HR-Analytics-IBM-Attrition
Install the required dependencies:
pip install -r requirements.txt
🚀 How to Run the Project

# Run Jupyter notebooks for exploratory data analysis and modeling:
cd notebooks/
jupyter notebook
Or run the Python scripts located in the src/ directory:
cd src/
python your_script.py
