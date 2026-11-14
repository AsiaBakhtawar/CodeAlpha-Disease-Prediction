# Breast Cancer Prediction using Machine Learning

## Objective
Demonstrate supervised classification on the public Breast Cancer Wisconsin dataset.

## Workflow
1. Load the dataset included with scikit-learn.
2. Split data into training and test sets using stratification.
3. Scale features for Logistic Regression using a pipeline.
4. Train Logistic Regression and Random Forest models.
5. Compare accuracy, precision, recall, F1-score and ROC-AUC.
6. Save the best model and evaluation plots.

## Run
```bash
pip install -r requirements.txt
python train.py
```

## Technologies
Python, Pandas, NumPy, scikit-learn, Matplotlib, Joblib.

## Dataset label meaning
In scikit-learn's dataset, target 0 means malignant and target 1 means benign.

## Important limitation
This is a learning project using a public benchmark dataset. It is not clinically validated, cannot diagnose anyone, and must not be used to make medical decisions.
