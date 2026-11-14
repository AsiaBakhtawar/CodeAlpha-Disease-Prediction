
"""
CodeAlpha Internship - Task 2
Disease Prediction using Machine Learning

Dataset: Breast Cancer Wisconsin (Diagnostic)
Source: Scikit-learn built-in dataset

Educational project only.
This model is NOT a medical diagnostic tool.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_dataset():

    print("\n" + "=" * 60)
    print("DISEASE PREDICTION - MACHINE LEARNING PROJECT")
    print("=" * 60)

    print("\nLoading Breast Cancer Wisconsin dataset...")

    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data
    y = dataset.target

    print("\nDataset loaded successfully!")
    print("Total records:", X.shape[0])
    print("Total features:", X.shape[1])

    print("\nTarget classes:")
    print("0 = Malignant")
    print("1 = Benign")

    print("\nDataset information:")
    print(X.info())

    return X, y


# --------------------------------------------------
# 3. DATA CLEANING AND ANALYSIS
# --------------------------------------------------

def analyze_data(X, y):

    print("\n" + "=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)

    print("\nFirst five records:")
    print(X.head())

    print("\nStatistical summary:")
    print(X.describe())

    print("\nMissing values:")
    print(X.isnull().sum().sum())

    print("\nDuplicate records:")
    print(X.duplicated().sum())

    print("\nClass distribution:")
    print(
        y.value_counts().rename(
            index={0: "Malignant", 1: "Benign"}
        )
    )

    # Class distribution graph
    plt.figure(figsize=(8, 5))

    sns.countplot(
        x=y.map({0: "Malignant", 1: "Benign"})
    )

    plt.title("Disease Class Distribution")
    plt.xlabel("Disease Category")
    plt.ylabel("Number of Records")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "class_distribution.png")
    )

    plt.close()

    # Correlation heatmap
    top_features = (
        X.var()
        .sort_values(ascending=False)
        .head(12)
        .index
    )

    plt.figure(figsize=(12, 9))

    correlation = X[top_features].corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    )

    plt.close()

    # Save dataset information
    missing_report = pd.DataFrame({
        "Column": X.columns,
        "Missing Values": X.isnull().sum().values
    })

    missing_report.to_csv(
        os.path.join(OUTPUT_DIR, "missing_values_report.csv"),
        index=False
    )


# --------------------------------------------------
# 4. SPLIT DATASET
# --------------------------------------------------

def prepare_data(X, y):

    print("\nSplitting dataset into training and testing sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 5. CREATE MACHINE LEARNING MODELS
# --------------------------------------------------

def create_models():

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),

        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            ))
        ]),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }

    return models


# --------------------------------------------------
# 6. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

def train_models(models, X_train, X_test, y_train, y_test):

    results = []
    trained_models = {}

    best_model = None
    best_model_name = None
    best_auc = -1

    for name, model in models.items():

        print("\n" + "-" * 60)
        print("Training:", name)
        print("-" * 60)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        auc = roc_auc_score(y_test, probabilities)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": auc
        })

        trained_models[name] = model

        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1 Score :", round(f1, 4))
        print("ROC AUC  :", round(auc, 4))

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                predictions,
                target_names=["Malignant", "Benign"],
                zero_division=0
            )
        )

        if auc > best_auc:

            best_auc = auc
            best_model = model
            best_model_name = name

    return results, trained_models, best_model, best_model_name


# --------------------------------------------------
# 7. MODEL COMPARISON
# --------------------------------------------------

def compare_models(results):

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ROC AUC",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    print(results_df.to_string(index=False))

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "model_comparison.csv"),
        index=False
    )

    # Comparison graph
    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=results_df,
        x="Model",
        y="Accuracy"
    )

    plt.title("Machine Learning Model Accuracy Comparison")
    plt.xticks(rotation=15)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "model_comparison.png")
    )

    plt.close()

    return results_df


# --------------------------------------------------
# 8. EVALUATE BEST MODEL
# --------------------------------------------------

def evaluate_best_model(
    best_model,
    best_model_name,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("BEST MODEL EVALUATION")
    print("=" * 60)

    predictions = best_model.predict(X_test)

    probabilities = best_model.predict_proba(X_test)[:, 1]

    print("\nSelected model:", best_model_name)

    print(
        "\nBest model accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"]
    )

    plt.title("Confusion Matrix - Best Model")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )

    plt.close()

    # ROC curve
    RocCurveDisplay.from_predictions(
        y_test,
        probabilities
    )

    plt.title("ROC Curve - Best Model")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "roc_curve.png")
    )

    plt.close()

    # Save classification report
    report = classification_report(
        y_test,
        predictions,
        target_names=["Malignant", "Benign"],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "classification_report.csv")
    )

    # Save individual predictions
    prediction_results = pd.DataFrame({
        "Actual": y_test.map({
            0: "Malignant",
            1: "Benign"
        }),
        "Predicted": pd.Series(
            predictions,
            index=y_test.index
        ).map({
            0: "Malignant",
            1: "Benign"
        }),
        "Benign Probability": probabilities
    })

    prediction_results.to_csv(
        os.path.join(OUTPUT_DIR, "test_predictions.csv"),
        index=False
    )

    return predictions, probabilities


# --------------------------------------------------
# 9. FEATURE IMPORTANCE
# --------------------------------------------------

def save_feature_importance(best_model, X_train, y_train, X):

    print("\nGenerating feature importance...")

    classifier = (
        best_model.named_steps["classifier"]
        if isinstance(best_model, Pipeline)
        else best_model
    )

    if hasattr(classifier, "feature_importances_"):

        importance = classifier.feature_importances_

    elif hasattr(classifier, "coef_"):

        importance = np.abs(classifier.coef_[0])

    else:

        from sklearn.inspection import permutation_importance

        permutation = permutation_importance(
            best_model,
            X_train,
            y_train,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
            n_jobs=-1
        )

        importance = permutation.importances_mean

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"),
        index=False
    )

    plt.figure(figsize=(11, 8))

    sns.barplot(
        data=importance_df.head(15),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 15 Important Features")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_importance.png")
    )

    plt.close()


# --------------------------------------------------
# 10. SAVE TRAINED MODEL
# --------------------------------------------------

def save_model(best_model, best_model_name):

    model_path = os.path.join(
        OUTPUT_DIR,
        "best_disease_model.joblib"
    )

    joblib.dump(best_model, model_path)

    print("\nBest model saved:", model_path)

    with open(
        os.path.join(OUTPUT_DIR, "best_model_info.txt"),
        "w"
    ) as file:

        file.write("Disease Prediction Project\n")
        file.write(f"Best Model: {best_model_name}\n")
        file.write("Dataset: Breast Cancer Wisconsin\n")
        file.write("Purpose: Educational classification demo\n")
        file.write("Not intended for medical diagnosis.\n")


# --------------------------------------------------
# 11. MAIN FUNCTION
# --------------------------------------------------

def main():

    X, y = load_dataset()

    analyze_data(X, y)

    X_train, X_test, y_train, y_test = prepare_data(X, y)

    models = create_models()

    results, trained_models, best_model, best_name = train_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    compare_models(results)

    evaluate_best_model(
        best_model,
        best_name,
        X_test,
        y_test
    )

    save_feature_importance(
        best_model,
        X_train,
        y_train,
        X
    )

    save_model(best_model, best_name)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("Best Model:", best_name)
    print("Results folder:", OUTPUT_DIR)

    print("\nAll graphs, reports and model files are saved.")


if __name__ == "__main__":
    main()
