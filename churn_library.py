"""
Utility functions for the churn prediction pipeline.

This module provides data ingestion, exploratory data analysis,
feature engineering, model training, and result export logic for
the Udacity customer churn prediction project.

Author: Alex Holtzapple
Date created: 2026-06-05
"""
# Standard library imports
import os
import joblib

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

os.environ["QT_QPA_PLATFORM"] = "offscreen"

sns.set()

EDA_DIR = "./images/eda"
RESULTS_DIR = "./images/results"
MODELS_DIR = "./models"
DATA_PATH = "./data/bank_data.csv"


def create_output_directories():
    """
    Create output directories used by the project.

    input:
            None
    output:
            None
    """
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def import_data(pth):
    """
    Return a dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    """
    Perform EDA on df and save figures.

    input:
            df: pandas dataframe
    output:
            None
    """
    create_output_directories()

    df.head()
    df.isnull().sum()
    df.describe(include="all")

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(f'{EDA_DIR}/churn_distribution.png')

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(f'{EDA_DIR}/customer_age_distribution.png')

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(f'{EDA_DIR}/marital_status_distribution.png')

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(f'{EDA_DIR}/total_trans_count_distribution.png')

    plt.figure(figsize=(20, 10))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(f'{EDA_DIR}/correlation_heatmap.png')


def encoder_helper(df, category_lst):
    """
    Encode categorical features.

    input:
            df: pandas dataframe
            category_lst: list of categorical columns
    output:
            df: updated dataframe
    """
    dummies = df[category_lst]
    encoded_df = pd.get_dummies(dummies, drop_first=True)
    df = pd.concat([df, encoded_df], axis=1)
    return df


def perform_feature_engineering(df, response):
    """
    Split dataset into train and test sets.

    input:
              df: pandas dataframe
              response: response column name
    output:
              x_train, x_test, y_train, y_test
    """

    y = df[response]
    x = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
       'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
       'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
       'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
       'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
       'Avg_Utilization_Ratio', 'Gender_M',
       'Education_Level_Doctorate', 'Education_Level_Graduate',
       'Education_Level_High School', 'Education_Level_Post-Graduate',
       'Education_Level_Uneducated', 'Education_Level_Unknown',
       'Marital_Status_Married', 'Marital_Status_Single',
       'Marital_Status_Unknown', 'Income_Category_$40K - $60K',
       'Income_Category_$60K - $80K', 'Income_Category_$80K - $120K',
       'Income_Category_Less than $40K', 'Income_Category_Unknown',
       'Card_Category_Gold', 'Card_Category_Platinum', 'Card_Category_Silver']

    x[keep_cols] = df[keep_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=25)

    return x_train, x_test, y_train, y_test


def classification_report_image(model_outputs):
    """
    Save classification reports as images.

    input:
            predictions and labels
    output:
            None
    """
    create_output_directories()

    y_train = model_outputs["y_train"]
    y_test = model_outputs["y_test"]
    y_train_preds_lr = model_outputs["y_train_preds_lr"]
    y_train_preds_rf = model_outputs["y_train_preds_rf"]
    y_test_preds_lr = model_outputs["y_test_preds_lr"]
    y_test_preds_rf = model_outputs["y_test_preds_rf"]
    x_test = model_outputs["x_test"]
    lr_model = model_outputs["lr_model"]
    rfc_model = model_outputs["rfc_model"]

    lrc_plot = RocCurveDisplay.from_estimator(
        lr_model, x_test.values, y_test.values)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    RocCurveDisplay.from_estimator(
        rfc_model,
        x_test,
        y_test,
        ax=ax)

    lrc_plot.plot(ax=ax)
    plt.savefig(f'{RESULTS_DIR}/roc_curve.png')

    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(f'{RESULTS_DIR}/rf_classification_report.png')

    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(f'{RESULTS_DIR}/logistic_classification_report.png')


def feature_importance_plot(model, x_data):
    """
    Save feature importance plot.

    input:
            model, x_data, output path
    output:
            None
    """
    if hasattr(
            model,
            "best_estimator_") and hasattr(
            model.best_estimator_,
            "feature_importances_"):
        importances = model.best_estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]

        names = [x_data.columns[i] for i in indices]

        plt.figure(figsize=(20, 5))

        plt.title("Feature Importance")
        plt.ylabel('Importance')

        plt.bar(range(x_data.shape[1]), importances[indices])

        plt.xticks(range(x_data.shape[1]), names, rotation=45, ha='right')
        plt.savefig(f'{RESULTS_DIR}/feature_importances_rf.png')

    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = "Logistic Regression Feature Coefficients"

        feature_names = x_data.columns.tolist()

        sorted_importances = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, max(6, len(feature_names) * 0.2)))
        sns.barplot(
            x=np.array(importances)[sorted_importances], y=[
                feature_names[i] for i in sorted_importances], palette="viridis", hue=[
                feature_names[i] for i in sorted_importances], legend=False)
        plt.title(title)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.savefig(f'{RESULTS_DIR}/feature_importances_lr.png')


def train_models(x_train, x_test, y_train, y_test):
    """
    Train models and save outputs.

    input:
            train/test data
    output:
            None
    """
    create_output_directories()

    rfc = RandomForestClassifier(random_state=25)

    param_dist = {
        'n_estimators': [200, 300],
        'max_features': ['sqrt'],
        'max_depth': [5, 8, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'criterion': ['gini']
    }

    cv_rfc = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=param_dist,
        n_iter=12,
        cv=3,
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )

    cv_rfc.fit(x_train, y_train)

    lrc = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=3000))])

    lrc.fit(x_train, y_train)
    lrc = lrc['model']

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train.values)
    y_test_preds_lr = lrc.predict(x_test.values)

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    model_outputs = {
        'y_train': y_train,
        'y_test': y_test,
        'y_train_preds_lr': y_train_preds_lr,
        'y_train_preds_rf': y_train_preds_rf,
        'y_test_preds_lr': y_test_preds_lr,
        'y_test_preds_rf': y_test_preds_rf,
        'x_test': x_test,
        'lr_model': lr_model,
        'rfc_model': rfc_model}

    classification_report_image(model_outputs)

    feature_importance_plot(cv_rfc, x_train)
    feature_importance_plot(lrc, x_train)


if __name__ == "__main__":
    create_output_directories()

    bank_data = import_data(DATA_PATH)

    perform_eda(bank_data)

    category_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"]

    bank_data = encoder_helper(bank_data, category_columns)

    x_train_data, x_test_data, y_train_data, y_test_data\
        = perform_feature_engineering(bank_data, response="Churn")

    train_models(x_train_data, x_test_data, y_train_data, y_test_data)
