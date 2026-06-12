"""Testing and logging script for the churn prediction pipeline.

This module defines end-to-end tests for the churn_library workflow,
including data import, exploratory data analysis, feature encoding,
feature engineering, model training and evaluation. It also configures logging so
that INFO and ERROR events are written to a dedicated log file.

Author: Alex Holtzapple
Date created: 2026-06-10
"""

import logging
import os
import churn_library as cls

LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")
DATA_PATH = "./data/bank_data.csv"
EDA_DIR = "./images/eda"
MODEL_DIR = "./models"
IMG_DIR = "./images/results"

category_columns = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category"]


def test_logs_directory():
    """Test that the logs directory can be created."""
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        logging.basicConfig(
            filename=LOG_FILE,
            level=logging.INFO,
            filemode='w',
            format='%(name)s - %(levelname)s - %(message)s',
            force=True)
        logging.info(
            "SUCCESS: Logs directory & file exists and is ready for logging.")

    except Exception as err:
        logging.error(
            "ERROR: Failed to create logs directory or configure logging: %s",
            err)
        raise err


def test_import():
    """Test data import."""
    try:
        assert os.path.exists(
            DATA_PATH), f"ERROR: File {DATA_PATH} does not exist."
        df = cls.import_data(DATA_PATH)
        assert df.shape[0] > 0, "ERROR: DataFrame is empty. Check the file and contents."
        assert df.shape[1] > 0, "ERROR: DataFrame has no columns. Check the file and contents."
        logging.info("SUCCESS: Data import successful.")

    except AssertionError as err:
        logging.error(err)
        raise err

    except Exception as err:
        logging.error("ERROR: Unexpected failure during file import: %s", err)
        raise err


def test_eda():
    """Test EDA."""
    try:
        df = cls.import_data(DATA_PATH)
        cls.perform_eda(df)

        image_files = [
            'churn_distribution.png',
            'customer_age_distribution.png',
            'marital_status_distribution.png',
            'total_trans_count_distribution.png',
            'correlation_heatmap.png'
        ]

        for image_file in image_files:
            assert os.path.exists(
                f'{EDA_DIR}/{image_file}'), \
                f"ERROR: Image file {image_file} not found in {EDA_DIR}."
        logging.info(
            "SUCCESS: EDA performed and images generated successfully.")

    except AssertionError as err:
        logging.error(err)
        raise err

    except Exception as err:
        logging.error(
            "ERROR: Unexpected failure during EDA image verification: %s", err)
        raise err


def test_encoder_helper():
    """Test encoding."""
    try:
        df = cls.import_data(DATA_PATH)
        df = cls.encoder_helper(df, category_columns)
        for category in category_columns:
            encoded_columns = [
                col for col in df.columns if col.startswith(
                    category + "_")]
            unique_values = df[category].nunique()

            assert len(
                encoded_columns) > 0, \
                f"ERROR: No encoded columns found for category {category}."
            assert len(encoded_columns) == unique_values - 1, \
                f"ERROR: Encoded columns for {category} \
                do not match expected number based on unique values.\nExpected {
                unique_values - 1} encoded columns for {category}, but found {
                len(encoded_columns)}."

        logging.info(
            "SUCCESS: Category column encoding successful and verified.")

    except AssertionError as err:
        logging.error(err)
        raise err

    except Exception as err:
        logging.error(
            "ERROR: Unexpected failure during encoder helper test: %s", err)
        raise err


def test_perform_feature_engineering():
    """Test feature engineering."""
    try:
        df = cls.import_data(DATA_PATH)
        cls.perform_eda(df)
        df = cls.encoder_helper(df, category_columns)
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            df, response="Churn")

        keep_cols = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio',
            'Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn']

        assert all(
            col in x_train.columns for col in keep_cols), \
            "ERROR: Not all expected columns are present after feature engineering."
        assert x_train.shape[0] > 0 and x_train.shape[1] > 0, \
            "ERROR: x_train is empty after feature engineering."
        assert x_test.shape[0] > 0 and x_test.shape[1] > 0, \
            "ERROR: x_test is empty after feature engineering."
        assert y_train.shape[0] > 0, \
            "ERROR: y_train is empty after feature engineering."
        assert y_test.shape[0] > 0, \
            "ERROR: y_test is empty after feature engineering."
        logging.info("SUCCESS: Feature engineering completed successfully.")

    except AssertionError as err:
        logging.error(err)
        raise err

    except Exception as err:
        logging.error(
            "ERROR: Unexpected failure during feature engineering test: %s",
            err)
        raise err


def test_train_models():
    """Test model training."""
    try:

        df = cls.import_data(DATA_PATH)
        cls.perform_eda(df)
        df = cls.encoder_helper(df, category_columns)
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            df, response="Churn")
        cls.train_models(x_train, x_test, y_train, y_test)

        model_files = [
            'rfc_model.pkl',
            'logistic_model.pkl'
        ]

        for model_file in model_files:
            assert os.path.exists(
                f'{MODEL_DIR}/{model_file}'), \
                f"ERROR: Model file {model_file} not found in {MODEL_DIR}."
        logging.info("SUCCESS: Models trained and saved successfully.")

        result_images = [
            'roc_curve.png',
            'feature_importances_lr.png',
            'feature_importances_rf.png',
            'logistic_classification_report.png',
            'rf_classification_report.png'
        ]
        for result_image in result_images:
            assert os.path.exists(
                f'{IMG_DIR}/{result_image}'), \
                f"ERROR: Result image {result_image} not found in {IMG_DIR}."
        logging.info("SUCCESS: Model result images saved successfully.")

    except AssertionError as err:
        logging.error(err)
        raise err

    except Exception as err:
        logging.error(
            "ERROR: Unexpected failure during model training test: %s", err)
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()

    print("Tests completed. Check logs for details.")
