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

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    """Test data import."""
    try:
        df = cls.import_data(DATA_PATH)
        logging.info("SUCCESS: Data import successful.")

    except FileNotFoundError:
        logging.error("ERROR: File not found.")
        raise FileNotFoundError("ERROR: File not found.")

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError:
        logging.error("ERROR: DataFrame is empty. Check the file and contents.")
        raise AssertionError("ERROR: DataFrame is empty. Check the file and contents.")


def test_eda(perform_eda):
    """Test EDA."""
    try:
        df = cls.import_data(DATA_PATH)
        cls.perform_eda(df)

        EDA_DIR = "./images/eda"
        image_files = [
            'churn_distribution.png',
            'customer_age_distribution.png',
            'marital_status_distribution.png',
            'total_trans_count_distribution.png',
            'correlation_heatmap.png'
        ]

        for image_file in image_files:
            try:
                assert os.path.exists(f'{EDA_DIR}/{image_file}')
            except AssertionError:
                logging.error("ERROR: Image file not found.")
                raise AssertionError(f"Image file {image_file} not found in {EDA_DIR}.")
        logging.info("SUCCESS: EDA images generated successfully.")

    except Exception as err:
        logging.error("ERROR: Failed to generate EDA images.")
        raise err


def test_encoder_helper(encoder_helper):
    """Test encoding."""
    try:
        df = cls.import_data(DATA_PATH)

        category_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"
        ]

        df = cls.encoder_helper(df, category_columns)
        for category in category_columns:
            encoded_columns = [col for col in df.columns if col.startswith(category + "_")]
            unique_values = df[category].nunique()

            try:
                assert len(encoded_columns) > 0
            except AssertionError:
                logging.error(f"ERROR. No encoded columns found for category {category}.")
                raise AssertionError(f"ERROR. No encoded columns found for category {category}.")

            try:
                assert len(encoded_columns) == unique_values - 1
            except AssertionError:
                logging.error("ERROR. Encoded columns do not match expected number based on unique values.")
                raise AssertionError(f"ERROR. Encoded columns do not match expected number based on unique values.\
                                     Expected {unique_values - 1} encoded columns for {category}, but found {len(encoded_columns)}.")
        
        logging.info("SUCCESS: Category column encoding successful and verified.")

    except Exception as err:
        logging.error("ERROR: Failed to encode category columns.")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """Test feature engineering."""
    try:
        df = cls.import_data(DATA_PATH)

        # TODO:
        # prepare data
        # call function
        # assert outputs

        # TODO: logging success

    except Exception as err:

        # TODO: logging failure

        raise err


def test_train_models(train_models):
    """Test model training."""
    try:
        df = cls.import_data(DATA_PATH)

        # TODO:
        # prepare data
        # call train_models
        # assert model files + images exist

        # TODO: logging success

    except Exception as err:

        # TODO: logging failure

        raise err


if __name__ == "__main__":
    # TODO: ensure logs directory exists

    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

    print("Tests completed. Check logs for details.")
