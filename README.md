# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree (Udacity)

---

## Project Description

The purpose of this project is to solve a classic customer retention problem by building an automated, end-to-end machine learning engineering workflow. Using a dataset of bank customer demographics and credit account behaviors, the pipeline automates:
- Data ingestion and strict schema/type validation.
- Exploratory Data Analysis (EDA) with headless automated plot exporting.
- Categorical feature transformation via target-mean encoding.
- Dataset train/test splitting and standardized feature scaling.
- Parallelized hyperparameter tuning (`RandomizedSearchCV`) and model training.
- Complete metric generation, including classification report graphics and ROC curve outputs.

The pipeline utilizes two core model families:
1. **Random Forest Classifier:** A non-linear ensemble tree model optimized for capturing complex feature interactions.
2. **Logistic Regression:** A robust, linear classification model used as an interpretable baseline.

---

## Files and Data Description

### Main Files

- `churn_library.py`  
  The core pipeline implementation file. It houses modular, production-ready functions that manage directory generation, data loading, EDA plotting, target encoding, feature matrix transformations, model fitting, and metrics serialization.

- `churn_script_logging_and_tests.py`  
  The integrated testing and logging framework written for the `pytest` runner. It runs isolated unit and integration validation checks on every step of the pipeline, using standard `try-except` wrappers to document success checkpoints and traceback assertions cleanly to a log file.

- `churn_notebook.ipynb`  
  The initial prototyping notebook used to map out the foundational code logic, data exploration patterns, and visualization styles prior to refactoring into production-grade Python scripts.

---

### Data

- `data/bank_data.csv`  
  The source dataset containing information for 10,127 bank customers. It comprises 21 columns mapping customer metadata (e.g., `Customer_Age`, `Gender`, `Education_Level`, `Marital_Status`, `Income_Category`) alongside financial variables (e.g., `Credit_Limit`, `Total_Revolving_Bal`, `Total_Trans_Amt`, `Total_Trans_Ct`, and `Avg_Utilization_Ratio`). The target feature is a synthesized binary indicator (`Churn`) derived from the original `Attrition_Flag` attribute.

---

### Output Directories

After running the project, outputs will be saved to:

- EDA images → `images/eda/` (`churn_distribution.png`, `customer_age_distribution.png`, `marital_status_distribution.png`, `total_transaction_distribution.png`, `heatmap.png`)
- Model results → `images/results/` (`roc_curve.png`, `feature_importances_rf.png`, `feature_importances_lr.png`, `logistic_classification_report.png`, `rf_classification_report.png`)
- Models → `models/` (`rfc_model.pkl`, `logistic_model.pkl`)
- Logs → `logs/churn_library.log`

---

## Dependencies and Installation

This project is optimized for Python `3.12` and manages dependencies using `uv` for environment parity and safety.

### 1. Install Dependencies
If using `uv`, you can synchronize the environment directly from the workspace definition:

```bash
uv pip install -r requirements.txt
```

Alternatively, use standard pip:
```bash
pip install -r requirements.txt
```

### Key Libraries Used
- **pandas & numpy:** Data manipulation and matrix arithmetic.
- **scikit-learn:** Feature preprocessing, training, randomized hyperparameter searches, and evaluation metrics.
- **matplotlib & seaborn:** Chart rendering and graphic generation using non-interactive vector graphics backends (`matplotlib.use('Agg')`).
- **joblib:** Binary object persistence for model serialization.
- **pytest:** Unified command-line testing runner.

---

## Running the Files

### 1. Run the Pipeline

```bash
python churn_library.py
```

Running this file triggers the full operational workflow. It builds the necessary workspace directory trees, pulls the source CSV, exports distribution charts and correlation heatmaps to disk, processes all string metrics into target-encoded variants, performs train/test splits, runs cross-validated hyperparameter sweeps, dumps optimized models into the `/models` path, and automatically saves performance evaluation curves.

---

### 2. Run Tests and Logging

```bash
python churn_script_logging_and_tests.py
```

This command runs the custom automated test suite via the pytest CLI runner. It validates file system architectures, data shapes, feature dimensions, and tracking states across your module. Every milestone and error catch writes directly to `./logs/churn_library.log` with formatted timestamps, trace logs, and success declarations.

---

## Expected Outputs

Executing the code and tests guarantees the automated creation of the following production artifacts:

1. Extracted EDA Plots (`images/eda/`)
    `churn_distribution.png`: Histogram detailing the relative proportion of active vs. churned clients.
    `customer_age_distribution.png`: Density breakdown of client age demographics.
    `marital_status_distribution.png`: Categorical status distribution bar charts.
    `total_transaction_distribution.png`: Kernel density plot of total account transaction frequencies.
    `heatmap.png`: High-resolution Pearson correlation matrix tracking numeric columns.

2. Model Performance Deliverables (`images/results/`)
    `roc_curve.png`: Comparison plot graphing receiver operating characteristics for both classifiers.
    `feature_importances_rf.png`: Sorted horizontal bar chart tracking feature importances for Random Forest.
    `feature_importances_lr.png`: Coefficients magnitude visualization detailing feature impact weight for Logistic Regression.
    `rf_classification_report.png`: Graphical text matrix displaying precision, recall, and f1-score splits for train and test segments.
    `logistic_classification_report`.png: Linear baseline evaluation matrix for train and test segments.

3. Production Models and Logs
    `models/rfc_model.pkl`: Serialized, optimized Random Forest estimator.
    `models/logistic_model.pkl`: Serialized baseline Logistic Regression estimator.
    `logs/churn_library.log`: Verified pipeline execution and error messages.
