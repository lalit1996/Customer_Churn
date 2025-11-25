# Customer Churn Model

**Repository:** Customer Churn Classification (Logistic Regression)

## Overview

This repository contains a simple machine-learning pipeline to train and test a customer churn prediction model using a logistic regression classifier. The code demonstrates data loading, basic preprocessing (imputation, encoding, scaling), training, evaluation (accuracy, F1 score, confusion matrix), and a class-based structure to encapsulate the model logic.

## Files

* `customer_churn_dataset-testing-master.csv` — test dataset (expected in repo root)
* `customer_churn_dataset-training-master.csv` — training dataset (expected in repo root)
* `churn_model.py` — main Python script (the code you provided)
* `README.md` — this file

> **Note:** The repository assumes the dataset files exist in the same directory as the script. Adjust paths if you store data elsewhere.

## Requirements

Create a Python virtual environment and install dependencies. Example using `pip`:

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate     # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

An example `requirements.txt` could contain:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## How to run

From the repository root (where the CSV files and the script live):

```bash
python churn_model.py
```

The script will:

1. Load the training and testing CSV files
2. Concatenate datasets (used during preprocessing)
3. Instantiate `Customer_Churn_Model`
4. Train a logistic regression model (with `class_weight='balanced'`)
5. Print training accuracy and test metrics (F1 score, accuracy, confusion matrix)

## Class: `Customer_Churn_Model`

The class wraps preprocessing, training and testing logic. Key methods:

* `__init__(...)` — stores a single-customer example (used by the commented `Prediction` method)
* `Preprocessing()` —

  * Drops NA from training data
  * Identifies categorical vs numerical columns
  * Builds pipelines: ordinal encoding + imputation + scaling for categorical, imputation + scaling for numerical
  * Returns `X_Scaled_training`, `Y_Training`, and the `ColumnTransformer` (named `final`)
* `Model_Train()` —

  * Calls `Preprocessing()`
  * Builds a `LogisticRegression` (penalty, solver, C, max_iter currently hard-coded)
  * Splits train/test using `train_test_split`
  * Fits preprocessing on `x_train` and transforms `x_test`
  * Trains logistic regression and returns `(accuracy_score, x_test, trained_model)`
* `Predict_testing()` —

  * Trains model via `Model_Train()`
  * Transforms the held-out testing dataset using the pipeline
  * Returns `(f1_score, accuracy_score, confusion_matrix)` on test CSV

## Important notes & recommendations

* **Column dtypes:** The current code treats all `float64` columns as numerical and everything else as categorical. If a column is `int64` or other numeric type, it will be misclassified as categorical. Consider using `pd.api.types.is_numeric_dtype` to detect numeric columns robustly.

* **OrdinalEncoder on unordered categories:** `OrdinalEncoder` imposes an order on categories. If there is no ordinal relationship, use `OneHotEncoder` or `OrdinalEncoder` with caution. For tree-based models you may keep label encoding, but for linear models one-hot encoding is often preferable.

* **Imputer for numerical columns:** The pipeline currently uses `SimpleImputer(strategy='most_frequent')` for both categorical and numerical columns. For numerical columns, consider `strategy='median'` or `mean`.

* **Scaling and encoding order:** The code scales categorical features after ordinal-encoding. This is acceptable for many models but be explicit which features are encoded vs. scaled.

* **Train/test leakage:** The `ColumnTransformer` is fit only on `x_train` inside `Model_Train()` — this is correct. Avoid fitting transformers on the entire dataset before splitting.

* **Class imbalance:** The model uses `class_weight='balanced'` in `LogisticRegression`. You can also try resampling (oversampling minority class or undersampling majority class) or experiment with `SMOTE`.

* **Hyperparameter tuning (commented):** Grid search with `StratifiedKFold` is present but commented out. If you need to run it, un-comment `HyperParameter()` and use fewer parameter combinations or smaller sample sizes for speed.

* **Prediction method:** The `Prediction()` method is commented and contains errors (indexing/transform usage). If you want single-record prediction, build a small helper that constructs a DataFrame with the same columns, applies the fitted `ColumnTransformer`, then calls `model.predict()`.

* **SVD / rank checks:** The randomized SVD code is commented. If you plan to check multicollinearity or linear dependence, prefer variance inflation factor (VIF) or correlation checks.

## Suggested improvements / TODO

* Add a `requirements.txt` and a small `run_example.py` that demonstrates training and predicting on a few records.
* Fix the `Prediction()` method to use the fitted `ColumnTransformer`, `OrdinalEncoder`, and `StandardScaler` correctly.
* Add unit tests for the preprocessing pipelines.
* Replace `OrdinalEncoder` with `OneHotEncoder` (or `ColumnTransformer` that applies OHE on nominal categorical columns) for linear model compatibility.
* Improve dtype detection to properly separate numerical and categorical columns.
* Add logging and better error handling when CSV files are missing or columns are unexpected.

## Example quickstart (Python snippet)

```python
from churn_model import Customer_Churn_Model

model = Customer_Churn_Model(55, 'Female', 14, 4, 6, 18, 'Basic', 'Quarterly', 185, 3)
train_acc, _, _ = model.Model_Train()
print('Train accuracy:', train_acc)

f1, test_acc, conf = model.Predict_testing()
print('Test F1:', f1)
print('Test accuracy:', test_acc)
print('Confusion matrix:\n', conf)
```

## License

This example is provided for educational purposes. Add a license file (for example, MIT) if you intend to open-source it.

---

If you'd like, I can:

* Generate a `requirements.txt` file for you.
* Update the code to fix the `Prediction()` method and improve dtype detection.
* Convert the categorical pipeline to use `OneHotEncoder` instead of `OrdinalEncoder`.

Tell me which of the above you'd like next.
