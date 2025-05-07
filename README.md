# Bank Customer Churn Classification

## Project Overview

This repository contains a complete machine learning pipeline to predict customer churn for a major bank. The goal is to identify clients at high risk of leaving, enabling proactive retention strategies.

### Key Highlights:

- Classification problem with a binary target (Exited).
- No missing values or imputation required.
- Feature removal to prevent data leakage and improve interpretability.
- Comprehensive model comparison with cross-validation and hyperparameter tuning.
- Final evaluation on a held-out test set with multiple metrics and visualizations.
- Decision to omit SMOTE due to observed negative impact on model generalization.

## Dataset

The dataset, obtained from Kaggle, contains 10,000 records with 14 features plus the target. Each row represents a unique customer and includes demographic and banking behavior attributes:

### Feature Descriptions:

| Feature         | Description                                         |
|-----------------|-----------------------------------------------------|
| **CreditScore**     | Customer credit score                               |
| **Geography**       | Customer country (one of France, Spain, Germany)    |
| **Gender**          | Male or Female                                      |
| **Age**             | Customer age                                        |
| **Tenure**          | Number of years as a customer                       |
| **Balance**         | Account balance                                     |
| **NumOfProducts**   | Number of banking products used by the customer     |
| **HasCrCard**       | Does the customer have a credit card? (0 = No, 1 = Yes) |
| **IsActiveMember**  | Is the customer an active member? (0 = No, 1 = Yes) |
| **EstimatedSalary** | Estimated annual salary                             |
| **Exited**          | Target: did the customer leave? (0 = No, 1 = Yes)  |

### Removed Features:

- **RowNumber, CustomerId, Surname:** Unique identifiers with no predictive power.
- **Complain:** Highly predictive flag (nearly perfect) that leaks future information. Excluded to avoid data leakage.

## Exploratory Data Analysis (EDA)

- Target distribution: ~20% churn rate, indicating class imbalance.
- Correlation heatmap and feature distributions explored with histograms, countplots, and KDE plots.
- Decision to not apply SMOTE in production pipeline based on empirical tests: SMOTE caused overfitting and degraded performance on the majority class.

## Data Preprocessing

### Feature Removal:
- Drop identifiers and leakage-prone columns.

### Encoding:
- **Label encoding** for binary variables (e.g., Gender).
- **One-hot encoding** for categorical variables (Geography).

### Scaling:
- Standard scaling applied to all numerical features for gradient-based models.

### Train-Test Split:
- 80/20 stratified split on the target to preserve class distribution.

## Model Selection and Training

Three tree-based models were chosen for their robustness:

- **Gradient Boosting Classifier (sklearn)**
- **XGBoost Classifier**
- **Random Forest Classifier**

All models underwent hyperparameter tuning using **GridSearchCV** with 5-fold stratified cross-validation, optimizing for F1-score (balanced measure of precision and recall).

### Hyperparameter Grids:

- **Random Forest:** n_estimators, max_depth, min_samples_split, max_features.
- **Gradient Boosting:** n_estimators, learning_rate, max_depth, subsample.
- **XGBoost:** n_estimators, learning_rate, max_depth, subsample, colsample_bytree.

## Evaluation Metrics

Final evaluation on the test set used multiple metrics to capture different aspects of performance:

- **Accuracy:** Overall correctness.
- **Precision:** Proportion of correctly predicted churners among all predicted churners.
- **Recall (Sensitivity):** Ability to detect actual churners.
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC AUC:** Area under the Receiver Operating Characteristic curve.
- **Precision-Recall AUC:** Area under the Precision-Recall curve.

### Final Model Performance (XGBoost):

- **Weighted F1-score:** ~0.86
- **ROC AUC:** ~0.88

Visuals include confusion matrix, ROC and PR curves, and feature importance plots.

## Model Choice Rationale

The **XGBoost Classifier** was selected as the final model because:

- Achieved the highest F1-score during cross-validation and on the test set.
- Demonstrated a good balance between precision and recall.
- Fast training time and built-in handling of feature interactions.


The dataset is available on [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn).
