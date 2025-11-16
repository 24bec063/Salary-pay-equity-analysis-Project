# Notebooks Directory

This directory contains Jupyter notebooks for the Machine Learning project.

## Notebooks

### project.ipynb
- **Description**: Main project notebook containing all machine learning tasks
- **Contents**:
  - **Subtask 1**: Salary Prediction (Regression)
    - Data loading and cleaning
    - Exploratory Data Analysis (EDA)
    - Linear Regression and Random Forest models
    - Evaluation metrics (MAE, RMSE, R²)
  - **Subtask 2**: Salary Fairness Classification (Binary Classification)
    - Fair vs Unfair salary labels
    - Logistic Regression model
    - Classification metrics (Accuracy, Precision, Recall, F1-score)

## How to Run

```bash
# Option 1: Using run.sh
bash run.sh

# Option 2: Using Jupyter directly
jupyter notebook notebooks/project.ipynb
```

## Output Files

- `project_output.ipynb`: Executed notebook with all results and visualizations
- `salary_prediction_results.csv`: Predictions and residuals from regression models
