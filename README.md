# Machine Learning: Salary Prediction & Fairness Analysis

A comprehensive machine learning project for predicting developer salaries and analyzing pay fairness using Stack Overflow survey data.

## 🎯 Project Overview

This project consists of two main tasks:

### **Task 1: Salary Prediction (Regression)**
- **Goal**: Predict developer salaries based on experience, education, role, and location
- **Models Used**: 
  - Linear Regression (baseline)
  - Random Forest Regressor (main model)
- **Evaluation Metrics**: MAE, RMSE, R²
- **Output**: Feature importance analysis showing key salary factors

### **Task 2: Salary Fairness Classification**
- **Goal**: Classify whether developers are paid fairly relative to their predicted salary
- **Definition**: Fair = Actual Salary ≥ 80% of Predicted Salary
- **Model Used**: Logistic Regression
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Output**: Feature analysis showing which factors affect pay fairness

## 📊 Dataset

- **Source**: Stack Overflow Developer Survey (2024)
- **Size**: ~65,000 survey responses, 114 columns
- **Key Features Used**:
  - ConvertedCompYearly (Target for regression)
  - YearsCodePro (Years of professional experience)
  - Country, Education Level, Job Role
  - Age, Employment Type, Organization Size

## 🗂️ Project Structure

```
project/
├── data/                           # Data directory
│   ├── README.md                  # Data documentation
│   └── survey_results_public.csv  # Main dataset (not in repo)
├── notebooks/                      # Jupyter notebooks
│   ├── README.md
│   └── project.ipynb              # Main project notebook
├── src/                            # Source code modules
│   └── README.md
├── reports/                        # Generated reports
│   ├── README.md
│   └── salary_prediction_results.csv
├── figures/                        # Generated visualizations
│   └── README.md
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── run.sh                         # Automated execution script
└── .gitignore                     # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- pip or conda

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/salary-prediction.git
cd salary-prediction
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download the dataset**:
- Download `survey_results_public.csv` from [Stack Overflow Survey](https://survey.stackoverflow.co/)
- Place it in the `data/` folder

### Running the Project

**Option 1: Automated execution**
```bash
bash run.sh
```

**Option 2: Manual Jupyter execution**
```bash
jupyter notebook notebooks/project.ipynb
```

## 📈 Key Results

### Salary Prediction Model (Random Forest)
- **R² Score**: ~0.65-0.70 (explains 65-70% of variance)
- **MAE**: ~$15,000-20,000
- **RMSE**: ~$25,000-35,000

### Top Salary Factors
1. Experience (Years of Professional Coding)
2. Country/Location
3. Job Role/Developer Type
4. Education Level
5. Organization Size

### Fairness Analysis
- Identifies underpaid vs fairly paid professionals
- Shows which roles/countries/education levels have pay equity issues
- Provides actionable insights for compensation analysis

## 📁 File Descriptions

| File | Purpose |
|------|---------|
| `project.ipynb` | Main Jupyter notebook with full analysis |
| `project_output.ipynb` | Executed notebook with all results |
| `run.sh` | Automated script to install dependencies and run notebook |
| `requirements.txt` | Python package dependencies |
| `salary_prediction_results.csv` | Model predictions and residuals |

## 🛠️ Technology Stack

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Notebooks**: Jupyter
- **Environment**: Python 3.13+

## 📊 Dependencies

All required packages are listed in `requirements.txt`:
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
jupyter>=1.0.0
nbconvert>=7.8.0
```

## 💡 Usage Examples

### Run the complete pipeline
```bash
bash run.sh
```

### Access results programmatically
```python
import pandas as pd
results = pd.read_csv('reports/salary_prediction_results.csv')
print(results.head())
```

## 📝 Notes

- The dataset is large (~153 MB) and is not included in the Git repository
- First run will take 5-10 minutes to process the data and train models
- All visualizations are saved in the `figures/` directory
- Results are saved to `reports/salary_prediction_results.csv`

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Last Updated**: November 17, 2025
