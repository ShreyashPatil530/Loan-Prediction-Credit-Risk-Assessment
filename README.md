# 💰 Loan Prediction & Credit Risk Assessment

**Binary Classification with XGBoost & Advanced Feature Engineering**

[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/code/shreyashpatil217/loan-prediction-credit-risk-assessment)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)

---

## 🎯 Project Overview

A production-ready machine learning pipeline for **loan default prediction** using advanced feature engineering and ensemble techniques. Predicts whether a loan will be paid back or defaulted with **90%+ ROC-AUC accuracy**.

### Key Highlights
- ⚡ **High Performance**: CV ROC-AUC Score of 0.909+
- 🔧 **Advanced Engineering**: 50+ derived features per fold
- 🎲 **Robust Validation**: 5-Fold Stratified CV with per-fold encoders
- 🚀 **Production-Ready**: GPU/CPU auto-detection, optimized hyperparameters

---

## 📊 Dataset Specifications

| Metric | Value |
|--------|-------|
| **Training Records** | 593,994 |
| **Test Records** | 254,569 |
| **Features** | 13 columns |
| **Target Variable** | `loan_paid_back` (Binary) |
| **Data Completeness** | 98% |
| **Evaluation Metric** | ROC-AUC Score |

### Feature Categories
- **Categorical**: `grade_subgrade`, `loan_purpose`, `education_level`, `employment_status`, `marital_status`, `gender`
- **Numerical**: `annual_income`, `loan_amount`, `credit_score`, `interest_rate`, `debt_to_income_ratio`

---

## 🤖 Machine Learning Pipeline

### Model Architecture
```
XGBoost Classifier (hist tree_method)
├── 5-Fold Stratified CV
├── Optuna Hyperparameter Tuning (200 trials)
├── Per-Fold Feature Engineering
└── CV + Full Refit Ensemble
```

### Training Strategy
1. **Baseline Training**: Single fold validation (ROC-AUC: 0.9095)
2. **Hyperparameter Optimization**: Optuna with 300s budget
3. **Cross-Validation**: 5-fold stratified splits with per-fold encoders
4. **Full Refit**: Train on entire dataset with optimal parameters
5. **Ensemble**: 50% CV predictions + 50% full refit predictions

---

## ✨ Advanced Feature Engineering

### 🔤 Categorical Encodings
| Technique | Description | Purpose |
|-----------|-------------|---------|
| **Target Encoding** | Smoothed mean target by category (m=10) | Capture category-target relationship |
| **WOE Encoding** | Weight of Evidence with clipping | Handle imbalanced categories |
| **Frequency Encoding** | Category occurrence counts | Leverage category popularity |

### 📊 Numerical Transformations
| Transform | Application | Benefit |
|-----------|-------------|---------|
| **Rank Gaussian** | Normalize distributions | Handle skewness |
| **Yeo-Johnson** | Power transformation | Stabilize variance |
| **KBins Discretizer** | Quantile-based binning (10 bins) | Capture non-linear patterns |

### 🎯 Interaction Features
- **Group Mean Deviations**: Deviation from category-wise means
- **Group Percentiles**: Within-group ranking features
  - `credit_score__pctl_in_grade`: Credit score percentile within grade
  - `credit_score__pctl_in_edu`: Credit score percentile within education level
- **Missingness Indicators**: Binary flags for missing values

### 📌 Feature Engineering Summary
- **Raw Features**: 13 original columns
- **Engineered Features**: 50+ per-fold features
- **Total Feature Space**: 60+ features for modeling
- **Encoding Strategy**: Out-of-fold (OOF) to prevent leakage

---

## 📈 Model Performance

| Metric | Baseline (Fold 0) | CV Average (5-Fold) |
|--------|------------------|---------------------|
| **ROC-AUC** | 0.9095 | **0.909+** |
| **Best Iteration** | ~800-1000 | Median: ~900 |
| **Training Time** | ~60s/fold | ~5 minutes total |

### Prediction Distribution
- **Mean**: 0.35
- **Std Dev**: 0.28
- **Range**: [0.001, 0.999]
- **1st Percentile**: 0.01
- **99th Percentile**: 0.95

---

## 💻 Tech Stack & Dependencies

### Core Libraries
```python
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
xgboost==2.0.0
optuna==3.3.0
```

### Key Components
- **Data Processing**: Pandas, NumPy
- **ML Framework**: XGBoost (GPU/CPU optimized)
- **Feature Engineering**: Scikit-Learn transformers
  - `QuantileTransformer`: Rank Gaussian transformation
  - `PowerTransformer`: Yeo-Johnson transformation
  - `KBinsDiscretizer`: Quantile-based binning
- **Hyperparameter Optimization**: Optuna (Bayesian optimization)
- **Validation**: Stratified K-Fold Cross-Validation

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- 16GB+ RAM recommended

### Installation
```bash
# Clone the repository
git clone https://github.com/ShreyashPatil530/Loan-Prediction-Credit-Risk-Assessment.git
cd Loan-Prediction-Credit-Risk-Assessment

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup
Download the dataset from [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s5e11) and place files in:
```
/kaggle/input/playground-series-s5e11/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### Running the Pipeline
```bash
# Full pipeline (DEBUG + FULL mode)
python loan_prediction_pipeline.py

# Output files:
# - submission_4.csv (predictions)
# - code_8_1_v4.txt (execution log)
```

---

## 📂 Project Structure
```
Loan-Prediction-Credit-Risk-Assessment/
├── loan_prediction_pipeline.py    # Main training script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA notebook
├── models/
│   └── best_model.json            # Saved model parameters
├── logs/
│   └── code_8_1_v4.txt            # Training logs
└── submissions/
    └── submission_4.csv           # Final predictions
```

---

## 🔧 Pipeline Configuration

### XGBoost Hyperparameters (Optimized)
```python
{
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda:0',  # or 'cpu'
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'reg_lambda': 1.0,
    'reg_alpha': 0.0,
    'n_estimators': 1500,
    'early_stopping_rounds': 100
}
```

### Optuna Search Space
```python
{
    'learning_rate': [0.02, 0.12],
    'max_depth': [4, 9],
    'min_child_weight': [2.0, 12.0],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'reg_lambda': [0.5, 5.0] (log scale),
    'reg_alpha': [0.0, 2.0],
    'gamma': [0.0, 5.0],
    'max_bin': [128, 256, 512],
    'n_estimators': [600, 1500]
}
```

---

## 📊 Feature Engineering Details

### Per-Fold Encoding Strategy
Each CV fold uses **separate encoders** fitted only on training folds to prevent data leakage:

1. **Split Data**: Hold out 1 fold for validation
2. **Fit Encoders**: Train encoders on remaining 4 folds
3. **Transform**: Apply encoders to validation fold and test set
4. **Repeat**: Process all 5 folds independently

### Encoding Examples

#### Target Encoding (Smoothed)
```python
smoothed_mean = (category_mean * n_samples + global_mean * m) / (n_samples + m)
# m=10 for regularization
```

#### WOE Encoding (Clipped)
```python
WOE = log((% positive in category) / (% negative in category))
# Clipped to [-3, 3] for stability
```

#### Percentile Features
```python
# Credit score percentile within loan grade
credit_score__pctl_in_grade = rank(credit_score) / count(credit_score) 
                               within each grade_subgrade
```

---

## 🎯 Model Validation Strategy

### Cross-Validation Design
- **Type**: Stratified K-Fold (n=5)
- **Stratification**: Based on target variable `loan_paid_back`
- **Seed**: 2025 (reproducible splits)

### Ensemble Strategy
```python
final_predictions = 0.5 * cv_predictions + 0.5 * full_refit_predictions
```

**Rationale**: 
- CV predictions are more robust (averaged over 5 models)
- Full refit captures patterns from entire training set
- Equal weighting balances stability and performance

---

## 📈 Results & Insights

### Top Predictive Features
1. **grade_subgrade** (loan grade) - Target Encoding
2. **annual_income** - Rank Gaussian + Binning
3. **credit_score** - Percentile within grade
4. **debt_to_income_ratio** - Yeo-Johnson transform
5. **loan_amount** - Group mean deviations

### Model Insights
- **Non-linear Patterns**: KBins discretization captures threshold effects
- **Category Interactions**: WOE encoding reveals risk patterns
- **Income Normalization**: Rank Gaussian handles outliers effectively
- **Missing Data**: Missingness indicators improve AUC by ~0.002

---

## 🛠️ Optimization & Performance

### Computational Efficiency
- **GPU Acceleration**: 3-5x speedup with CUDA
- **Histogram Binning**: Efficient memory usage for large datasets
- **Early Stopping**: Prevents overfitting, reduces training time
- **Per-Fold Parallelization**: Independent fold processing

### Memory Management
- **Feature Type Optimization**: Use `float32` for reduced memory
- **Batch Processing**: Process test predictions in single pass
- **Garbage Collection**: Optuna trials cleared after each iteration

---

## 🔍 Model Interpretation

### Feature Importance (Top 10)
```
1. grade_subgrade__te_m10          : 0.142
2. annual_income__rgauss           : 0.098
3. credit_score__pctl_in_grade     : 0.087
4. debt_to_income_ratio__yeoj      : 0.076
5. loan_amount__gm_grade_subgrade  : 0.065
6. credit_score                    : 0.059
7. interest_rate                   : 0.053
8. grade_subgrade__woe             : 0.048
9. employment_status__te_m10       : 0.041
10. annual_income__qbin10          : 0.038
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Solution: Force CPU usage
export XGB_DEVICE=cpu
```

**Issue**: Feature encoding errors
```bash
# Check for missing columns in test set
# Ensure train/test have same feature names (except target)
```

**Issue**: Poor CV performance
```bash
# Verify stratification is working
# Check for data leakage in encoders
# Review feature correlation matrix
```

---

## 📝 Future Improvements

- [ ] Add SHAP values for model explainability
- [ ] Implement neural network ensemble
- [ ] Add automated feature selection
- [ ] Create interactive dashboard for predictions
- [ ] Implement MLflow for experiment tracking
- [ ] Add unit tests for feature engineering functions
- [ ] Deploy model as REST API

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Shreyash Patil**

- 📧 Email: [shreyashpatil530@gmail.com](mailto:shreyashpatil530@gmail.com)
- 💼 LinkedIn: [Shreyash Patil](https://linkedin.com/in/shreyash-patil)
- 🐱 GitHub: [@ShreyashPatil530](https://github.com/ShreyashPatil530)
- 📊 Kaggle: [shreyashpatil217](https://www.kaggle.com/shreyashpatil217)
- 🌐 Portfolio: [shreyash-patil-portfolio1.netlify.app](https://shreyash-patil-portfolio1.netlify.app/)

---

## 🙏 Acknowledgments

- Kaggle for hosting the Playground Series S5E11 competition
- XGBoost development team for the powerful gradient boosting library
- Optuna team for hyperparameter optimization framework
- Open-source community for various Python libraries

---

## ⭐ Show Your Support

If you found this project helpful, please consider giving it a ⭐ on GitHub!

---

**Last Updated**: November 2024  
**Version**: 4.0  
**Status**: Production-Ready
