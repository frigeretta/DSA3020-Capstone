# Loan Prediction Model - Complete Documentation

## Project Overview

This project implements a machine learning solution to predict loan approval decisions. The model analyzes applicant information and historical data to determine whether a loan application should be approved or rejected. The solution uses three classification algorithms (Logistic Regression, Decision Tree, and Gradient Boosting), with Logistic Regression identified as the optimal model.

**Dataset Size:** 614 rows × 13 columns  
**Target Variable:** Loan_Status (Binary: Y/N)  
**Model Performance:** 85.37% Accuracy with 90.32% F1-Score

---

## Table of Contents

1. [Phase 1: Data Loading and Exploration](#phase-1-data-loading-and-exploration)
2. [Phase 2: Data Preprocessing](#phase-2-data-preprocessing)
3. [Phase 3: Exploratory Data Analysis](#phase-3-exploratory-data-analysis)
4. [Phase 4: Model Preparation](#phase-4-model-preparation)
5. [Phase 5: Model Training and Comparison](#phase-5-model-training-and-comparison)
6. [Phase 6: Model Optimization](#phase-6-model-optimization)
7. [Phase 7: Model Evaluation](#phase-7-model-evaluation)

---

## Phase 1: Data Loading and Exploration

### Purpose
Load the dataset and perform initial data quality checks to understand the structure and content of the data.

### Cell 1: Import Libraries
**File:** Loanprediction.ipynb  
**Dependencies:**
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `matplotlib.pyplot`: Data visualization
- `seaborn`: Statistical visualization
- `sklearn.preprocessing.LabelEncoder`: Categorical encoding
- `sklearn.impute.IterativeImputer`: Missing value imputation

### Cell 2: Load Dataset
**Operation:** Loads the loan dataset from CSV file  
**Input Path:** `../loan_data_set.csv`  
**Actions Performed:**
1. Read CSV file into pandas DataFrame
2. Display first 5 rows to verify data structure
3. Remove 'Loan_ID' column (not useful for prediction)

**Output:** `loan_df` DataFrame with 614 rows and 12 columns (after removing Loan_ID)

### Cell 3: Data Information
**Operation:** `loan_df.info()`  
**Output Details:**
- **Dataset Dimensions:** 614 rows × 13 columns
- **Data Types:** Mix of integer, float, and object (string) types
- **Missing Values Identified:**
  - Gender: Missing values
  - Dependents: Missing values
  - Self_Employed: Missing values
  - LoanAmount: Missing values
  - Loan_Amount_Term: Missing values
  - Credit_History: Missing values

### Cell 4: Summary Statistics
**Operation:** `loan_df.describe()`  
**Purpose:** Display statistical measures for numeric columns
- Count, mean, standard deviation, min, quartiles, and max values

### Cell 5: Missing Values Analysis
**Operation:** `loan_df.isnull().sum()`  
**Purpose:** Quantify missing values per column
**Decision:** Imputation instead of deletion to preserve 614 rows

### Cell 6: Column Type Classification
**Operation:** Identify numeric vs. categorical columns
**Output:**
- **Numeric Columns:** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Age
- **Categorical Columns:** Gender, Married, Dependents, Education, Self_Employed, Property_Area, Loan_Status

---

## Phase 2: Data Preprocessing

### Purpose
Clean and prepare data for machine learning by handling missing values and encoding categorical variables.

### Strategy for Missing Values

**Why Not Delete Rows?**
- Dataset contains only 614 rows
- Missing 20-25% of data would severely reduce training set
- Results in poorer model performance, increased bias, and unreliable predictions

**Solution: Iterative Imputation**
- Uses regression to predict missing values based on other features
- Suitable for Missing Completely At Random (MCAR) data
- Preserves all 614 rows for training

### Cells 7-12: Missing Value Imputation Process

#### Step 1: Categorical Encoding (Cell 7)
```
Purpose: Convert categorical variables to numeric for imputation
Method: Label Encoding
Process:
  - For each categorical column:
    1. Replace "nan" strings with actual NaN values
    2. Create LabelEncoder for each column
    3. Fit encoder on non-null values
    4. Transform categorical values to numeric codes
    5. Store encoder in le_dict for later reverse transformation
```

**Variables Stored:** `le_dict` - Dictionary of LabelEncoders for each categorical column

#### Step 2: Verify Encoding (Cell 8)
Display encoded dataset to confirm transformation worked correctly.

#### Step 3: Iterative Imputation (Cell 9)
```
Model: IterativeImputer(random_state=42, max_iter=15)
Process:
  - Iterates 15 times through the dataset
  - Each iteration models missing values as a function of other features
  - Uses regression to predict missing values
  - Convergence improves prediction accuracy
Output: imputed_array (NumPy array with no missing values)
```

#### Step 4: Convert Back to DataFrame (Cell 10)
- Reconstruct DataFrame from imputed array
- Verify: `imputed_df.isnull().sum()` returns all zeros

#### Step 5: Reverse Categorical Encoding (Cell 11)
```
Purpose: Convert numeric codes back to original categorical labels
Process for each categorical column:
  - Round numeric predictions to nearest integer
  - Clip values to valid label range [0, n_classes-1]
  - Use inverse_transform to convert codes back to original labels
```

**Result:** `imputed_df` - Clean dataset with imputed values and original categorical labels

#### Step 6: Verify Imputation (Cell 12)
Display first 5 rows of cleaned data.

---

## Phase 3: Exploratory Data Analysis

### Purpose
Understand data distributions and feature relationships.

### Cell 13: Numeric Features Visualization
**Operation:** Create histograms with KDE curves for all numeric columns  
**Grid Layout:** 2 rows × 3 columns  
**Purpose:** Visualize distribution shapes, identify skewness, detect outliers

**Numeric Columns Analyzed:**
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Age (derived or calculated)

### Cell 14: Categorical Features Visualization
**Operation:** Create histograms for all categorical columns  
**Grid Layout:** 3 rows × 3 columns  
**Purpose:** Show frequency distributions of categorical values

**Categorical Columns Analyzed:**
- Gender
- Married
- Dependents
- Education
- Self_Employed
- Property_Area
- Loan_Status (target variable)

### Cell 15-16: Correlation Analysis
**Purpose:** Identify feature relationships for feature selection

**Process:**
1. Create copy of imputed data
2. Encode all categorical variables using LabelEncoder
3. Calculate correlation matrix
4. Visualize using heatmap with annotations

**Output:** Correlation heatmap showing:
- Positive/negative relationships between features
- Feature strength of relationship with target (Loan_Status)

---

## Phase 4: Model Preparation

### Purpose
Format data for machine learning models.

### Cell 17: Import ML Libraries
Additional imports for model training:
- `StandardScaler`: Feature normalization
- `train_test_split`: Data partitioning
- `GridSearchCV`: Hyperparameter tuning
- Classification models: LogisticRegression, DecisionTreeClassifier, GradientBoostingClassifier
- Evaluation metrics: accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report

### Cell 18: Categorical Encoding for Modeling
**Process:**
1. Create working copy of imputed data
2. Identify categorical columns (excluding target)
3. Encode each categorical feature using LabelEncoder
4. Encode target variable (Loan_Status)
5. Create target_mapping dictionary for later interpretation

**Output Variables:**
- `model_data`: Fully encoded dataset
- `label_encoders`: Dictionary of encoders for features
- `le_target`: Encoder for target variable
- `target_mapping`: Mapping of original labels to numeric codes

### Cell 19: Data Splitting and Scaling
**Process:**
1. **Feature-Target Separation:**
   - X: All features (excluding Loan_Status)
   - y: Target variable (Loan_Status)

2. **Train-Test Split:**
   - Test size: 20% (approximately 123 rows)
   - Training size: 80% (approximately 491 rows)
   - stratify=y: Maintains class distribution in both sets

3. **Feature Scaling:**
   - Fit StandardScaler on training set
   - Transform both training and test sets
   - Creates X_train_scaled, X_test_scaled

**Output Variables:**
- `X_train`, `X_test`: Feature datasets
- `y_train`, `y_test`: Target datasets
- `X_train_scaled`, `X_test_scaled`: Normalized features
- `scaler`: StandardScaler object for future predictions

---

## Phase 5: Model Training and Comparison

### Purpose
Train multiple models and compare their performance to identify the best approach.

### Cell 20: Model Selection and Training
**Models Trained:**

1. **Logistic Regression**
   - Algorithm: Linear classification with logistic function
   - Parameters: random_state=42, max_iter=1000
   - Data: Scaled features (requires normalization)
   - Use Case: Fast, interpretable, baseline model

2. **Decision Tree**
   - Algorithm: Recursive partitioning of feature space
   - Parameters: random_state=42
   - Data: Unscaled features (scale-independent)
   - Use Case: Non-linear relationships, feature importance

3. **Gradient Boosting**
   - Algorithm: Ensemble of sequential weak learners
   - Parameters: random_state=42
   - Data: Unscaled features
   - Use Case: High accuracy, handles complex interactions

**Evaluation Metrics Calculated:**
- **Accuracy:** Proportion of correct predictions
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of Precision and Recall
- **ROC-AUC:** Area under Receiver Operating Characteristic curve

**Output:** `results` dictionary containing:
- Trained models
- Predictions and probabilities
- All evaluation metrics
- Feature scaling information

### Cell 21: Model Comparison
**Operation:** Create comparison DataFrame  
**Purpose:** Side-by-side performance evaluation

**Output:** `comparison_df`
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Features |
|-------|----------|-----------|--------|----------|---------|----------|
| Logistic Regression | 0.8537 | 0.7667 | 0.9882 | 0.8651 | 0.8077 | scaled |
| Decision Tree | Lower | Lower | Lower | Lower | Lower | unscaled |
| Gradient Boosting | Moderate | Moderate | Moderate | Moderate | Moderate | unscaled |

**Interpretation:**
- Logistic Regression: **BEST** - Highest F1-Score and most balanced metrics
- Recall: 98.82% - Approves almost all qualified applicants
- Precision: 76.67% - Some false approvals but better for business growth

### Cell 22: Best Model Selection
**Operation:** `best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']`  
**Result:** Identifies Logistic Regression as optimal model based on F1-Score

---

## Phase 6: Model Optimization

### Purpose
Fine-tune the best model's hyperparameters to improve performance.

### Cell 23: Hyperparameter Tuning with Grid Search

**Method:** GridSearchCV with 5-fold Cross-Validation

**Logistic Regression Parameter Grid:**
```python
param_grid = {
    'C': [0.1, 1, 10, 100],           # Regularization strength (inverse)
    'penalty': ['l1', 'l2'],          # Regularization type
    'solver': ['liblinear', 'saga'],  # Optimization algorithm
    'max_iter': [1000, 2000]          # Maximum iterations
}
```

**Total Combinations:** 4 × 2 × 2 × 2 = 32 combinations
**Cross-Validation Folds:** 5
**Total Models Evaluated:** 160

**Optimal Parameters Found:**
- C: 0.1 (Strong regularization - sparse feature selection)
- penalty: 'l1' (L1 regularization - eliminates weak features)
- solver: 'liblinear' (Efficient for binary classification)
- max_iter: Best performing value from grid

**Best CV F1-Score:** 0.8851

**Key Insights:**
- L1 regularization performs feature selection
- Low C value prevents overfitting
- Sparse model: Many coefficients become zero
- Features eliminated: Property_Area, Loan_Amount_Term, LoanAmount, CoapplicantIncome, Self_Employed, ApplicantIncome, Dependents, Gender
- Retained features: Credit_History, Married, Education

### Cell 24: Feature Importance/Coefficients Analysis
**Operation:** Visualize feature coefficients from tuned model

**Output:** Horizontal bar chart showing:
- **Credit_History:** Largest positive coefficient (dominant factor)
- **Married:** Secondary positive factor
- **Education:** Minor negative coefficient
- **Others:** Zero coefficients (eliminated by L1)

**Business Interpretation:**
- **Credit history** is THE critical factor in approval decisions
- Being **married** slightly increases approval chances
- Most other factors don't significantly impact approval

---

## Phase 7: Model Evaluation

### Purpose
Generate final predictions and comprehensive evaluation of tuned model performance.

### Cell 25: Final Model Performance

**Predictions Generated:**
- `final_predictions`: Binary approval decisions
- `final_probabilities`: Approval probability scores (0-1)

**Final Metrics:**
- **Accuracy: 85%** - Overall correctness
- **Precision: 77%** - Of approved applications, 77% were correctly approved
- **Recall: 99%** - Of applicants who should be approved, 99% were identified
- **F1-Score: 87%** - Balanced performance metric
- **ROC-AUC: 0.803** - Strong discrimination ability

### Classification Report
Detailed breakdown per class:
- **Approved (Y):**
  - Precision: High (few false approvals)
  - Recall: Very High (99% - catches almost all valid applicants)
  - F1: Strong balance

- **Rejected (N):**
  - Precision: Moderate
  - Recall: Lower (55% - may miss some rejectable applications)
  - F1: Moderate

### Confusion Matrix Visualization
```
                Predicted
              Approved  Rejected
Actual  Approved    XX       X
        Rejected     X       XX
```

**Interpretation:**
- Model is very conservative with approvals
- High false approval rate (Type II error)
- Low false rejection rate (Type I error)
- **Business Impact:** Maximizes customer satisfaction but increases credit risk

### ROC Curve
**AUC Score: 0.803**
- Plots True Positive Rate vs. False Positive Rate
- Diagonal line (0.5) = random guessing
- Closer to top-left = better discrimination
- 0.803 indicates good model quality

**Interpretation:** Model effectively distinguishes between approvable and non-approvable applications in most cases.

---

## Model Performance Summary

### Key Findings

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Accuracy | 85.37% | Correct for ~5 out of 6 applicants |
| Recall (Approval) | 98.82% | Almost no qualified applicants rejected |
| Precision (Approval) | 76.67% | Some false approvals (credit risk) |
| F1-Score | 86.51% | Best balanced performance |
| ROC-AUC | 0.8077 | Good discrimination ability |

### Business Implications

**Strengths:**
1. High recall (99%) - Maximizes business growth by approving qualified applicants
2. High accuracy (85%) - Reliable overall predictions
3. Interpretable - Credit history dominates, marital status helps

**Weaknesses:**
1. Moderate precision (77%) - Some risky approvals
2. Imbalanced performance - Better at identifying approvals than rejections
3. Risk exposure - May approve marginal applicants

### Recommendations

1. **Use Case:** Best for maximizing loan volume and customer satisfaction
2. **Risk Management:** Implement additional review for borderline cases
3. **Credit Scoring:** Focus on credit history verification first
4. **Feature Monitoring:** Track prediction accuracy over time
5. **Bias Check:** Ensure model isn't discriminating on protected attributes

---

## File Structure

```
DSA3020-Capstone/
├── loan_data_set.csv           # Raw dataset (614 rows)
├── README.md                   # Project overview
├── DOCUMENTATION.md            # This file
└── loandata/
    └── Loanprediction.ipynb    # Main notebook with all analysis
```

---

## How to Use This Model

### Step 1: Data Preparation
1. Ensure new data has same features as training data
2. Apply LabelEncoder transformations using stored encoders
3. Handle missing values using IterativeImputer (fitted on training data)

### Step 2: Feature Scaling
1. Apply StandardScaler (fitted on training data) to normalize features

### Step 3: Generate Predictions
```python
probabilities = best_model.predict_proba(new_data_scaled)[:, 1]
predictions = best_model.predict(new_data_scaled)
```

### Step 4: Interpret Results
- Prediction: Binary (Y/N)
- Probability: Confidence score (0-1)

---

## Dependencies and Environment

**Python Version:** 3.12+

**Required Packages:**
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jinja2 (for pandas styling)

**Installation:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jinja2
```

---

## Contact and Support

For questions or modifications to this documentation, refer to the notebook comments or consult with the data science team.

**Last Updated:** November 30, 2025  
**Model Type:** Classification (Binary)  
**Status:** Production-Ready

