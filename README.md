
# Loan Approval Analysis & Prediction

This project analyzes a loan approval dataset to uncover patterns in loan decisions and build predictive models using machine learning techniques in R. The goal is to understand the factors influencing loan approvals and predict future outcomes based on applicant information.

## Project Structure

- `Final.script.R`: Main R script for data cleaning, visualization, statistical testing, and machine learning.
- `Report.docx`: Project report detailing the methodology, visualizations, results, and insights.
- `data/loan_approval.csv`: Input dataset used (must be placed in the `data` folder).

## Dataset Description

The dataset contains **4269 observations** and **13 features**, including:

- `loan_id`
- `no_of_dependents`
- `education` (Graduate/Not Graduate)
- `self_employed` (Yes/No)
- `income_annum`
- `loan_amount`
- `loan_term`
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`
- `loan_status` (Approved/Rejected)

### Cleaning Steps
- Removed NAs and duplicates.
- Converted string-formatted numerics (with commas/spaces) to numeric values.
- Verified dataset balance (~62% approved, ~38% rejected).

## Exploratory Data Analysis

- Performed statistical summaries and visualizations for each feature.
- Generated boxplots and density plots to compare training and test distributions.
- Created a correlation matrix to identify strong linear relationships (e.g., income and loan amount: **0.93**).

## Statistical Analysis

- **T-tests** and **ANOVA** identified:
  - **CIBIL score** and **Loan term** as significant predictors.
  - Other variables showed no significant difference across loan statuses.
- **Chi-Square tests** found no significant association between loan approval and:
  - Education level
  - Self-employment status

## Machine Learning Models

Three models were trained and evaluated using 75% training and 25% test split:

| Model              | Accuracy | AUC  |
|-------------------|----------|------|
| Logistic Regression | 91.8%    | 0.970 |
| Decision Tree       | 97.0%    | 0.996 |
| Random Forest       | 97.8%    | 0.996 |

### Highlights:
- Random Forest was the most accurate and robust.
- ROC curves and confusion matrices were used to compare performance.

## Hypothesis Testing

### 1. **CIBIL Score & Loan Approval**
- Two-sample t-test and ANOVA showed a **significant difference** in scores.
- Approved loans had much higher average CIBIL scores.

### 2. **Annual Income & Dependents**
- ANOVA showed **no significant variation** in income across dependent groups.

### 3. **Total Assets & Loan Status**
- No significant link between total assets and loan approval.

## How to Run

1. Place `loan_approval.csv` in a `data/` folder.
2. Open `Final.script.R` in RStudio.
3. Install required packages (if not already installed):

```r
install.packages(c("tidyverse", "caTools", "ggplot2", "pROC", "corrplot", "GGally", "rpart", "randomForest", "caret", "gridExtra"))
```

4. Run the script line by line or source it.

## 📌 Key Takeaways

- CIBIL score and loan term are major drivers in loan approval.
- Education, self-employment, and assets alone do not guarantee approval.
- Random Forest is the best-performing model in this setting.
