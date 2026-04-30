# Credit Card Fraud Detection

This project analyzes a dataset of anonymized credit card transactions to detect fraudulent activity using exploratory data analysis, feature evaluation, and machine learning classification models.

## The Problem

The dataset is highly imbalanced, with only **0.17% of transactions being fraudulent** (492 out of 284,807). This required special attention at every stage of the project, from how visualizations were constructed to how models were evaluated. Standard accuracy metrics are meaningless at this imbalance level; a model that predicts "legitimate" for every transaction would score 99.83% accuracy while catching zero fraud cases.

## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions over a 48-hour period
- **Features:** 28 PCA-transformed components (V1 through V28), transaction Amount, and Time
- **Target:** Class (0 = legitimate, 1 = fraudulent)

## Project Structure

```
01_data_overview.ipynb    # Initial data validation and class imbalance findings
02_EDA.ipynb              # Exploratory data analysis and feature evaluation
03_preprocessing.ipynb    # Feature scaling and train/test split
04_models.ipynb           # Model training, evaluation, and comparison
fraud.db                  # SQLite database storing all data and results
```

## Key Findings

**Class imbalance is severe.** At a 0.17% fraud rate, naive models are heavily biased toward predicting legitimate transactions. Precision, recall, and AUC-PR were used as evaluation criteria instead of accuracy.

**No single feature cleanly separates the two classes.** While Cohen's d effect size analysis identified components with meaningful discriminating power (V14, V4, V11), all features showed significant overlap between classes. Legitimate transactions cluster tightly and symmetrically around zero; fraudulent transactions are dispersed and multimodal, suggesting fraud does not follow a single behavioral pattern.

**Fraud exhibits a cyclic daily pattern.** Fraudulent activity spiked consistently during low-transaction-volume windows across both days in the dataset, suggesting deliberate targeting of periods with reduced monitoring.

**Two distinct fraud behaviors are visible in the amount distribution.** A high density of very small fraudulent transactions near $0 indicates card testing behavior, where fraudsters charge small amounts to verify cards are active before making larger purchases. This explains why fraudulent transactions have a higher average amount ($122) than legitimate ones ($88) despite a lower median ($9.25 vs $22.00).

## Methodology

### Preprocessing
- `Amount` and `Time` scaled using `StandardScaler`, as the only features not already transformed by PCA
- Stratified train/test split (80/20) to preserve the 0.17% fraud rate in both sets
- `class_weight='balanced'` applied to both models to compensate for class imbalance during training

### Models
Two models were trained and compared:

| Model | Recall | Precision | F1 | AUC-PR |
|---|---|---|---|---|
| Logistic Regression | 0.918 | 0.061 | 0.114 | 0.716 |
| Random Forest (tuned) | 0.847 | 0.922 | 0.883 | 0.854 |

**Logistic Regression** was trained as an interpretable baseline. It achieved strong recall but generated significant false positives, flagging 1,386 legitimate transactions as fraudulent in the test set.

**Random Forest** was selected as the primary model given the high overlap between classes in feature space, which a linear model cannot adequately capture. The decision threshold was tuned to 0.3 (from the default 0.5) based on precision-recall tradeoff analysis. Below 0.3, false positives increased sharply with minimal recall gains. The tuned threshold recovered 10 additional fraud cases at the cost of only 4 additional false positives.

### Evaluation Metrics
Given the severe class imbalance, the following metrics were prioritized:
- **Recall:** of all actual fraud cases, how many were caught
- **Precision:** of all flagged transactions, how many were actually fraud
- **F1 Score:** harmonic mean of precision and recall
- **AUC-PR:** precision-recall curve area, the most informative metric at this imbalance level

## Results

The tuned Random Forest is the recommended model. It catches 84.7% of fraudulent transactions while maintaining 92.2% precision, meaning fewer than 1 in 10 flagged transactions is a false alarm. In a fraud detection context, minimizing missed fraud cases carries higher business priority than minimizing false alarms, making the precision-recall tradeoff at threshold 0.3 acceptable.

## Tableau Dashboard

An interactive dashboard summarizing the EDA findings and model comparison is available here:

[Credit Card Fraud Detection Dashboard](https://public.tableau.com/app/profile/alexander.bolton/viz/CreditCardFraudDetection_17775077780780/Dashboard1)

## Tools and Libraries

- **Python:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **SQLite:** data storage and retrieval across notebooks
- **Tableau Public:** interactive dashboard and presentation layer

## Business Insights

1. Fraud is extremely rare (0.17%), making this a highly imbalanced classification problem where accuracy is a misleading metric.
2. Fraudulent transactions exhibit two distinct behavioral patterns, small card testing charges and larger exploitation purchases, visible in the amount distribution.
3. Fraud activity peaks during low-transaction-volume windows, suggesting deliberate targeting of periods with reduced monitoring.
4. No single feature is sufficient for fraud detection; effective classification requires combining multiple features in a non-linear multivariate model.
5. Detection cannot rely on simple thresholds alone. The Random Forest with a tuned decision threshold outperforms the linear baseline significantly on F1 and precision while maintaining strong recall.
