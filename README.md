# Nedbank Transaction Volume Forecasting Challenge

## Project Overview

This project predicts the total number of bank transactions each customer will make during a 3-month prediction window (**November 2015 – January 2016**). The challenge leverages comprehensive customer transaction history, financial snapshots, and demographic data to build a robust forecasting model.

**Target Variable:** `next_3m_txn_count` (non-negative integer)  
**Evaluation Metric:** RMSLE (Root Mean Squared Logarithmic Error)  
**Customer Base:** 11,944 unique customers across South Africa

---

## 📊 Dataset Overview

### File Structure & Sizes

| File | Type | Rows | Description |
|------|------|------|-------------|
| **Train.csv** | CSV | 8,360 | Training labels (customer IDs + target transaction counts) |
| **Test.csv** | CSV | 3,584 | Test customer IDs for which to predict |
| **SampleSubmission.csv** | CSV | 3,584 | Submission template |
| **transactions_features.parquet** | Parquet | 18M | Historical transaction records (Dec 2012 – Oct 2015) |
| **financials_features.parquet** | Parquet | 372K | Financial snapshots (Dec 2013 – Oct 2015) |
| **demographics_clean.parquet** | Parquet | 11,944 | Customer demographic profiles (1 row per customer) |
| **VariableDefinitions.csv** | CSV | - | Data dictionary for all columns |

---

## 📋 Data Files Detailed Description

### 1. **Train.csv** – Training Labels
Contains 8,360 customer IDs with their actual transaction counts during the prediction window.

| Column | Type | Description |
|--------|------|-------------|
| `UniqueID` | string (UUID) | Unique customer identifier |
| `next_3m_txn_count` | integer | **Target variable**: Total transactions in Nov 2015 – Jan 2016 |

**Key Stats:**
- Ranges from 0 to several hundred transactions per customer
- Distribution is likely skewed (many low-activity customers, fewer high-activity)
- No missing values

---

### 2. **Test.csv** – Test Customer IDs
Contains 3,584 customer IDs for which predictions must be made.

| Column | Type | Description |
|--------|------|-------------|
| `UniqueID` | string (UUID) | Unique customer identifier |

**Note:** These customers do NOT appear in Train.csv; you must join with features data to build predictions.

---

### 3. **SampleSubmission.csv** – Submission Template
Shows the expected format for your predictions.

| Column | Type | Description |
|--------|------|-------------|
| `UniqueID` | string (UUID) | Must match Test.csv exactly |
| `next_3m_txn_count` | float | Your predicted transaction count |

---

### 4. **transactions_features.parquet** – Transaction History
18 million individual transaction records spanning **3 years** (Dec 2012 – Oct 2015). This is the richest data source.

| Column | Type | Description |
|--------|------|-------------|
| `UniqueID` | string | Customer identifier (join key) |
| `AccountID` | string | Bank account identifier — *a customer may have multiple accounts* |
| `TransactionDate` | timestamp | Date transaction occurred (all BEFORE prediction window) |
| `TransactionAmount` | float | Signed amount in ZAR (negative = debit/outflow, positive = credit/inflow) |
| `TransactionTypeDescription` | string | Category of transaction (13 types: Transfers & Payments, Charges & Fees, etc.) |
| `TransactionBatchDescription` | string | Batch processing category (8 types) |
| `StatementBalance` | float | Running account balance after this transaction |
| `IsDebitCredit` | string | `D` = Debit (out), `C` = Credit (in) |
| `ReversalTypeDescription` | string | Type of reversal if transaction was reversed |

**Key Insights:**
- **Temporal coverage:** Full 3 years of history allows trend detection and seasonality analysis
- **Multiple accounts:** Aggregate by customer to capture total activity
- **Signed amounts:** Important for cash flow analysis; need to handle debit/credit semantics
- **Reversals:** Track reversed transactions separately if needed
- **Balance tracking:** Can derive account stability/volatility features

**Feature Engineering Ideas:**
- Transaction frequency (per month, per day of week)
- Average transaction amount and volatility
- Debit vs. credit ratio
- Seasonal patterns (holiday spending, salary deposits)
- Account-level aggregations (total value, count)

---

### 5. **financials_features.parquet** – Financial Snapshots
372K financial records capturing account snapshots at various dates (Dec 2013 – Oct 2015).

| Column | Type | Description |
|--------|------|-------------|
| `UniqueID` | string | Customer identifier (join key) |
| `AccountID` | string | Account identifier (14.4% null; 100% null for Mortgages) |
| `RunDate` | timestamp | Snapshot date for this record |
| `Product` | string | Product type: *Transactional*, *Investments*, or *Mortgages* |
| `NetInterestIncome` | float | Net interest earned on this account |
| `NetInterestRevenue` | float | Net interest revenue (NII + fee adjustments) |

**Key Insights:**
- **Lower temporal coverage** than transactions (only 2 years vs. 3)
- **Product diversity:** Customers may hold multiple product types
- **Sparse data:** 14.4% missing AccountIDs; 100% null for Mortgages
- **Join challenge:** Must use LEFT JOIN on financials to avoid losing customers without financial data
- **Account-level aggregation:** Summarize by customer or product type

**Data Quality Notes:**
- Handle null AccountIDs carefully (join via UniqueID for Mortgages)
- Products may have different feature distributions

---

### 6. **demographics_clean.parquet** – Customer Profiles
11,944 unique customer records with one row per customer. **Clean data** with minimal missing values.

| Column | Type | Missing % | Description |
|--------|------|-----------|-------------|
| `UniqueID` | string | 0% | Customer identifier (join key) |
| `BirthDate` | timestamp | ~0% | Date of birth (**data quality varies** — inspect before using) |
| `Gender` | string | ~0% | `M` or `F` (cleaned for whitespace) |
| `IncomeCategory` | string | ~0% | Low, Lower-Middle, Middle, Upper-Middle, High, Very High, No Income, Not Disclosed |
| `CustomerStatus` | string | ~0% | Customer lifecycle status |
| `ClientType` | string | ~0% | Individual – Adult, Foreign Individual, etc. |
| `MaritalStatus` | string | ~0% | Single, Married, Divorced, Widowed |
| `OccupationCategory` | string | ~0% | 19 occupation groupings |
| `IndustryCategory` | string | ~0% | 18 industry groupings |
| `CustomerBankingType` | string | 3.4% | Primary/Secondary/Main banking relationship |
| `CustomerOnboardingChannel` | string | <0.1% | How customer was onboarded |
| `ResidentialCityName` | string | ~0% | Customer residential city |
| `CountryCodeNationality` | string | ~0% | ISO country code of nationality |
| `AnnualGrossIncome` | float | 6.2% | Annual gross income in ZAR |
| `LowIncomeFlag` | string | ~0% | Low income classification flag |
| `CertificationTypeDescription` | string | ~0% | ID document type (RSA ID, Passport, etc.) |
| `ContactPreference` | string | ~0% | Contact consent preference |

**Key Insights:**
- **Demographic stability:** These profiles are snapshots; use with caution if customer status changed during observation period
- **Income quality:** 6.2% missing AnnualGrossIncome; IncomeCategory may be more reliable for segmentation
- **Geographic dimension:** Valuable for regional analysis and potential location-based seasonality
- **Employment signals:** Occupation and Industry provide business context

**Feature Engineering Ideas:**
- Age (derived from BirthDate, but handle quality issues)
- Segment by income category, occupation, or industry
- Geographic encoding (city/region dummies)
- Client type classifications (individual vs. other)

---

### 7. **VariableDefinitions.csv** – Data Dictionary
Complete reference of all columns, types, and descriptions (provided as reference).

---

## 🔗 Join Strategy

All files use **UniqueID** as the primary key. Here's the recommended approach:

```python
import pandas as pd

# Load base data
train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')

# Load feature tables
transactions = pd.read_parquet('data/transactions_features.parquet')
financials = pd.read_parquet('data/financials_features.parquet')
demographics = pd.read_parquet('data/demographics_clean.parquet')

# Aggregate transactions by customer
txn_features = transactions.groupby('UniqueID').agg({
    'TransactionAmount': ['sum', 'mean', 'std', 'count'],
    'TransactionDate': ['min', 'max'],
    # Add more as needed
}).reset_index()

# LEFT JOIN to preserve all customers
train_enriched = (train
    .merge(txn_features, on='UniqueID', how='left')
    .merge(financials, on='UniqueID', how='left')  # 567 customers missing
    .merge(demographics, on='UniqueID', how='left')
)
```

**Important:** Some customers lack financial data (567 of 11,944). Use LEFT JOINs to preserve them.

---

## ⚠️ Data Quality Notes

1. **BirthDate Quality:** Data varies in quality. Consider:
   - Calculating age and checking for unrealistic values (< 18 or > 120)
   - May have data entry errors
   - Consider age categories instead of raw age

2. **AnnualGrossIncome:** 6.2% null (~735 customers)
   - Use IncomeCategory as alternative
   - Consider imputation or separate "unknown" category

3. **Multiple Accounts:** Customers can have multiple AccountIDs
   - Aggregate transaction and financial data by customer
   - May want account-level features separately

4. **Missing AccountID in Financials:** 14.4% overall; 100% for Mortgages
   - Join Mortgages on UniqueID only
   - Handle with conditional logic in feature engineering

5. **Signed Amounts:** TransactionAmount is signed
   - Negative = debit (money out)
   - Positive = credit (money in)
   - Reversals may create additional complexity

6. **Temporal Mismatch:** 
   - Transactions: Dec 2012 – Oct 2015 (3 years)
   - Financials: Dec 2013 – Oct 2015 (2 years)
   - Demographics: Current snapshot (not time-varying)

7. **Seasonality:** Prediction window includes holiday season (Nov-Jan)
   - November-December: Holiday spending
   - January: Post-holiday adjustments
   - Consider interaction with income category

---

## 🚀 Quick Start

### 1. Load and Explore
```python
import pandas as pd
import numpy as np

# Load training data
train = pd.read_csv('data/Train.csv')
print(f"Training samples: {len(train)}")
print(f"Target range: {train['next_3m_txn_count'].min()} - {train['next_3m_txn_count'].max()}")
print(f"Target distribution:\n{train['next_3m_txn_count'].describe()}")

# Check for skewness
print(f"Skewness: {train['next_3m_txn_count'].skew():.2f}")
```

### 2. Join Features
```python
# Load all feature data (see Join Strategy above)
# Build training dataset with features
train_features = train.merge(txn_features, on='UniqueID', how='left')
```

### 3. Feature Engineering
- Count transactions per month, day of week
- Average/std transaction amounts
- Debit/credit ratios
- Income-based segmentation
- Geographic features
- Seasonality indicators

### 4. Model & Predict
```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'UniqueID': test['UniqueID'],
    'next_3m_txn_count': y_pred
})
submission.to_csv('submission.csv', index=False)
```

### 5. Local Evaluation
```bash
python evaluate.py submission.csv data/PublicReference.csv
```

---

## 📈 Expected Challenges

- **Class Imbalance:** Many low-activity customers skew distribution
- **Missing Data:** Handle gracefully with left joins
- **Feature Engineering:** Temporal aggregation from high-volume transactions
- **RMSLE Metric:** Log errors penalize under-prediction; important for sparse predictions
- **Leakage Risk:** Ensure all features use data strictly BEFORE Oct 2015

---

## 📝 Submission Requirements

1. Format: CSV with columns `UniqueID`, `next_3m_txn_count`
2. UniqueIDs must match Test.csv exactly (3,584 customers)
3. Predictions should be non-negative integers (or floats)
4. Register at Nedbank for eligibility and prizes

---

## 🔗 Resources

- **Data Dictionary:** See `VariableDefinitions.csv`
- **Baseline Model:** See `StarterNotebook.ipynb`
- **Local Scoring:** `python evaluate.py <submission> <reference>`
- **Submission:** Upload via Zindi platform

---

## 📌 Key Takeaways

| Aspect | Value |
|--------|-------|
| Total Customers | 11,944 |
| Training Samples | 8,360 |
| Test Samples | 3,584 |
| Transaction Records | 18M |
| Temporal Window | Dec 2012 – Oct 2015 |
| Prediction Window | Nov 2015 – Jan 2016 |
| Primary Join Key | UniqueID |
| Evaluation Metric | RMSLE |

