# Mathematica Notebook Assistant Prompt Pack

This document is a copy-paste prompt kit for using Mathematica locally on this
repo. It is meant to help you get started quickly inside Wolfram Notebook
Assistant without letting the assistant invent paths, columns, or a whole
production pipeline.

Use Mathematica for exploration, visual diagnostics, and modelling intuition.
Keep the final repeatable pipeline in Python.

```text
Mathematica AI = exploration + mathematical modelling + visual diagnostics
ChatGPT/Codex  = planning + prompt design + code review + Python translation
Python repo    = final repeatable pipeline, validation, submissions
```

## Start Here

Open:

```text
H:\hikrepos\bank-transaction-volume-forecast\notebooks\local_data_access.nb
```

or open:

```text
H:\hikrepos\bank-transaction-volume-forecast\notebooks\main_mathematica.nb
```

Then run these Wolfram Language cells first:

```wolfram
Get[FileNameJoin[{NotebookDirectory[], "mathematica_data_setup.wl"}]]
VerifyBankForecastFiles[]
SmokeTestBankForecastImports[]
```

Expected result:

```text
TrainLabels imports with 8360 rows.
Demographics imports with 11944 rows.
All required files exist.
SampleSubmission.csv may be missing and is optional.
```

Do not paste modelling prompts until those checks pass.

## Local Files And Columns

The local project root is:

```text
H:\hikrepos\bank-transaction-volume-forecast
```

Input files are under:

```text
H:\hikrepos\bank-transaction-volume-forecast\data\inputs
```

Use these exact files:

```text
transactions_features.parquet
financials_features.parquet
demographics_clean.parquet
Train.csv
Test.csv
VariableDefinitions.csv
```

`SampleSubmission.csv` is optional and is not currently required for notebook
exploration.

Known columns:

```text
Train.csv:
- UniqueID
- next_3m_txn_count

Test.csv:
- UniqueID

transactions_features.parquet:
- UniqueID
- AccountID
- TransactionDate
- TransactionAmount
- TransactionTypeDescription
- TransactionBatchDescription
- StatementBalance
- IsDebitCredit
- ReversalTypeDescription

financials_features.parquet:
- UniqueID
- AccountID
- RunDate
- Product
- NetInterestIncome
- NetInterestRevenue

demographics_clean.parquet:
- UniqueID
- BirthDate
- Gender
- IncomeCategory
- CustomerStatus
- ClientType
- MaritalStatus
- OccupationCategory
- IndustryCategory
- CustomerBankingType
- CustomerOnboardingChannel
- ResidentialCityName
- CountryCodeNationality
- AnnualGrossIncome
- LowIncomeFlag
- CertificationTypeDescription
- ContactPreference
```

Prediction target:

```text
next_3m_txn_count = customer transaction count for November 2015 through January 2016
```

Main leakage cutoff:

```text
2015-10-31
```

## Rules For Every Prompt

Paste this at the end of any prompt if Notebook Assistant starts drifting:

```text
Use only the local files and columns listed in this prompt. Do not invent new
columns. Do not ask me to upload files. Use Wolfram Language code that can run
inside this local notebook. Keep this section focused and do not build the full
project.
```

## Prompt 0: Notebook Outline

Copy and paste this into Notebook Assistant after the startup cells pass:

```text
I am working locally in Mathematica on this project:

Project root:
H:\hikrepos\bank-transaction-volume-forecast

Notebook folder:
H:\hikrepos\bank-transaction-volume-forecast\notebooks

Data folder:
H:\hikrepos\bank-transaction-volume-forecast\data\inputs

I have already run:

Get[FileNameJoin[{NotebookDirectory[], "mathematica_data_setup.wl"}]]
VerifyBankForecastFiles[]
SmokeTestBankForecastImports[]

The project goal is to predict `next_3m_txn_count`, the total number of bank
transactions each customer will make during November 2015 through January 2016.
The cutoff date for historical features is 2015-10-31.

Use this notebook as a research cockpit only. The final repeatable pipeline will
stay in Python.

Please create a compact notebook outline with sections for:
1. local data access and schema inspection
2. transaction volume aggregation
3. missingness and data quality checks
4. seasonality diagnostics
5. customer behaviour segmentation
6. aggregate numerical forecasting baselines
7. customer-level feature ideas before the 2015-10-31 cutoff
8. anomaly and pattern detection
9. sequence-pattern exploration
10. export of findings to CSV and markdown

For each section, give:
- a short purpose
- the expected input variables
- the expected output variables
- one or two validation checks
- what should be exported or translated to Python later

Do not write all code yet. Only create the outline.
```

## Prompt 1: Import And Schema Inspection

Copy and paste:

```text
Create notebook section 1: local data import and schema inspection.

Use the local helper functions that are already loaded from:
H:\hikrepos\bank-transaction-volume-forecast\notebooks\mathematica_data_setup.wl

Use these helper functions:
- VerifyBankForecastFiles[]
- ImportBankForecastFile[name]
- LoadBankForecastData[names]

Use these exact local logical table names:
- "Transactions" for transactions_features.parquet
- "Financials" for financials_features.parquet
- "Demographics" for demographics_clean.parquet
- "TrainLabels" for Train.csv
- "TestCustomers" for Test.csv
- "VariableDefinitions" for VariableDefinitions.csv

Known columns:

Transactions:
UniqueID, AccountID, TransactionDate, TransactionAmount,
TransactionTypeDescription, TransactionBatchDescription, StatementBalance,
IsDebitCredit, ReversalTypeDescription

Financials:
UniqueID, AccountID, RunDate, Product, NetInterestIncome, NetInterestRevenue

Demographics:
UniqueID, BirthDate, Gender, IncomeCategory, CustomerStatus, ClientType,
MaritalStatus, OccupationCategory, IndustryCategory, CustomerBankingType,
CustomerOnboardingChannel, ResidentialCityName, CountryCodeNationality,
AnnualGrossIncome, LowIncomeFlag, CertificationTypeDescription,
ContactPreference

TrainLabels:
UniqueID, next_3m_txn_count

TestCustomers:
UniqueID

Write Wolfram Language code that:
1. Runs VerifyBankForecastFiles[].
2. Imports TrainLabels, TestCustomers, VariableDefinitions, Demographics,
   Financials, and Transactions into variables named:
   trainLabels, testCustomers, variableDefinitions, demographics, financials,
   transactions.
3. If loading the full Transactions table is slow, show how to first create a
   sample named transactionSample.
4. Prints row counts, column names, inferred simple types, and five sample rows
   for each table.
5. Checks duplicate UniqueID values in TrainLabels, TestCustomers, and
   Demographics.
6. Checks duplicate AccountID values only in Transactions and Financials.
7. Converts TransactionDate, RunDate, and BirthDate to DateObject values if
   needed.
8. Creates a Dataset named SchemaSummary with columns:
   Table, Rows, Columns, DateColumns, IDColumns, NumericColumns,
   CategoricalColumns, MissingnessNotes.

Validation checks:
- Confirm TrainLabels has 8360 rows.
- Confirm Demographics has 11944 rows.
- Confirm UniqueID exists in every imported table.
- Confirm next_3m_txn_count exists only in TrainLabels.

End with a short interpretation of any schema risks for Python modelling.

Do not continue to aggregation or modelling.
```

## Prompt 2: Transaction Volume Aggregation

Copy and paste after Prompt 1 has created `transactions`:

```text
Create notebook section 2: transaction volume aggregation.

Use the existing variable `transactions`.

Required transaction columns:
- UniqueID
- AccountID
- TransactionDate
- TransactionAmount
- IsDebitCredit

Optional transaction columns:
- TransactionTypeDescription
- TransactionBatchDescription
- StatementBalance
- ReversalTypeDescription

Write Wolfram Language code that:
1. Validates that UniqueID and TransactionDate exist before doing any grouping.
2. Converts TransactionDate to DateObject if needed.
3. Creates a month key named TransactionMonth using the first day of each month.
4. Creates a Dataset named CustomerMonthlyVolume grouped by UniqueID and
   TransactionMonth.
5. For each customer-month, computes:
   - txn_count
   - total_amount
   - avg_amount
   - median_amount
   - min_amount
   - max_amount
   - debit_count using IsDebitCredit == "D"
   - credit_count using IsDebitCredit == "C"
   - debit_share
   - credit_share
6. Creates a Dataset named PortfolioMonthlyVolume grouped by TransactionMonth
   with:
   - total_txn_count
   - active_customers
   - total_amount
   - debit_count
   - credit_count
7. Visualizes total monthly transaction count over time.
8. Visualizes the top 20 customers by total transaction count.
9. Creates a sparse-customer summary with:
   - active month count per customer
   - zero or missing month count over the observed range
   - total transaction count

Validation checks:
- Show the min and max TransactionDate.
- Show the number of distinct UniqueID values in transactions.
- Show whether any TransactionMonth values are after 2015-10-31.
- Confirm all txn_count values are non-negative integers.

End with a small table of Python feature ideas from this section.
```

## Prompt 3: Missingness And Data Quality

Copy and paste:

```text
Create notebook section 3: missingness and data quality checks.

Use these existing variables if available:
- transactions
- financials
- demographics
- trainLabels
- testCustomers

Known important columns:
- UniqueID
- AccountID
- TransactionDate
- TransactionAmount
- StatementBalance
- IsDebitCredit
- RunDate
- Product
- NetInterestIncome
- NetInterestRevenue
- BirthDate
- Gender
- IncomeCategory
- AnnualGrossIncome
- CustomerBankingType
- CustomerOnboardingChannel
- next_3m_txn_count

Write Wolfram Language code that:
1. Builds a Dataset named MissingnessSummary with:
   Table, Column, MissingCount, MissingRate, NonMissingCount.
2. Builds a join coverage table named JoinCoverageSummary showing:
   - Train UniqueIDs found in transactions
   - Test UniqueIDs found in transactions
   - Train UniqueIDs found in financials
   - Test UniqueIDs found in financials
   - Train UniqueIDs found in demographics
   - Test UniqueIDs found in demographics
3. Checks bad or surprising values:
   - missing UniqueID
   - missing TransactionDate
   - missing TransactionAmount
   - IsDebitCredit values other than "D" or "C"
   - negative or missing StatementBalance
   - missing Product in financials
   - missing or implausible BirthDate
   - missing AnnualGrossIncome
4. Visualizes missingness rate by table and column.
5. Visualizes target distribution for next_3m_txn_count using histogram and
   log1p histogram.

Validation checks:
- Confirm Train and Test UniqueID sets do not overlap, or report overlap count.
- Confirm all train labels have non-negative next_3m_txn_count.
- Confirm no target column exists in testCustomers.

End with recommended missingness indicators to create later in Python.
```

## Prompt 4: Seasonality Diagnostics

Copy and paste after Prompt 2 has created `CustomerMonthlyVolume` and
`PortfolioMonthlyVolume`:

```text
Create notebook section 4: seasonality diagnostics for bank transaction volumes.

Use:
- transactions
- CustomerMonthlyVolume
- PortfolioMonthlyVolume

Known date column:
- TransactionDate

Known target window:
- November 2015 through January 2016

Historical cutoff:
- 2015-10-31

Write Wolfram Language code that:
1. Plots aggregate monthly total_txn_count from PortfolioMonthlyVolume.
2. Decomposes the aggregate monthly series into trend and seasonal components
   where possible.
3. Computes average transaction count by calendar month.
4. Uses TransactionDate to compute day-level effects:
   - day_of_week
   - is_weekend
   - day_of_month
   - is_month_start_window for days 1 through 5
   - is_mid_month_window for days 13 through 17
   - is_month_end_window for the last 5 days of the month
   - is_payday_window for days 24 through 31
5. Creates a year-month heatmap of transaction intensity.
6. Creates a day-of-month heatmap or bar chart of transaction intensity.
7. Checks whether December and January behaviour is unusual compared with other
   months.
8. Creates a Dataset named CandidateSeasonalityFeatures with columns:
   FeatureName, Description, RequiredGranularity, WhyItMayHelp, PythonPriority.

Include at least these candidate features:
- month_of_year
- is_december
- is_january
- is_month_start_window
- is_mid_month_window
- is_month_end_window
- is_payday_window
- day_of_week
- is_weekend
- rolling_3_month_count
- rolling_6_month_count
- rolling_12_month_count
- same_month_last_year_count
- month_end_transaction_share
- payday_window_transaction_share
- holiday_period_activity_ratio

Validation checks:
- Confirm no seasonality feature uses dates after 2015-10-31.
- Show date coverage by year and month.
- Show the number of months available for monthly seasonality estimates.

End with a short interpretation focused on features to implement in Python.
```

## Prompt 5: Customer Behaviour Segmentation

Copy and paste:

```text
Create notebook section 5: customer behaviour segmentation.

Use:
- CustomerMonthlyVolume
- transactions if daily/month-end shares are needed
- trainLabels and testCustomers only for coverage checks, not for clustering

Use customer-level history only up to 2015-10-31.

Write Wolfram Language code that:
1. Creates a customer feature Dataset named CustomerBehaviourFeatures with:
   - UniqueID
   - total_txn_count
   - avg_monthly_count
   - median_monthly_count
   - sd_monthly_count
   - coefficient_of_variation
   - active_months
   - inactive_months
   - recent_3m_count
   - previous_3m_count
   - recent_activity_ratio
   - max_monthly_count
   - max_monthly_spike_ratio
   - december_activity_ratio if months exist
   - january_activity_ratio if months exist
   - month_end_transaction_share if daily data exists
   - payday_window_transaction_share if daily data exists
2. Handles sparse customers carefully. Do not divide by zero.
3. Standardizes numerical features.
4. Runs k-means clustering for k = 3 through 8.
5. Runs hierarchical clustering on a sample if the full data is too large.
6. Helps choose a reasonable k using an elbow plot or silhouette-style score if
   available.
7. Visualizes customers in 2D using PCA, dimension reduction, or another built-in
   Wolfram method.
8. Creates a Dataset named CustomerBehaviourSegments with:
   - UniqueID
   - ClusterID
   - SegmentLabel
   - key feature values
9. Creates a Dataset named SegmentProfileSummary with one row per segment.

Suggested plain-English labels:
- dormant
- low stable
- seasonal
- bursty
- high-volume
- declining
- growing

Validation checks:
- Confirm every clustered row has a UniqueID.
- Confirm train/test labels were not used as clustering features.
- Show segment counts.
- Show train/test coverage by segment if trainLabels and testCustomers exist.

End with a table of segment features that should be translated to Python.
```

## Prompt 6: Aggregate Forecasting Baseline

Copy and paste:

```text
Create notebook section 6: aggregate numerical forecasting baseline.

Use only:
- PortfolioMonthlyVolume

Keep this aggregate-level only. Do not train customer-level ML.

Target series:
- total_txn_count by TransactionMonth

Write Wolfram Language code that:
1. Sorts PortfolioMonthlyVolume by TransactionMonth.
2. Creates a TimeSeries or TemporalData object for total_txn_count.
3. Splits the last 3 available historical months as a holdout period.
4. Fits simple baselines:
   - naive last value
   - moving average using 3 months
   - moving average using 6 months
   - exponential smoothing if available
   - ARIMA or TimeSeriesModelFit if appropriate
5. Forecasts the holdout period.
6. Computes RMSE, MAE, and RMSLE, with RMSLE only if counts are non-negative.
7. Plots actual vs predicted holdout values.
8. Plots residuals by month.
9. Creates a Dataset named AggregateForecastBaselineResults with:
   ModelName, RMSE, MAE, RMSLE, Notes.

Validation checks:
- Confirm the holdout period occurs before or at 2015-10-31.
- Confirm forecast horizon equals the number of holdout months.
- Confirm all predictions are non-negative, or clip only for metric reporting
   and clearly state that clipping was used.

End with a short conclusion about whether seasonality appears captured.
```

## Prompt 7: Customer-Level Feature Table

Copy and paste:

```text
Create notebook section 7: customer-level feature table for future transaction
count forecasting.

Use:
- CustomerMonthlyVolume
- transactions
- financials
- demographics
- trainLabels
- testCustomers

Explicit cutoff date:
- 2015-10-31

Prediction target:
- next_3m_txn_count in Train.csv

Important leakage rule:
- Do not use any TransactionDate or RunDate after 2015-10-31.
- Do not use next_3m_txn_count as an input feature.

Write Wolfram Language code that:
1. Defines cutoffDate = DateObject[{2015, 10, 31}].
2. Filters transactions to TransactionDate <= cutoffDate.
3. Filters financials to RunDate <= cutoffDate.
4. Creates a Dataset named CustomerFeatureTable with one row per UniqueID from
   the union of TrainLabels and TestCustomers.
5. Adds transaction history features:
   - txn_count_total
   - txn_count_last_1m
   - txn_count_last_3m
   - txn_count_last_6m
   - txn_count_last_12m
   - txn_count_mean_3m
   - txn_count_mean_6m
   - txn_count_mean_12m
   - txn_count_sd_6m
   - txn_count_max_12m
   - active_months
   - inactive_months
   - months_since_last_transaction
   - recent_3m_vs_prior_3m_ratio
   - same_month_last_year_count if available
6. Adds amount features:
   - amount_sum_total
   - amount_sum_last_3m
   - amount_mean_last_3m
   - debit_share_last_3m
   - credit_share_last_3m
7. Adds time-window behaviour features:
   - month_end_transaction_share
   - payday_window_transaction_share
   - weekend_transaction_share
   - december_activity_ratio
   - january_activity_ratio
8. Adds latest financial features by customer:
   - latest NetInterestIncome
   - latest NetInterestRevenue
   - latest Product counts or shares where practical
   - financial_missing_indicator
9. Adds demographic features:
   - Gender
   - IncomeCategory
   - CustomerStatus
   - ClientType
   - MaritalStatus
   - OccupationCategory
   - IndustryCategory
   - CustomerBankingType
   - AnnualGrossIncome
   - demographic_missing indicators where useful
10. Adds a column named DatasetRole with values "train" or "test".
11. Joins next_3m_txn_count only for train rows.

Validation checks:
- Show max TransactionDate used after filtering.
- Show max RunDate used after filtering.
- Confirm max dates are <= 2015-10-31.
- Confirm CustomerFeatureTable has one row per UniqueID.
- Confirm no test row has next_3m_txn_count filled from train.
- Confirm no feature column is named next_3m_txn_count except the final target
  column for train rows.

End with a compact Python implementation spec for these features.
```

## Prompt 8: Anomaly And Pattern Detection

Copy and paste:

```text
Create notebook section 8: pattern recognition and anomaly detection.

This is for neutral pattern discovery, not fraud accusation.

Use:
- CustomerBehaviourFeatures
- CustomerMonthlyVolume
- transactions if daily details are needed

Write Wolfram Language code that:
1. Detects unusually high-volume customers.
2. Detects customers with sudden monthly spikes.
3. Detects customers with sudden monthly drops.
4. Detects customers whose recent 3-month activity differs from their historical
   behaviour.
5. Uses at least two methods:
   - robust z-score based on median and MAD
   - cluster distance, nearest-neighbour distance, or another built-in anomaly
     score if available
6. Creates a Dataset named AnomalyCandidates with:
   - UniqueID
   - AnomalyScore
   - Reason
   - TotalTxnCount
   - Recent3MCount
   - HistoricalMonthlyMean
   - MaxMonthlyCount
   - SupportingFeatureValues
7. Visualizes the monthly trajectories of the top 10 most unusual customers.
8. Uses neutral labels such as:
   - high volume
   - recent spike
   - recent drop
   - sparse but bursty
   - behaviour shift

Validation checks:
- Confirm every anomaly candidate exists in CustomerMonthlyVolume.
- Confirm scores are finite numeric values.
- Confirm no target label is used in the anomaly score.

End with Python feature ideas such as high_tail_flag, spike_flag,
recent_shift_score, and burstiness_score.
```

## Prompt 9: Sequence-Pattern Exploration

Copy and paste:

```text
Create notebook section 9: interpretable sequence-pattern exploration.

Use:
- CustomerMonthlyVolume

Do not train a deep neural network.

Write Wolfram Language code that:
1. Finds the full observed monthly range.
2. Converts each customer's monthly txn_count history into a fixed-length vector
   ordered by month.
3. Fills missing customer-month values with 0, but also creates an active-month
   mask so missing/inactive behaviour is explicit.
4. Optionally normalizes each sequence by customer mean or total count for shape
   comparison.
5. Computes sequence similarity using Euclidean distance and correlation
   distance, where practical.
6. Clusters customers by sequence shape.
7. Visualizes representative trajectories per cluster.
8. Identifies clusters or examples with these plain-English patterns:
   - periodic
   - growing
   - declining
   - bursty
   - dormant
   - consistently high volume

Create:
- CustomerMonthlySequences
- SequenceClusterAssignments
- SequenceClusterProfiles

Validation checks:
- Confirm all sequences have the same length.
- Confirm month ordering is correct.
- Confirm no dates after 2015-10-31 are used.

End with features that can be translated into Python, such as trend slope,
sequence volatility, zero-run length, burst count, and normalized last-3-month
shape.
```

## Prompt 10: Export Findings

Copy and paste:

```text
Create notebook section 10: export findings.

Use this output folder:
H:\hikrepos\bank-transaction-volume-forecast\outputs\mathematica

Create the folder if it does not exist.

Export any variables that exist. Before exporting each one, check whether it
exists and skip it safely if missing.

Export these tables to CSV when available:
- SchemaSummary
- MissingnessSummary
- JoinCoverageSummary
- CustomerMonthlyVolume
- PortfolioMonthlyVolume
- CandidateSeasonalityFeatures
- CustomerBehaviourFeatures
- CustomerBehaviourSegments
- SegmentProfileSummary
- AggregateForecastBaselineResults
- CustomerFeatureTable
- AnomalyCandidates
- SequenceClusterProfiles

Also create a markdown report named:
H:\hikrepos\bank-transaction-volume-forecast\outputs\mathematica\mathematica_research_report.md

The markdown report should summarize:
1. local files used
2. data shapes
3. date coverage
4. missingness and join coverage
5. seasonality findings
6. customer behaviour segments
7. aggregate forecast baseline results
8. anomaly and sequence-pattern observations
9. candidate Python features
10. recommended next Python implementation tasks

Validation checks:
- Confirm each exported CSV path.
- Confirm the markdown report path.
- Confirm exports do not overwrite input files under data/inputs.
```

## Prompt 11: Convert Findings Into Python Tasks

Copy and paste after the notebook has produced findings:

```text
Based on the Mathematica notebook findings so far, produce a Python
implementation plan for this repo.

Repo root:
H:\hikrepos\bank-transaction-volume-forecast

Existing Python feature modules may include:
- src/features.py
- src/features_rolling.py
- src/features_event_temporal.py
- src/features_seq.py
- src/validation.py

Create a Dataset or table with:
- PythonFile
- FunctionName
- InputColumns
- OutputColumns
- Description
- ValidationChecks
- Priority
- DependsOnNotebookOutput

Focus on:
1. seasonality features
2. rolling transaction-count features
3. amount/debit/credit features
4. customer behaviour segment features
5. high-tail customer flags
6. spike/drop/recent-shift features
7. time-based validation checks
8. leakage checks around the 2015-10-31 cutoff

Do not write Python code yet. Produce an implementation specification that can
be copied into GitHub issues or handed to Codex.
```

## Correction Prompt: Too Broad

Copy and paste if the assistant tries to build too much:

```text
Stop. This section is too large.

Refactor the answer into:
1. one small Wolfram Language function
2. one example call
3. one validation check
4. one visualization if relevant
5. a short explanation

Do not introduce new modelling ideas in this response.
```

## Correction Prompt: Invented Columns

Copy and paste if the assistant assumes columns that do not exist:

```text
You assumed columns that are not in this repo.

Use only these known columns:

Transactions:
UniqueID, AccountID, TransactionDate, TransactionAmount,
TransactionTypeDescription, TransactionBatchDescription, StatementBalance,
IsDebitCredit, ReversalTypeDescription

Financials:
UniqueID, AccountID, RunDate, Product, NetInterestIncome, NetInterestRevenue

Demographics:
UniqueID, BirthDate, Gender, IncomeCategory, CustomerStatus, ClientType,
MaritalStatus, OccupationCategory, IndustryCategory, CustomerBankingType,
CustomerOnboardingChannel, ResidentialCityName, CountryCodeNationality,
AnnualGrossIncome, LowIncomeFlag, CertificationTypeDescription,
ContactPreference

Train:
UniqueID, next_3m_txn_count

Test:
UniqueID

Rewrite the code so it checks whether required columns exist before using them.
If a required column is missing, return a clear message and skip that part
safely.
```

## Correction Prompt: Code Quality

Copy and paste when the generated Wolfram code gets messy:

```text
Refactor this Wolfram Language code for readability and reliability.

Requirements:
- use descriptive variable names
- avoid deeply nested expressions where possible
- add short comments only where useful
- add input validation
- keep outputs as Dataset where practical
- do not change the analytical intent
- do not introduce new columns or files
```

## Correction Prompt: Leakage

Copy and paste if a feature might use future data:

```text
This may leak future information.

Rewrite the section using this explicit cutoff:
cutoffDate = DateObject[{2015, 10, 31}]

Rules:
- filter transactions to TransactionDate <= cutoffDate
- filter financials to RunDate <= cutoffDate
- do not use next_3m_txn_count as an input feature
- do not use November 2015, December 2015, or January 2016 transactions as
  features
- include validation checks that prove max used dates are <= cutoffDate
```
