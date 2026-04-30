# ChatGPT Research Handover: Path To Sub-0.35 Public Score

## Mission

Research credible ways to move this Zindi bank transaction volume forecast from the current safe public score around `0.3899` toward a public score below `0.35`.

Important wording note: this is an RMSE-style score, so lower is better. The user's phrase "greater than 0.35" means "better than the 0.35 range"; treat the target as `public_score <= 0.35`.

## Current Competition Framing

- Task: predict each test customer's transaction count over the next 3 months.
- Upload column: `next_3m_txn_count`.
- Critical discovery: Zindi expects submitted values in `log1p(count)` space, not raw counts.
- Hard validation rule: any submission with max/mean suggesting raw counts must be rejected before upload.
- Train rows: `8360` unique IDs.
- Test rows: `3584` unique IDs.
- Prediction window for the main task: starts `2015-11-01`, next 3 months.

## Current Repo State

The main pipeline is guarded:

```bash
docker exec bank-transaction-volume-forecast-jupyter-1 python -u run_pipeline_all.py
```

Default final `submission_stacked.csv` is the public-safe tree stack:

- LightGBM
- CatBoost GPU
- XGBoost CUDA

Experimental stacks are report-only unless explicitly enabled:

```bash
ALLOW_EXPERIMENTAL_STACK=1
```

Raw event temporal model is implemented but off by default:

```bash
RUN_EVENT_TEMPORAL=1
```

## Known Scores And Lessons

Known public-safe tree stack:

- Scenario: `lgbm_catboost_xgb`
- Local OOF: about `0.3779`
- Public score: `0.389916456`
- This is the fallback baseline.

Rolling + xgb_deep experimental stack:

- Local OOF: about `0.3728` to `0.3732`
- Public score: `0.391326105`
- Lesson: local OOF improvement did not transfer. Rolling/experimental validation is over-optimistic or misaligned with the public split.

Earlier invalid-looking public submissions:

- Scores around `197-200` happened when output scale was wrong.
- This is now guarded by log-space validation.

PyTorch monthly sequence:

- Previous monthly/aggregate PyTorch was weak, around `0.55` standalone OOF.
- It used only 35 monthly rows with 3 numeric fields, so it was not truly a raw temporal model.
- It is not trusted by default.

Band mixture-of-experts:

- Standalone OOF around `0.3818`.
- Did not produce a meaningful stack gain.

Rolling decomposition/high-tail:

- Produced strong local stacking gains.
- Failed public transfer, so any future use needs stricter validation alignment.

Raw event temporal model:

- Newly implemented.
- Uses recent exact transaction events plus older monthly pooled context.
- GRU/attention design, GPU PyTorch, group-safe rolling folds.
- Full training has not yet been evaluated.

## Data Nature Already Observed

Strongest known raw signal:

- `active_days_aug_oct` correlated with `log1p(target)` around `0.921`.
- Raw Aug-Oct transaction count correlated around `0.768`.

Overlooked signal families already added or explored:

- Activity continuity.
- Daily burstiness.
- Inactive gaps.
- Calendar/procedural behavior.
- Transaction type and batch mix.
- Account concentration.
- Birthday window features.
- Rolling pseudo-training snapshots.
- High-tail correction.
- Banded target specialists.

Residual issue:

- High activity users are usually underpredicted.
- Low activity users can be overpredicted.
- Public degradation suggests local validation is not matching leaderboard distribution or split logic.

## Research Questions

### 1. Validation Alignment

Find ways to validate that better predict public score than random 5-fold OOF.

Investigate:

- Could the public test split be time-biased, customer-segment-biased, region-biased, product-biased, or activity-level-biased?
- Does a validation split based on customer lifecycle, onboarding month, account age, target band, city, income category, or recent activity level better match the public gap?
- Should validation be adversarial: train/test classifier, then hold out train users most similar to test?
- Should we use repeated stratified group folds over target bands and activity bands rather than plain KFold?
- Can a blend selected by multiple stress-test validations beat the one selected by optimistic OOF?

Desired output:

- A concrete validation protocol to replace or complement current KFold.
- Exact split definitions.
- Metrics to track.
- Pass/fail rules for public submission candidates.

### 2. Temporal Generalization

The biggest plausible improvement may require modeling "future activity after this history" more directly.

Investigate:

- Whether rolling pseudo-examples are leaking optimism because early historical periods differ from the final test period.
- Whether rolling rows should be recency-weighted, using only later cutoffs, or calibrated against the real `2015-11-01` train snapshot.
- Whether monthly seasonality, holidays, salary cycles, debit-order cycles, or year-end banking behavior affects Nov-Jan uniquely.
- Whether future active days should be predicted as the primary target, with count derived secondarily.
- Whether event-level models should predict a daily intensity process instead of a 3-month aggregate.

Desired output:

- A temporal modeling approach that is likely to transfer to public.
- Specific target decomposition and calibration strategy.
- Whether the new raw event GRU is likely enough, or whether another architecture is preferable.

### 3. Public-Test Distribution Diagnosis

Investigate why local OOF can be `0.373` while public worsens to `0.391`.

Known likely causes:

- The public split may emphasize a subgroup where rolling models overpredict.
- The stack may be locally overfitting because OOF predictors share leakage-like correlated errors.
- The public leaderboard may contain only part of test with a different activity distribution.
- Test users may have different final-history behavior than train users.

Desired output:

- Suggested diagnostics using only train/test features and existing predictions.
- Distribution comparisons to run.
- Candidate "do not submit" indicators.

### 4. Raw Event Model Research

A raw event temporal model is implemented:

- Last `2048` exact events before cutoff.
- Token fields: date offsets, day/month/week features, signed amount, abs amount, balance, debit/credit, transaction type, batch type, reversal type, account rank.
- Older history: monthly pooled summaries.
- Model: categorical embeddings + continuous projection + GRU + masked attention pooling + static feature fusion.
- Heads: count regression, active-day auxiliary regression, target-band classification.

Investigate:

- Is GRU/TCN sufficient, or should we try event-time Transformer variants such as Performer/Linear Attention/Longformer-like local attention?
- Should sequences be daily compressed instead of event-level for heavy users?
- Should events be sampled by recency and type rather than simply taking the latest 2048?
- Should the model predict daily counts over the next 92 days and sum them?
- What auxiliary targets would improve transfer: next active days, max daily count, tail class, trend class, churn/reactivation?
- How to calibrate raw event outputs against the tree baseline before stacking?

Desired output:

- Ranked architecture recommendations for a 6GB GTX 1660 SUPER.
- Expected memory/runtime tradeoffs.
- Concrete next experiment configs.

### 5. Better Target Modeling

Research target-shape approaches that may help below `0.35`.

Candidates:

- Two-stage model:
  - active/inactive classifier for each future day or week.
  - conditional count model for active periods.
- Hurdle or zero-inflated count model.
- Quantile or distributional regression with calibrated mean/median blend.
- Tweedie/Poisson/Gamma objective variants, converted to log-space predictions.
- Mixture by target band with soft gating, but with better calibration than the current band MOE.
- Residual correction model trained only on baseline residual structure, using conservative shrinkage.

Desired output:

- Which target formulations are most credible for this dataset.
- How to evaluate them without another misleading public submission.

### 6. Feature Ideas Not Yet Fully Exploited

Do not just propose more generic aggregates. Look for features that encode procedural banking behavior.

Possible themes:

- Salary deposit rhythm and salary amount stability.
- Debit-order cadence and failed debit-order patterns.
- Month-end liquidity stress.
- Reversal chains and bounce recovery.
- Account switching between salary, spending, savings, mortgage/investments.
- Merchant/channel/batch behavior if available in encoded descriptions.
- Account-level age and product lifecycle.
- Dormancy/reactivation states.
- User-specific transaction rhythm: entropy, periodicity, autocorrelation, burst spacing.
- Last-N active-day features rather than calendar-window aggregates.
- Trend in active days, not just count.

Desired output:

- Prioritized feature families with expected impact.
- Exact definitions.
- Whether they should be added to tree features, raw event model, or both.

## Constraints

- Hardware: GTX 1660 SUPER, 6GB VRAM.
- Training should be GPU-first for heavy models:
  - CatBoost GPU.
  - XGBoost CUDA.
  - PyTorch CUDA.
- Polars GPU is unreliable for current feature expressions; CPU feature generation is acceptable if necessary, but model training should stay GPU.
- Public submissions are limited. Do not recommend submitting purely because local OOF improved.
- Any candidate must output log-space predictions.
- Safe fallback must remain `lgbm_catboost_xgb`.

## Desired Research Deliverable

Return a ranked action plan with:

1. Most likely root causes of the `0.3779 local -> 0.3899 public` gap.
2. A validation protocol that is more leaderboard-aligned than current KFold.
3. Top 5 experiments likely to move public score toward `<0.35`.
4. For each experiment:
   - hypothesis,
   - implementation sketch,
   - runtime estimate,
   - expected failure mode,
   - local validation gate,
   - public submission criteria.
5. Specific recommendations for the raw event temporal model.
6. Specific recommendations for high-tail and low-activity calibration.
7. A "stop doing this" list to avoid repeating locally-overfit work.

## Non-Negotiable Submission Gate

A new candidate should not be publicly submitted unless:

- It beats `lgbm_catboost_xgb` on the primary validation.
- It also beats or ties on at least one public-alignment stress split.
- It does not materially worsen `<20` target-band residuals.
- It does not materially worsen `500+` target-band residuals.
- Its test prediction distribution is close to or better calibrated than the safe public submission.
- It passes submission validation:
  - `3584` rows.
  - `3584` unique IDs.
  - no NaNs.
  - no negatives.
  - log-space scale, max under `20`.

## Current Best Practical Next Step

Before another public submission, run one of these:

1. Full raw event temporal experiment:

   ```bash
   docker exec -e RUN_EVENT_TEMPORAL=1 -e EVENT_BATCH_SIZE=8 bank-transaction-volume-forecast-jupyter-1 python -u run_pipeline_all.py
   ```

2. Validation-alignment diagnostics:

   - adversarial train/test classifier,
   - stratified target/activity folds,
   - residual reports by customer/product/activity segment,
   - prediction distribution comparison against the safe public file.

Only consider `ALLOW_EXPERIMENTAL_STACK=1` after the validation report gives a clear reason to trust the candidate beyond plain OOF.
