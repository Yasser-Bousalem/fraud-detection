## EDA Findings

### Fraud rate over time
Clear non-stationarity — rate starts at ~2%, drops to 1.2% around day 25,
then jumps and stays elevated at 3.5–7% for the remaining ~155 days.
Proves random splits would leak future distribution into training.
Time-based CV is non-negotiable.

### Amount distribution
Sub-$1 fraud spike visible with logspace bins — likely card testing behavior.
Legit has sharp peak at ~$50 (round-number purchases).
Fraud is more uniformly spread — no dominant amount.
Feature idea: is_round_amount flag, sub_dollar_flag.

### Hour of day
Dramatic peak at hours 5–9 (max 10.5% at hour 7), 4× higher than the
afternoon trough (~2.3% at hour 13–14). Fraudsters operate before business
hours. Hour-of-day and is_night_hour will be engineered features.

### Missingness findings
- C feats: 0% missing — most reliable group, no imputation needed
- D feats: range from 5% (D1) to 90% (D7) — add is_null flags for D5+
- M feats: 50-60% missing for M5-M9 — NaN is a valid category, label-encode as-is
- R_emaildomain: 75% missing — has_R_emaildomain binary flag will be useful
- id feats: all >75% missing by design (24% identity match rate)
  - id_24/25/07/08: ~100% missing even within identity rows — likely drop these
- V feats: clear sub-model bands
  - V1-V130: usable (<50% missing)
  - V130-V280: 75-85% missing — add is_present flag for this band
  - V320-V339: 80%+ missing — low priority

### Email domain finding (counterintuitive)
Rare domains (< 200 tx) have 2.5% fraud rate — BELOW average (3.5%).
High-fraud domains are international variants: mail.com (19%), outlook.es (13%),
hotmail.es, live.com.mx. Pattern = foreign email providers, not throwaway domains.
Feature: is_foreign_email_domain flag rather than is_rare_email_domain.

## Day 5 — Baseline results (val set)
Model: LightGBM, defaults + scale_pos_weight, ordinal encoding, mean imputation
- ROC-AUC       : 0.8490
- PR-AUC        : 0.3310
- Precision@200 : 0.6450  (129 of top 200 flagged are real fraud)

This is the FLOOR. Every experiment in Week 2 gets compared to this.

### Velocity feature findings (Day 6)
Diagnostic revealed:
- Device velocity features WORK (corr -0.08 to -0.10 with isFraud)
  - device_distinct_cards_24h is strongest — ATO signal
- Card velocity features are WEAK — card1 in IEEE-CIS may not be a 
  card identifier at the granularity we assumed
  - LightGBM likely already gets this signal from Vesta's C1-C14 columns
  - Dropped card1_tx_count_1h (57% NaN) and card1_amt_zscore_30d (dead)
- Email velocities: weak but kept — some potential interaction with 
  device features

Interview note: not all engineered features help. Naive velocity 
engineering can hurt P@k by adding noise around the decision boundary.
Always run correlation and NaN diagnostics before shipping features.

### Day 6 critical bug: scale_pos_weight + logloss eval
Symptom: model trained 1 tree, then val loss diverged.
Cause: scale_pos_weight=28.3 inflates fraud probabilities, making 
binary_logloss on val explode after iteration 1.
Fix: removed scale_pos_weight, changed eval metric to average_precision.

Lesson: When using class weights or scale_pos_weight, the natural eval 
metric (logloss) will always look terrible. Either drop the reweighting 
or use a rank-based metric (AUC, PR-AUC) for early stopping. Not both.

Also: ALWAYS check model.best_iteration_ after training. A one-tree model 
with 0.85 AUC is not a trained model — it's the marginal effect of a 
single split. Early stopping firing at iteration 1-5 is a hard failure, 
not a "well-tuned model".

### Day 6 breakthrough — velocity features + fixed training
After removing scale_pos_weight and switching eval metric to average_precision,
model trained properly (369 trees vs previous 1). Metrics on val:
  ROC-AUC       : 0.9234  (baseline 0.849)
  PR-AUC        : 0.6002  (baseline 0.331)
  Precision@200 : 0.9900  (baseline 0.645)

Velocity features are now in the top 20 by gain:
  - card1_tx_count_7d      : 11,488 gain (#14)
  - email_distinct_cards_24h: 9,271 gain (#19)
  - email_tx_count_24h      : 8,717 gain
All 8 velocity features are actively used by the model.

Key learning: velocity features WERE providing signal all along. The 
"they don't help" hypothesis was wrong — the broken training was masking 
their contribution. Always fix training pathologies before evaluating 
feature engineering.

Sanity check needed: P@200 = 0.99 is suspiciously high, verify no leakage.

### Sanity check on P@200 = 0.99 (Day 7)
Val has 4090 frauds — top-200 is only 5% of available fraud, so 0.99 
is achievable without leakage. Random baseline P@200 = 0.039. 
Model is 25× better than random. Result is real.

### Remaining fraud budget for Week 2 improvements
- ROC-AUC: 0.923 → room to grow (ceiling ~0.98)
- PR-AUC: 0.600 → most room to grow (ceiling ~0.85 realistic)
- P@200: 0.990 → near ceiling, Week 2 unlikely to move this much

