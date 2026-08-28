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

### Remaining fraud budget for Week 2 improvements
- ROC-AUC: 0.923 → room to grow (ceiling ~0.98)
- PR-AUC: 0.600 → most room to grow (ceiling ~0.85 realistic)
- P@200: 0.990 → near ceiling, Week 2 unlikely to move this much

### CV design trade-off (Day 8)
val_fraction is auto-computed from n_folds so window sizes fit within
the CV data without overlap. But this creates a trade-off:
- More folds → smaller val → noisier per-fold metrics
- Fewer folds → bigger val → fewer independent measurements

Chose 5 folds (val_fraction ≈ 10%) — gives ~1600 frauds per val window,
comfortably above the 500-fraud threshold for stable metrics.

Added min_val_frauds guardrail that warns if any fold's val window is 
too small to trust.

### Walk-forward CV results (Day 8)
5-fold walk-forward, initial train=50%, val_fraction=10% (auto), test held out.

Metric         Mean     Std      Range
ROC-AUC       0.926    0.015    [0.90, 0.94]
PR-AUC        0.637    0.048    [0.56, 0.68]  
P@200         0.973    0.010    [0.96, 0.99]

Single-split val (Day 6) reported PR-AUC 0.600 — that was one draw from a 
distribution actually centered at 0.637. CV gives more honest headline number.

Fold 2 is a soft outlier: PR-AUC 0.56 vs ~0.65 elsewhere, best_iter 539 vs 
~700 avg. Interpretation: fraud tactics or customer mix shifted in that 
window — the model learned less transferable patterns from prior data. 
This is real concept drift, and it's the kind of thing K-fold would have 
averaged away.

Best iteration variation (539-865) suggests periodic retraining would be 
necessary in production. Adds to Week 4 monitoring/drift story.


### Day 9 — Optuna results (30 trials, 5 folds walk-forward)
Best trial: 13
Best PR-AUC: 0.6499 (baseline 0.6371, +2% relative lift)
Best params:
  learning_rate     0.072
  num_leaves        128
  max_depth         12
  min_child_samples 11
  feature_fraction  0.50   ← strong feature regularization preferred
  bagging_fraction  0.81
  bagging_freq      3
  lambda_l1         ~0
  lambda_l2         0.004

Optuna preferred higher capacity per tree + aggressive feature 
subsampling — model works better when forced to diversify across features
rather than relying on the same few top features every tree.

Fold 2 remained persistently ~0.1 below other folds across ALL trials — 
this is real concept drift, not a hyperparameter issue. Will address 
via monitoring (Week 4), not modeling.