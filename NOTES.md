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

### Day 10 — Imbalance strategy experiment

Tested four strategies on 5-fold walk-forward CV using tuned hyperparameters:

| Strategy | PR-AUC | Std | P@200 | Notes |
|---|---|---|---|---|
| baseline (no rebalancing) | 0.6499 | 0.037 | 0.980 | Best AND most stable |
| SMOTE | 0.6476 | 0.058 | 0.970 | ~tied on mean, 60% more variance |
| undersample 5:1 | 0.6277 | 0.056 | 0.977 | Threw away information |
| scale_pos_weight | 0.6262 | 0.069 | 0.962 | Worst — distorted feature learning |

**Result: baseline wins.** No rebalancing is applied.

Why: LightGBM handles imbalance well through pure-region tree splits. Rank-based
metrics don't reward the calibrated probabilities that rebalancing would improve.
SMOTE-synthesized minority samples in high-dim sparse fraud data don't reflect 
real fraud patterns. Undersampling costs information. scale_pos_weight distorts 
feature-target relationships during learning.

This matches the general finding in fraud literature that tree-based models with 
rank-based eval rarely benefit from rebalancing. Calibration (Day 11) will be 
applied afterward to fix probability scores without touching training data.

### Day 10 MLflow logging
Logged parameters and metrics only — no model artifacts, since Day 10 was 
a strategy comparison and no individual fold-trained model was retained.
The winning strategy (baseline, no rebalancing) will be applied when 
training the final champion model on Day 11 with calibration, and that 
model will be fully logged with artifacts.
### Day 11 — Calibration results
Champion model (baseline, tuned params, 355 trees) evaluated on val:
- ROC-AUC       : 0.9231
- PR-AUC        : 0.6142
- P@200         : 1.0000
- Brier (raw)   : 0.02263
- Brier (cal)   : 0.02214 (in-sample; honest test-set measurement on Day 14)

Key finding: model is already reasonably calibrated out-of-the-box.
Isotonic calibration provided only 2.2% Brier improvement. This is a 
consequence of:
  1. No rebalancing applied (baseline strategy from Day 10)
  2. Early stopping on PR-AUC (not logloss) prevented over-fitting toward
     extreme probabilities

Reliability curves for raw and calibrated overlap almost perfectly along
the diagonal for the observable score range (0 to ~0.42). Score 
distribution is heavily concentrated at the low end (as expected for
imbalanced data), so no observations exist beyond ~0.42 mean predicted
probability in this val set.

Calibrator kept for pipeline consistency — will still be applied at
scoring time so downstream cost-sensitive threshold calculations (Day 12)
use meaningful probabilities. Contribution is small but non-zero.
### Day 12 — Cost-sensitive threshold optimization

Business context (derived from data):
  Daily transaction volume : 2953
  Avg fraud amount         : $172
  Analyst review cost      : $5 (estimated)
  Daily review capacity    : 200 alerts

Optimal operating threshold: 0.1063 (calibrated probability)

At this threshold:
  Alerts per day    : 191 (95% capacity utilization)
  Precision         : 41.5%
  Recall            : 67.7%
  Fraud caught      : 2,771 (of 4,092)
  Fraud missed      : 1,321
  False alarms      : 3,905 (over 35-day val window)

Daily economics:
  Gross savings     : $13,568
  Review cost       : $953
  Net daily savings : $12,615
  Annualized (×365) : $4,604,658

Caveats to communicate:
  - Avg fraud recovery is likely 60-80% of face value in practice
    → realistic annual savings: $2.8M–$3.7M
  - Review cost is estimated at $5 flat; real analyst time varies
    from ~$0.50 (fast reject) to $30 (complex case)
  - Model + threshold assume production data distribution matches val

Threshold was chosen to maximize net daily savings SUBJECT TO the 
capacity constraint. Without the constraint, the theoretical peak savings
threshold is ~0.02, but that generates thousands of alerts/day — 
unusable by a real fraud team.

Result: model saves an estimated $3-4M annually while staying within
analyst review capacity.

## Results

On a held-out test set of 118K transactions (~42 days, 3.4% fraud rate), 
the model achieves:

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | 0.887 | [0.882, 0.893] |
| PR-AUC | 0.461 | [0.446, 0.477] |
| Precision@200 | 0.960 | [0.905, 0.975] |
| Brier score | 0.024 | — |

At the cost-optimized operating threshold (0.106 on calibrated probability):
- **58% of fraud caught** while flagging 181 transactions per day (91% capacity utilization)
- **31% precision** at operating point (top-200 flagged: 96% precision)
- **Net daily savings: ~$7,500**, or **~$2.7M annually** at the assumed 
  $172 average fraud amount and $5 analyst review cost

Adjusted for realistic fraud recovery rates (60–80% of face value), the 
estimated annual net savings range is **$1.6M–$2.2M**.

Test performance is meaningfully below validation performance 
(PR-AUC 0.46 vs 0.61) — consistent with the concept drift observed 
during walk-forward cross-validation. Production deployment would 
require monthly retraining and continuous drift monitoring.

### Day 14 — Final test evaluation

The gap between val and test performance (PR-AUC 0.61 → 0.46) is the 
single most important finding of the project. This is fraud drift 
made visible.

Val was drawn from days ~110-140 (middle of timeline).
Test was drawn from days 145-190 (end of timeline).

Between val and test, roughly 5 weeks pass. In fraud, tactics evolve 
significantly in 5 weeks — new attack rings emerge, existing rings adapt 
after countermeasures, seasonal patterns shift.

The model still delivers strong business value ($2.7M annualized) but
degrades meaningfully on temporally-distant data. This is precisely why:

1. Test set was held out (never touched during any tuning)
2. Walk-forward CV was used instead of random K-fold  
3. Week 4 will implement drift monitoring
4. Production deployment plan includes monthly retraining

Also interesting: P@200 held up much better (1.00 → 0.96) than 
PR-AUC (0.61 → 0.46). Top-of-distribution ranking is robust; deep 
score ranking is fragile. This validates the top-k operating strategy — 
the model is more trustworthy at its most confident predictions.

### Day 15 — Global SHAP explainability

Top 5 features by SHAP (mean |value|):
  C13            0.263  (Vesta's own count feature)
  TransactionAmt 0.256
  TransactionDT  0.221
  card6          0.173  (credit vs debit)
  P_emaildomain  0.170  (purchaser email domain)

Engineered features in top 15:
  card1_tx_count_7d at rank #12 — SHAP-verified that Day 6 velocity 
  engineering added real signal, not just numerical lift.

Key insight — SHAP vs LightGBM built-in gain importance disagree 
meaningfully on 5+ features:
  - C5 / V91 / V70 / card6 / P_emaildomain: 
    LOW split count, HIGH per-split impact
    → decisive but rarely-used features → drift-sensitive
  - card2 / C14: 
    HIGH split count, MODEST per-split impact
    → granular features → drift-robust

This distinction guides monitoring priorities in Week 4.

Beeswarm reveals directional effects:
  - LOW C13 → pushes toward fraud (short history is risky)
  - LOW TransactionAmt → pushes toward fraud (card testing pattern)
  - LOW card1_tx_count_7d → pushes toward fraud (dormant/new cards)
  - Categorical features (card6, P_emaildomain) show clean split behavior

SHAP TreeExplainer used on 5000-row val sample. Exact algorithm for tree 
ensembles — no approximation. Full explainer saved to models/shap_explainer.pkl 
for Day 16 reason codes.