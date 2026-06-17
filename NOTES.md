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