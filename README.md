# Real-Time Card Fraud Detection

## Problem

A payment processor receives a large number of transactions, but the fraud team can manually review only 200 alerts per day. The goal is to build a fraud detection system that ranks transactions by risk and sends the most suspicious ones for review.

## Dataset

IEEE-CIS Fraud Detection dataset from Kaggle.

Main files:
- train_transaction.csv
- train_identity.csv

Main challenge:
- severe class imbalance
- time-based fraud patterns
- anonymized features
- missing values
- risk of temporal leakage

## Approach

Planned approach:
1. Time-based train/validation/test split
2. LightGBM baseline
3. Velocity and entity feature engineering
4. PR-AUC and precision@200 evaluation
5. Cost-sensitive threshold optimization
6. SHAP reason codes
7. FastAPI scoring service
8. Streamlit analyst dashboard
9. Drift monitoring

## Current Status

Project initialized.