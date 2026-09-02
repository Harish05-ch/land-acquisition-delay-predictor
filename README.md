# Land Acquisition Delay Predictor

An AI-based early warning system that predicts whether a land acquisition case is likely to miss its deadline and estimates the expected delay.

## Problem

Land acquisition cases can be delayed due to:
- Land disputes
- Pending documents
- Compensation delays
- Court cases
- Large numbers of affected families

## Solution

Our system uses machine learning to:

1. Predict the probability of deadline delay
2. Estimate the number of additional delay days
3. Identify key risk factors
4. Suggest recommended actions

## Machine Learning

We use Random Forest models for:

- Classification: Predict whether a case will be delayed
- Regression: Estimate the number of delay days

## Dataset

The prototype uses 500 synthetic land acquisition cases.

The dataset contains:

- Land area
- Affected families
- Disputes
- Pending documents
- Compensation delay
- Court cases

## Results

- Classification accuracy: 95%
- Average delay prediction error: 9.4 days

These results are based on synthetic data and do not represent real-world accuracy.

## How to Run

Install the required libraries:

```bash
pip install pandas scikit-learn
