# Model Card — German Credit Scorecard

## Overview
This repository implements a traditional **scorecard** for retail credit risk on the
UCI "German Credit" dataset: variable binning, Weight-of-Evidence (WOE) transformation,
logistic-regression scoring, and a train / holdout / cross-validation evaluation harness.

## Intended use
- Educational reference for the scorecard development workflow.
- Baseline model for binary default prediction (good vs. bad loans).

## Data
- Source: UCI German Credit (`data/german_credit.csv`), 1000 observations, 20 features.
- Target: `bad` (1 = credit application rejected / default, 0 = accepted).
- The train/holdout split is stratified; cross-validation is stratified k-fold.

## Methodology
1. **Binning** — supervised binning per predictor with a monotonic WOE trend.
2. **WOE encoding** — each bin is mapped to its Weight-of-Evidence.
3. **Model** — logistic regression on the WOE features.
4. **Evaluation** — KS and AUC on train, holdout and 5-fold CV.

## Performance (reference run)
Reported from the bundled pipeline (see `output/metrics.json` for the live numbers):

| Split | KS | AUC |
|-------|-----|-----|
| Train | ~0.52 | ~0.83 |
| Holdout | ~0.47 | ~0.75 |
| 5-fold CV (mean ± sd) | ~0.47 ± 0.04 | — |

> Figures are illustrative of a correctly-specified scorecard; re-run the pipeline
> on your own data for exact values.

## Limitations
- Small sample (n = 1000) → wide confidence intervals.
- Linear WOE assumption; non-linearities are handled only through binning.
- Not calibrated for a specific approval threshold or regulatory use.

## Usage
```bash
python scorecard.py --data data/german_credit.csv --holdout-frac 0.3 --cv 5
```
Outputs land in `output/` (IV table, WOE table, scorecard, metrics).
