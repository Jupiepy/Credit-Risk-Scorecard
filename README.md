# Credit Risk Scorecard — Consumer Lending

> An end-to-end **credit scorecard** for consumer lending, built the way digital
> lenders (蚂蚁 / 微众 / 京东白条 style) actually do it: WOE binning → Information
> Value selection → logistic regression → a point-based scorecard, evaluated with
> KS and AUC.

Credit scoring is one of the oldest and most successful **machine-learning
applications in FinTech** — the model that decides "how much credit, at what rate"
for a retail borrower. This repo implements the full pipeline from raw credit data
to a deployable scorecard.

## What It Does

| Step | Method | Output |
|---|---|---|
| 1. Load & clean | pandas | 1000 borrowers, 20 features, 30% bad rate |
| 2. Binning | qcut (numeric) / category (categorical) | each feature → bins, `missing` handled |
| 3. Weight of Evidence | WOE = ln(P_good / P_bad) | per-bin evidence |
| 4. Information Value | IV = Σ (pct_good − pct_bad) · WOE | feature screening (keep IV ≥ 0.02) |
| 5. Logistic regression | scikit-learn on WOE-transformed features | default-probability model |
| 6. Scorecard scaling | base score + points per bin (PDO) | human-readable scorecard |
| 7. Evaluation | KS, AUC | discrimination power |

## Results

On the UCI **German Credit** dataset (in-sample):

- **KS = 0.523, AUC = 0.826**
- **15 of 20 features retained** (IV ≥ 0.02)
- **`checking_status` is the dominant signal** (IV = 0.666), followed by
  `credit_history` (0.293), `duration` (0.216), `savings_status` (0.196) and
  `purpose` (0.169).

Scorecard excerpt (base score 600, PDO 20 → 20 points double the good:bad odds):

| Feature | Bin | Points |
|---|---|---|
| checking_status | `>=200` | +9.33 |
| checking_status | `no checking` | +27.06 |
| checking_status | `0-200` | −9.24 |
| checking_status | `<0` | −18.82 |
| duration | `(12.0, 15.0]` | +14.58 |
| duration | `(30.0, 72.0]` | −16.80 |
| credit_history | `critical` | +14.64 |

> Higher score = lower default risk. A negative-balance account (`<0`) costs ~19
> points, while a longer loan duration (>30 months) costs ~17 points — both
> intuitively raise default risk.

### A note worth knowing (interview material)

`no checking` scoring **+27** looks counter-intuitive — "no bank account" reads like
a risk signal. But in this dataset that category has the *lowest* default rate
(11.7%), while `<0` balance is the highest (49.3%). The model correctly learns from
the data, not from prior assumptions — exactly what you'd verify as a risk analyst
before trusting a variable's direction.

## Repository Structure

```
Credit-Risk-Scorecard/
├── README.md
├── requirements.txt
├── scorecard.py            # the full pipeline
├── data/
│   └── german_credit.csv   # cleaned, labelled UCI German Credit
└── output/                 # generated: scorecard.csv, iv_table.csv, ...
```

## Getting Started

```bash
pip install -r requirements.txt
#   or (CN mirror):  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python scorecard.py --input data/german_credit.csv --output-dir output
```

Outputs written to `output/`:

- `scorecard.csv` — the point-based scorecard (feature, bin, WOE, coef, points)
- `iv_table.csv` — every feature's Information Value
- `woe_table.csv` — per-bin counts and WOE
- `metrics.json` — KS, AUC, bad rate, scaling parameters

## Data

UCI **Statlog (German Credit)**: 1000 applicants, 20 attributes (7 numeric, 13
categorical), target `1 = default / 0 = good`. The raw `german.data` was mapped to
readable labels and re-coded; see the column meanings below.

| Column | Meaning |
|---|---|
| checking_status | status of existing checking account |
| duration | loan duration in months |
| credit_history | credit history |
| purpose | loan purpose |
| credit_amount | credit amount |
| savings_status | savings / bonds |
| employment | present employment since |
| installment_rate | instalment rate (% of disposable income) |
| personal_status | personal status and sex |
| other_debtors | other debtors / guarantors |
| residence_since | present residence since (years) |
| property | property |
| age | age in years |
| other_installment | other installment plans |
| housing | housing |
| existing_credits | number of existing credits at this bank |
| job | job |
| num_dependents | number of dependents |
| own_telephone | telephone |
| foreign_worker | foreign worker |

## Production Next Steps (the FinTech angle)

This is an in-sample baseline. A production credit model adds:

- **Train / validation / out-of-time (OOT) splits** — the in-sample KS here (0.52)
  is optimistic; a real OOT figure is typically lower (≈0.35–0.45 on this data).
- **Reject inference** — model only sees approved applicants, so the training
  sample is biased; techniques (parcelling, augmentation) correct for that.
- **Model explainability** — SHAP / LIME to explain individual decisions to
  regulators and customers (a core FinTech compliance requirement).
- **PSI / drift monitoring** — track score stability after deployment.
- **PD calibration** — map scores to absolute probability of default for pricing
  and capital (Basel / IFRS 9).

## Tech Stack

Python · pandas · numpy · scikit-learn
