# Fraud and Coordinated-Account Detection with Graph Analytics

Finding the accounts that consistently extract value from a platform, both
individually informed accounts and **coordinated groups** acting together,
from their transaction history. The project ranks them into an analyst review
queue and measures what restricting them is worth.

Methods: per-account value z-scores, a **coordination graph** with Louvain
community detection, a supervised model with out-of-fold evaluation,
precision/recall at k, and bootstrap confidence intervals for the business
impact.

> **Setting: a simulated market with real prices.** The testbed is a
> sports-betting market. Its prices are public and it has a well-defined
> "fair value" (the sharp closing price), so whether each transaction had an
> edge can be measured exactly. Transaction flow is simulated on 1,508 real
> Premier League matches (2021/22 to 2024/25), using real Bet365 and Pinnacle
> prices from [football-data.co.uk](https://www.football-data.co.uk/). The
> price movement is real; the accounts, and who placed what and why, are
> synthetic. All detection numbers are against planted accounts and are not
> evidence about any real customer base.

## Results

Detection runs on 2021/22–2022/23. Restriction is applied to the flagged
accounts and measured on 2023/24–2024/25, so the impact figures are
out-of-time. Each figure is the mean over 5 simulation seeds (sd in brackets).
Each seed has about 4,300 accounts, 91–100 of them planted, and about 440,000
transactions.

### Who gets found

| Ranking | Precision@25 | Precision@50 | Precision@100 | Recall@100 |
|---|---|---|---|---|
| Profit (P&L) | 0.30 (0.07) | 0.18 (0.06) | 0.12 (0.05) | 0.12 (0.05) |
| Value z-score (CLV) | 1.00 (0.00) | 1.00 (0.00) | 0.59 (0.06) | 0.62 (0.08) |
| Value + coordination graph | 1.00 (0.00) | 1.00 (0.00) | 0.76 (0.07) | 0.80 (0.09) |
| Supervised (out-of-fold) | 1.00 (0.00) | 1.00 (0.00) | 0.95 (0.03) | 1.00 (0.00) |

![Precision at k](reports/figures/precision_at_k.png)

- **Ranking by profit finds lucky ordinary accounts, not informed ones.** Over
  two seasons, outcome variance swamps a small edge. Across the 5 seeds, the
  informed accounts' median closing-price value was +0.1% to +1.2%, yet their
  median return in the detection period was −9.4% to −13.5%.
- **The coordination graph recovered all 10 planted coordinated groups as
  communities in every seed.** It also kept all 5 copy-trading groups apart
  from them. Copy-traders act together too, but their joint transactions lose
  value (in seed 0, group z −60 to −75), while coordinated groups' joint
  transactions beat the fair price (up to +12.7). Pooling a group's joint
  activity finds members whose own record is diluted by ordinary-looking
  activity. Individually, their median z-score is −0.14 to +1.5 across seeds,
  below the z = 2 an analyst would act on.
- **The supervised model needs labels.** Here they come from the simulation;
  in practice they would come from analyst-reviewed cases. The value and graph
  scores need none.

At a review threshold of z ≥ 2 rather than a fixed k:

| Score | Accounts flagged | Precision | Recall | Expected margin change | Volume refused |
|---|---|---|---|---|---|
| Value z-score | 33.0 (4.6) | 1.00 (0.00) | 0.35 (0.06) | +3,115 (637) | 3.1% |
| Value + coordination graph | 58.8 (6.0) | 1.00 (0.00) | 0.62 (0.08) | +3,360 (728) | 5.2% |

### What restricting them is worth

The scenario refuses every transaction the top-100 accounts make in the
following two seasons. Platform margin (GGR) over those seasons averaged
160,577 units:

| Ranking | Expected margin change | Realised margin change | Volume refused |
|---|---|---|---|
| Profit (P&L) | −7,480 (781) | −6,005 (2,606) | 5.8% |
| Value z-score | +3,770 (634) | −4,589 (9,254) | 5.9% |
| Value + coordination graph | +3,701 (755) | −6,259 (10,432) | 7.4% |
| Supervised (out-of-fold) | +3,136 (1,005) | −11,696 (10,752) | 10.6% |
| All planted accounts | +3,112 (1,096) | −12,069 (11,030) | 10.6% |

![Margin impact](reports/figures/ggr_impact.png)

- **Restricting by profit loses money**: about −7,500 expected (−4.7% of
  margin), because it refuses ordinary accounts the platform profits from.
- **Restricting by value gains about +3,700 expected, roughly +2.3% of
  margin.** Ranking by value gains more than refusing every planted account,
  because some planted accounts have no real edge and their other activity is
  profitable.
- **The realised change cannot be measured on two seasons of outcomes.** For
  seed 0, the 95% bootstrap interval for the value + graph restriction is
  −28,494 to +31,934, against an expected +3,007. The seeds share the same real
  outcomes, so their spread understates this uncertainty. I checked that this
  is variance, not bias: on the choices informed accounts made, actual
  outcomes match the fair probabilities in every seed (|z| ≤ 0.7). This is why
  the expected measure is the one to manage by.

## How it works

**Prices** (`src/sbrisk/odds.py`). The platform offers Bet365's prices, moving
linearly in implied probability from the opening to the closing price over
the 96 hours before each event. Fair probability is Pinnacle's closing price
with the margin removed. A transaction's closing-price value (CLV) is
`price × fair probability − 1`. Two markets are used: match result and
over/under 2.5 goals. 2025/26 is excluded because Pinnacle closing prices are
missing for 170 of its 380 matches.

**Accounts** (`src/sbrisk/simulate.py`).

| Group | Behaviour | Label |
|---|---|---|
| Ordinary (4,000) | Acts close to the event and favours favourites; no information | 0 |
| Copy-traders (5 groups of 30–60) | Copy a shared pick within minutes: coordinated, no edge | 0 |
| Informed accounts (40) | Noisy estimate of the fair price; act 36–96 h out when the price beats it by 2% | 1 |
| Coordinated groups (10 groups of 3–8) | One shared estimate; members act within 5 minutes of each other | 1 |

Informed and coordinated accounts also make ordinary-looking transactions, to
dilute their record. They act early because that is where the real edge is:
96 hours out, 46% of events have a choice priced more than 2% above fair
value; at 24 hours, 4.6%.

**Detection** (`features.py`, `graph.py`, `detect.py`).
- A per-account value z-score.
- A coordination graph joining accounts that make the same choice within 10
  minutes (at least 5 times, and at least 10% of the smaller account's
  activity). It is split into Louvain communities, and each community is
  scored by the z-score of its joint transactions.
- A review queue with the evidence for each account
  (`reports/review_queue_seed0.csv`).

**Evaluation** (`evaluate.py`).
- Precision and recall at k, against the planted accounts.
- Margin change from refusing flagged accounts in the next period, both
  realised (with a bootstrap over events) and expected from fair prices.

## Limits

- **Synthetic accounts.** Real coordinated groups spread activity over more
  accounts, platforms and time, and real coordination is much noisier than
  here. The graph's clean recovery and the zero false positives at z ≥ 2 are
  best cases.
- **Two price snapshots per event.** The path between them is assumed linear.
- **The platform is a price-taker.** It does not adjust its own prices in
  response to informed activity, so the value of that signal is not counted.
- **Restriction means refusing all activity.** Limits instead scale the effect
  roughly linearly. Lost ordinary volume from restricted accounts is counted;
  the wider commercial cost of restricting customers is not.

## Run it

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt     # Windows; venv/bin/pip elsewhere
venv/Scripts/python scripts/run_experiment.py    # downloads the CSVs, ~20 s for 5 seeds
venv/Scripts/python -m pytest -q
```

Outputs: `reports/results.json` (every metric, per seed and summarised),
`reports/review_queue_seed0.csv`, `reports/figures/`.
