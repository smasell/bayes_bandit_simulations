# Bayes bandit simulations

Thompson sampling to maximize revenue per visitor across prices 60, 80, 100, and 120.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python price_bandit.py --output-dir results
```

Change assumed purchase probabilities (in price order):

```bash
python price_bandit.py --rates 0.20 0.17 0.15 0.10 --runs 300 --visitors 50000 --output-dir results
```

Default probabilities are hypothetical, not measured. Independent Beta(1, 1) priors are updated with immediate purchase/no-purchase feedback. Each decision maximizes price times a sampled conversion probability.

The reported convergence metric is the first time an optimal price receives at least 90% of the last 1,000 visitors. It is not permanent convergence or posterior confidence. Runs that never reach the threshold remain censored when calculating quantiles. The CSV also reports the fraction currently above the threshold.

Outputs: summary.json, convergence.csv, convergence.svg. The simulation assumes stationary probabilities and independent visitors.

## Historical day-7 revenue simulation

`historical_bandit.py` simulates prices 70, 80, 90, and 100 from a one-column,
user-level revenue CSV, including zero-revenue users. It constructs a 15% lift
for price 100 in two ways: multiplying payer spending by 1.15, or multiplying
the payer probability by 1.15. Prices 70/80/90 have the same baseline distribution.
A third, no-lift scenario measures false winner declarations.

```bash
python historical_bandit.py --data /path/to/revenue_day7.csv --runs 2000 --horizon 300000 --output-dir results/historical_15pct
```

Compare 5%, 10%, and 15% uplifts with a longer horizon for small effects:

```bash
python historical_bandit.py --data /path/to/revenue_day7.csv --uplifts .05 .10 .15 --runs 2000 --horizon 1500000 --output-dir results/historical_uplifts
```

This runs both lift mechanisms for each effect and one no-lift check, producing
`comparison.md`, `summary.json`, and `progress.csv`. Seeds are shared across
uplift levels. A completion quantile beyond the cap is reported as unavailable,
not calculated only from the subset of successful runs.

The policy starts with 2,000 users per price, then allocates 2,000-user batches
using Thompson sampling mixed with 5% exploration per price. Its posterior is
a normal approximation to the Bayesian bootstrap for mean revenue, with exact
Bayesian-bootstrap mean and variance. Posterior probabilities use 1,024
screening draws and 8,192 independent confirmation draws. The run stops when
the posterior probability that any arm is best reaches 95%.

Outputs are `summary.json` and `progress.csv`. Completion quantiles include all
runs: a wrong winner or unfinished run cannot count as successful completion.
This posterior stopping rule does not guarantee a 5% frequentist error rate;
consult the simulated no-lift results. All batch outcomes mature before the
next allocation. These estimates concern completed day-7 outcomes, not calendar
duration or a live delayed-feedback pipeline. Prices are labels, not revenue
multipliers: the CSV alone does not identify price elasticity. Outcomes assume
independent users and a stationary distribution; uncertainty in the historical
distribution itself is not simulated.
