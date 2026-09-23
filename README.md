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
