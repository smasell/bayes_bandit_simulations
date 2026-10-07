# Historical revenue Bayesian bandit comparison

2,000 simulations per scenario. All user counts are totals across prices 70, 80, 90, and 100.

| Lift | Mechanism | 50% correct completion | 80% correct completion | 90% correct completion | Wrong winner at cap | Unfinished at cap |
|---|---|---:|---:|---:|---:|---:|
| 5% | Higher spending | 368,000 | 634,000 | 830,000 | 1.00% | 0.55% |
| 5% | More payers | 384,000 | 678,000 | 906,000 | 1.50% | 0.65% |
| 10% | Higher spending | 102,000 | 180,000 | 236,000 | 0.65% | 0.00% |
| 10% | More payers | 98,000 | 174,000 | 222,000 | 0.25% | 0.00% |
| 15% | Higher spending | 46,000 | 82,000 | 104,000 | 0.10% | 0.00% |
| 15% | More payers | 48,000 | 82,000 | 102,000 | 0.20% | 0.00% |

## Decision rule and assumptions

Thompson sampling; approximate Bayesian bootstrap normal posterior for mean revenue; 2,000 warm-up users per arm; 2,000-user mature batches; 5% minimum exploration per arm; stop at P(arm is best)>=95%; 1,024 screening and 8,192 confirmation posterior draws.

With no lift, 8.65% of runs falsely declared a winner by 1,500,000 users.

Completion means stopping with price 100 as best, not establishing that its lift exceeds the scenario percentage. Wrong and unfinished runs count as failures. Quantiles have Monte Carlo uncertainty.

Normal approximation to Bayesian bootstrap, not exact posterior sampling. Empirical stationary IID outcomes. All batch outcomes mature before the next allocation. Prices 70/80/90 share the historical distribution; price 100 has the specified constructed lift. This does not estimate actual price elasticity. Repeated posterior stopping has no guaranteed frequentist 5% error control.

All batch outcomes are observed before the next allocation. Calendar duration and continuous enrollment with delayed feedback are not simulated. The empirical distribution is treated as known; historical-distribution uncertainty is excluded.

The same seeds are reused across uplift scenarios for comparability. Full parameters and results are in summary.json; milestone probabilities are in progress.csv.
