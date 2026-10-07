"""Empirical day-7 revenue bandit with an approximate Bayesian bootstrap posterior.

Allocation and stopping use posterior draws of mean revenue, never true arm means.
Each allocation batch is observed in full before the next batch (mature cohorts).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def posterior_draws(rng, n, sums, squares, draws):
    # Bayesian bootstrap: Dirichlet(1,...,1) observation weights.
    # Exact posterior mean = sample mean; variance = sample variance*(n-1)/(n*(n+1)).
    # Normal approximation is used after a 2,000-user-per-arm warm-up.
    mean = sums / n
    variance = np.maximum(squares / n - mean**2, 0) / (n + 1)
    return mean[None] + rng.standard_normal((draws,) + mean.shape) * np.sqrt(variance)[None]


def simulate(x, scenario, runs=500, horizon=300000, seed=42, uplift=.15):
    prices = np.array([70, 80, 90, 100])
    positive = x[x > 0]
    payer_rate = (x > 0).mean()
    if scenario not in ('spend_lift', 'payer_lift', 'null') or not np.isfinite(uplift) or uplift < 0:
        raise ValueError('Unknown scenario or invalid uplift.')
    scales = np.array([1., 1., 1., 1 + uplift if scenario == 'spend_lift' else 1.])
    rates = np.full(4, payer_rate)
    if scenario == 'payer_lift':
        rates[-1] *= 1 + uplift
    if np.any(rates > 1):
        raise ValueError('Uplift implies an impossible payer probability.')
    true_means = rates * positive.mean() * scales
    streams = np.random.SeedSequence(seed).spawn(3)
    env, policy, inference = [np.random.default_rng(s) for s in streams]
    n = np.zeros((runs, 4), dtype=int)
    sums = np.zeros_like(n, dtype=float)
    squares = np.zeros_like(sums)
    stopped = np.zeros(runs, dtype=int)
    winners = np.full(runs, -1, dtype=int)
    snapshots = []

    def observe(ids, allocation):
        payers = env.binomial(allocation, rates)
        for arm in range(4):
            count = payers[:, arm]
            # Draw every positive outcome from the original positive-revenue sample.
            values = env.choice(positive, size=int(count.sum())) * scales[arm]
            boundaries = np.r_[0, np.cumsum(count)]
            cumulative = np.r_[0., np.cumsum(values)]
            cumulative_sq = np.r_[0., np.cumsum(values**2)]
            sums[ids, arm] += np.diff(cumulative[boundaries])
            squares[ids, arm] += np.diff(cumulative_sq[boundaries])
        n[ids] += allocation

    observe(np.arange(runs), np.full((runs, 4), 2000))
    batch = 2000
    for total in range(10000, horizon + 1, batch):
        ids = np.flatnonzero(stopped == 0)
        if len(ids):
            ts = posterior_draws(policy, n[ids], sums[ids], squares[ids], 256)
            chosen = ts.argmax(axis=2)
            probabilities = np.stack([(chosen == arm).mean(axis=0) for arm in range(4)], axis=1)
            probabilities = .8 * probabilities + .05  # 5% exploration per arm.
            allocation = np.array([policy.multinomial(batch, row) for row in probabilities])
            observe(ids, allocation)
            draws = posterior_draws(inference, n[ids], sums[ids], squares[ids], 1024)
            best = draws.argmax(axis=2)
            chances = np.stack([(best == arm).mean(axis=0) for arm in range(4)], axis=1)
            candidates = np.flatnonzero(chances.max(axis=1) >= .93)
            if len(candidates):
                candidate_ids = ids[candidates]
                # Independent, larger draw set keeps Monte Carlo noise near 95% small.
                confirmed = posterior_draws(inference, n[candidate_ids], sums[candidate_ids], squares[candidate_ids], 8192).argmax(axis=2)
                chance = np.stack([(confirmed == arm).mean(axis=0) for arm in range(4)], axis=1)
                hit = chance.max(axis=1) >= .95
                stopped[candidate_ids[hit]] = total
                winners[candidate_ids[hit]] = chance[hit].argmax(axis=1)
        if total % 10000 == 0:
            correct = (winners == 3) if scenario != 'null' else np.zeros(runs, dtype=bool)
            snapshots.append({
                'users': total,
                'correct_stop_fraction': float(correct.mean()),
                'wrong_stop_fraction': float(((stopped > 0) & ~correct).mean()),
                'unfinished_fraction': float((stopped == 0).mean()),
                'mean_price100_allocation_share': float((n[:, 3] / n.sum(axis=1)).mean()),
            })
        if not np.any(stopped == 0):
            break
    correct = (winners == 3) if scenario != 'null' else np.zeros(runs, dtype=bool)
    completion = np.sort(np.where(correct, stopped, np.inf))
    quantiles = {}
    for q in [.5, .8, .9]:
        value = completion[int(np.ceil(q * runs)) - 1]
        quantiles[str(q)] = int(value) if np.isfinite(value) else None
    summary = {
        'scenario': scenario, 'uplift': uplift if scenario != 'null' else 0.,
        'runs': runs, 'seed': seed, 'horizon': horizon,
        'true_means': true_means.tolist(),
        'correct_stop_fraction': float(correct.mean()),
        'wrong_stop_fraction': float(((stopped > 0) & ~correct).mean()),
        'unfinished_fraction': float((stopped == 0).mean()),
        'users_for_correct_completion_quantiles': quantiles,
        'median_users_among_stopped_runs': float(np.median(stopped[stopped > 0])) if np.any(stopped > 0) else None,
        'mean_users_at_stop_or_cap': float(n.sum(axis=1).mean()),
        'mean_arm_users_at_stop_or_cap': n.mean(axis=0).tolist(),
        'winner_counts': {str(p): int((winners == i).sum()) for i, p in enumerate(prices)},
        'null_note': 'Every declared winner is a false superiority declaration when all arms tie.' if scenario == 'null' else None,
    }
    return summary, snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=500)
    parser.add_argument('--horizon', type=int, default=300000)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--uplifts', type=float, nargs='+', default=[.15])
    parser.add_argument('--output-dir', type=Path, default=Path('results/historical_15pct'))
    args = parser.parse_args()
    x = np.loadtxt(args.data, delimiter=',', skiprows=1)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)) or np.any(x < 0) or not np.any(x > 0):
        raise ValueError('Expected a finite, nonnegative, one-column user-level revenue CSV.')
    if args.runs < 1 or args.horizon < 10000 or args.horizon % 2000:
        raise ValueError('Runs must be positive; horizon must be >=10,000 and divisible by 2,000.')
    if any(not np.isfinite(u) or u <= 0 for u in args.uplifts) or len(set(args.uplifts)) != len(args.uplifts):
        raise ValueError('Uplifts must be distinct, finite positive fractions.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'source': str(args.data.resolve()),
        'historical': {'users': len(x), 'mean': float(x.mean()), 'sd': float(x.std(ddof=1)), 'zero_fraction': float((x == 0).mean())},
        'method': 'Thompson sampling; approximate Bayesian bootstrap normal posterior for mean revenue; 2,000 warm-up users per arm; 2,000-user mature batches; 5% minimum exploration per arm; stop at P(arm is best)>=95%; 1,024 screening and 8,192 confirmation posterior draws.',
        'limitations': 'Normal approximation to Bayesian bootstrap, not exact posterior sampling. Empirical stationary IID outcomes. All batch outcomes mature before the next allocation. Prices 70/80/90 share the historical distribution; price 100 has the specified constructed lift. This does not estimate actual price elasticity. Repeated posterior stopping has no guaranteed frequentist 5% error control.',
        'scenarios': [],
    }
    all_history = []
    scenarios = [(u, s, args.seed + i) for u in args.uplifts
                 for i, s in enumerate(['spend_lift', 'payer_lift'])]
    scenarios.append((0., 'null', args.seed + 2))
    for uplift, scenario, seed in scenarios:
        summary, history = simulate(x, scenario, args.runs, args.horizon, seed, uplift)
        report['scenarios'].append(summary)
        all_history += [dict(scenario=scenario, uplift=uplift, **row) for row in history]
        print(json.dumps(summary), flush=True)
        (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2))
    with (args.output_dir / 'progress.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_history[0]))
        writer.writeheader()
        writer.writerows(all_history)
    lines = [
        '# Historical revenue Bayesian bandit comparison', '',
        f'{args.runs:,} simulations per scenario. All user counts are totals across prices 70, 80, 90, and 100.', '',
        '| Lift | Mechanism | 50% correct completion | 80% correct completion | 90% correct completion | Wrong winner at cap | Unfinished at cap |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for result in report['scenarios']:
        if result['scenario'] == 'null':
            continue
        quantiles = result['users_for_correct_completion_quantiles']
        counts = [f"{quantiles[q]:,}" if quantiles[q] is not None else f'>{args.horizon:,}' for q in ['0.5', '0.8', '0.9']]
        label = 'Higher spending' if result['scenario'] == 'spend_lift' else 'More payers'
        lines.append(f"| {result['uplift']:.0%} | {label} | {' | '.join(counts)} | {result['wrong_stop_fraction']:.2%} | {result['unfinished_fraction']:.2%} |")
    null = report['scenarios'][-1]
    lines += ['', '## Decision rule and assumptions', '', report['method'], '',
              f"With no lift, {null['wrong_stop_fraction']:.2%} of runs falsely declared a winner by {args.horizon:,} users.", '',
              'Completion means stopping with price 100 as best, not establishing that its lift exceeds the scenario percentage. Wrong and unfinished runs count as failures. Quantiles have Monte Carlo uncertainty.', '',
              report['limitations'], '',
              'All batch outcomes are observed before the next allocation. Calendar duration and continuous enrollment with delayed feedback are not simulated. The empirical distribution is treated as known; historical-distribution uncertainty is excluded.', '',
              'The same seeds are reused across uplift scenarios for comparability. Full parameters and results are in summary.json; milestone probabilities are in progress.csv.', '']
    (args.output_dir / 'comparison.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
