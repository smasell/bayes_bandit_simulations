"""Thompson sampling for revenue per visitor.

Install: pip install numpy
Run: python price_bandit.py
Custom: python price_bandit.py --rates .20 .17 .15 .10 --runs 300 --visitors 50000
Outputs: summary.json, convergence.csv, convergence.svg next to this script.
Assumptions: independent visitors, stationary conversions, immediate binary feedback.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def simulate(prices, rates, runs=300, visitors=50000, window=1000,
             threshold=.9, seed=42):
    prices, rates = np.asarray(prices, float), np.asarray(rates, float)
    if prices.shape != rates.shape or prices.ndim != 1 or len(prices) < 2:
        raise ValueError('Prices and rates must be equal-length vectors.')
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError('Prices must be finite and positive.')
    if not np.all(np.isfinite(rates)) or np.any((rates < 0) | (rates > 1)):
        raise ValueError('Rates must be between 0 and 1.')
    if runs < 1 or not 1 <= window <= visitors or not 0 < threshold <= 1:
        raise ValueError('Invalid simulation parameters.')
    expected = prices * rates
    best_mask = np.isclose(expected, expected.max(), rtol=1e-10, atol=1e-12)
    if best_mask.all():
        raise ValueError('All prices are equally good; convergence is undefined.')
    # Independent random streams for decisions and the simulated environment.
    policy_seed, env_seed = np.random.SeedSequence(seed).spawn(2)
    policy = np.random.default_rng(policy_seed)
    environment = np.random.default_rng(env_seed)
    alpha = np.ones((runs, len(prices)))
    beta = np.ones_like(alpha)
    rows = np.arange(runs)
    recent = np.zeros((window, runs), dtype=bool)
    recent_count = np.zeros(runs, dtype=int)
    first_hit = np.zeros(runs, dtype=int)
    revenue = np.zeros(runs)
    regret = np.zeros(runs)
    history = []
    check_every = max(1, window // 10)

    for t in range(1, visitors + 1):
        # Crucial: sample REVENUE, not just probability of purchase.
        arm = np.argmax(prices * policy.beta(alpha, beta), axis=1)
        purchase = environment.random(runs) < rates[arm]
        alpha[rows, arm] += purchase
        beta[rows, arm] += ~purchase
        revenue += prices[arm] * purchase
        regret += expected.max() - expected[arm]

        slot = (t - 1) % window
        recent_count -= recent[slot]
        recent[slot] = best_mask[arm]
        recent_count += recent[slot]
        if t >= window:
            reached = recent_count >= int(np.ceil(threshold * window))
            first_hit[(first_hit == 0) & reached] = t
            if t % check_every == 0 or t == visitors:
                shares = recent_count / window
                estimated = prices * alpha / (alpha + beta)
                recommended = estimated.argmax(axis=1)
                history.append({
                    'visitors': t,
                    'mean_optimal_share_last_window': float(shares.mean()),
                    'share_p10': float(np.quantile(shares, .1)),
                    'share_p90': float(np.quantile(shares, .9)),
                    'fraction_currently_above_threshold': float(reached.mean()),
                    'fraction_ever_above_threshold': float((first_hit > 0).mean()),
                    'fraction_correct_recommendation': float(best_mask[recommended].mean()),
                })

    # Unreached runs stay right-censored. Do not drop them from quantiles.
    ordered = np.sort(np.where(first_hit > 0, first_hit, np.inf))
    quantiles = {}
    for q in (.1, .5, .9):
        value = ordered[max(0, int(np.ceil(q * runs)) - 1)]
        quantiles[str(q)] = int(value) if np.isfinite(value) else None
    summary = {
        'prices': prices.tolist(), 'assumed_purchase_rates': rates.tolist(),
        'true_revenue_per_visitor': expected.tolist(),
        'optimal_prices': prices[best_mask].tolist(),
        'runs': runs, 'visitors_per_run': visitors, 'seed': seed,
        'prior': 'Independent Beta(1, 1) for each conversion rate',
        'window': window, 'threshold': threshold,
        'first_hit_quantiles_visitors': quantiles,
        'not_reached_fraction': float((first_hit == 0).mean()),
        'final_metrics': history[-1],
        'mean_realized_revenue_per_visitor': float(revenue.mean() / visitors),
        'mean_cumulative_expected_regret': float(regret.mean()),
        'note': 'First threshold crossing is not permanent convergence or 90% posterior confidence. Null quantiles exceed the simulated horizon.',
    }
    return summary, history


def write_svg(history, destination):
    # Standalone static chart, no plotting dependency required.
    width, height = 900, 430
    left, top, right, bottom = 75, 55, 865, 350
    n = history[-1]['visitors']
    x = lambda v: left + (right-left) * v / n
    y = lambda v: bottom - (bottom-top) * v
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>Price bandit convergence over repeated simulations</title>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,sans-serif" font-size="14" fill="#17212b">',
             '<text x="75" y="27" font-size="19">Как быстро бандит находит лучшую цену</text>']
    for p in (0, .25, .5, .75, 1):
        parts.append(f'<path d="M {left} {y(p)} H {right}" stroke="#dce1e5"/><text x="62" y="{y(p)+5}" text-anchor="end">{p:.0%}</text>')
    for t in np.linspace(0, n, 6):
        parts.append(f'<text x="{x(t)}" y="374" text-anchor="middle">{int(t):,}</text>')
    parts.append('<text x="465" y="401" text-anchor="middle">Посетителей в каждом эксперименте</text>')
    for key, color, label, offset in [
        ('mean_optimal_share_last_window', '#2266bb', 'Средняя доля лучшей цены за последнее окно', 0),
        ('fraction_currently_above_threshold', '#b25213', 'Доля запусков, сейчас достигших порога', 1),
    ]:
        points = ' '.join(f'{x(h["visitors"]):.1f},{y(h[key]):.1f}' for h in history)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        lx = 80 + offset*445
        parts.append(f'<path d="M {lx} 421 h 22" stroke="{color}" stroke-width="3"/><text x="{lx+28}" y="425" font-size="12">{label}</text>')
    parts.append('</g></svg>')
    destination.write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=float, nargs='+', default=[60, 80, 100, 120])
    parser.add_argument('--rates', type=float, nargs='+', default=[.20, .17, .15, .10])
    parser.add_argument('--runs', type=int, default=300)
    parser.add_argument('--visitors', type=int, default=50000)
    parser.add_argument('--window', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    summary, history = simulate(args.prices, args.rates, args.runs, args.visitors,
                                args.window, args.threshold, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (args.output_dir / 'convergence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_svg(history, args.output_dir / 'convergence.svg')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
