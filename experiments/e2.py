"""e2: fixed-budget online replay of RouterBench answers and recorded costs.

python e2.py --quick
python e2.py --queries 20 --horizon 2000 --runs 10 --noise-sigma 0.05
python e2.py --queries 100 --algorithms OMBSC CLCB-SC LFU --benchmark greedy
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from algorithms.registry import ALGORITHMS, DEFAULT_ALGORITHMS, create_policy
from algorithms.lazy_density_greedy import lazy_density_greedy
from evaluation import exact_budgeted_losses
from objective import cache_loss, MemoizedGain
from online import run_online_agent
from real_world import (DATA_ROOT, DATASET_REVISION, MODEL_REVISION, DEFAULT_CSV,
                        read_records, cached_embeddings, make_problem)

DEFAULT_ALPHA = (1 - np.exp(-1)) / 2


def run_experiment(*, dataset=DEFAULT_CSV, data_dir=DATA_ROOT,
                   model=None, n_queries=20, data_seed=0, eval_name=None,
                   horizon=1000, runs=5, seed=0, budget_fraction=0.25, budget=None,
                   algorithms=DEFAULT_ALGORITHMS, arrival='uniform', noise_sigma=0.05,
                   gamma=1.0, alpha=DEFAULT_ALPHA, epsilon=0.15, vsocb_growth=1.0,
                   benchmark='auto', records=None, embeddings=None, vsocb_oracle='auto',
                   ombsc_confidence_scale=1.0):
    """Truth stays in the environment; selection sees only online estimates.

    Optional records/embeddings allow replay of an already prepared workload.
    The data subset is fixed across trials; each trial changes arrivals/noise.
    """
    names = list(algorithms)
    if horizon < 2 or runs < 1 or seed < 0 or data_seed < 0 or n_queries < 2:
        raise ValueError('require horizon >= 2, runs >= 1, queries >= 2 and non-negative seeds')
    if not names or len(names) != len(set(names)) or set(names) - set(ALGORITHMS):
        raise ValueError('algorithms must be distinct known names and nonempty')
    if not np.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError('noise_sigma must be finite and non-negative')
    if not np.isfinite(ombsc_confidence_scale) or ombsc_confidence_scale <= 0:
        raise ValueError('ombsc_confidence_scale must be finite and positive')
    if not np.isfinite(gamma) or gamma < 0 or not 0 < alpha <= 1 or not 0 <= epsilon <= 1:
        raise ValueError('require finite gamma >= 0, alpha in (0,1] and epsilon in [0,1]')
    if not np.isfinite(vsocb_growth) or vsocb_growth <= 0:
        raise ValueError('vsocb_growth must be finite and positive')
    if not 0 < budget_fraction <= 1 or (budget is not None and (not np.isfinite(budget) or budget <= 0)):
        raise ValueError('require budget_fraction in (0,1] and a positive finite budget')
    if benchmark not in ('auto', 'exact', 'greedy') or arrival not in ('uniform', 'zipf'):
        raise ValueError('invalid benchmark or arrival distribution')
    if vsocb_oracle not in ('auto', 'exact', 'approx'):
        raise ValueError('vsocb_oracle must be auto, exact or approx')
    count = len(records.prompts) if records is not None else n_queries
    if count > 40 and 'VSOCB-Semantic' in names and vsocb_oracle == 'exact':
        raise ValueError('VSOCB exact additive knapsack supports at most 40 queries; '
                         'use --vsocb-oracle auto or approx for larger e2 runs')
    if benchmark == 'exact' and count > 22:
        raise ValueError('exact semantic benchmark supports at most 22 queries; use --benchmark greedy')
    if records is None:
        records = read_records(dataset, model=model, n_queries=n_queries, seed=data_seed, eval_name=eval_name)
    if embeddings is None:
        embeddings = cached_embeddings(records.prompts, data_dir)
    problem, storage, usd_scale = make_problem(records, embeddings, arrival=arrival,
                                               budget_fraction=budget_fraction, budget=budget)
    sizes = storage.query_sizes + storage.answer_sizes
    exact = benchmark == 'exact' or (benchmark == 'auto' and problem.m <= 22)
    if exact:
        reference_loss = float(exact_budgeted_losses(problem, sizes, [storage.budget], gamma=gamma)[0])
        benchmark_name = 'exact'
    else:
        reference = lazy_density_greedy(MemoizedGain(problem.p, problem.cost, problem.dist, gamma=gamma),
                                        sizes, storage.budget)
        reference_loss = cache_loss(reference, problem.p, problem.cost, problem.dist, gamma=gamma)
        benchmark_name = 'offline_greedy_reference'
    empty_loss = cache_loss(set(), problem.p, problem.cost, problem.dist, gamma=gamma)
    delta = 1 / horizon
    rounds = np.arange(1, horizon + 1)
    rows = []
    for trial in range(seed, seed + runs):
        stream = random.Random(trial).choices(range(problem.m), weights=problem.p, k=horizon)
        for name in names:
            policy = create_policy(name, seed=trial, gamma=gamma, epsilon=epsilon,
                                   vsocb_growth=vsocb_growth, noise_sigma=noise_sigma,
                                   vsocb_oracle=vsocb_oracle, ombsc_confidence_scale=ombsc_confidence_scale)
            agent = policy.make_agent(problem, storage, horizon, delta)
            result = run_online_agent(problem, storage, horizon, delta, trial,
                                      gamma=gamma, agent=agent, stream=stream, paired_noise=True)
            gap = result.cumulative_objective - rounds * reference_loss
            row = {
                'algorithm': name, 'mode': 'online', 'seed': trial,
                'budget': storage.budget, 'budget_fraction': storage.budget_fraction,
                'empty_loss': empty_loss, 'reference_loss': reference_loss,
                'average_loss': float(result.losses.mean()),
                'average_objective': float(result.cumulative_objective[-1] / horizon),
                'average_realized_serving_cost': float(result.serving_costs.mean()),
                'average_realized_objective': float(result.cumulative_realized_objective[-1] / horizon),
                'total_switching_cost': float(result.switching_costs.sum()),
                'switches': result.switches, 'service_llm_call_count': int(result.llm_calls.sum()),
                'population_entry_count': int(result.population_entries.sum()),
                'oracle_calls': result.oracle_calls, 'revealed_candidates': result.revealed_candidates,
                'vsocb_exact_oracle_calls': policy.oracle_usage['exact'] if name == 'VSOCB-Semantic' else None,
                'vsocb_approx_oracle_calls': policy.oracle_usage['approx'] if name == 'VSOCB-Semantic' else None,
                'total_selection_runtime_s': result.total_selection_runtime,
                'entry_sizes': sizes.tolist(), 'query_stream': result.query_stream.tolist(),
                'actions': result.actions, 'cache_losses': result.losses.tolist(),
                'switching_costs': result.switching_costs.tolist(),
                'realized_serving_costs': result.serving_costs.tolist(),
                'service_llm_calls': result.llm_calls.tolist(),
                'population_entries': result.population_entries.tolist(),
                'memory_used': result.memory_used.tolist(), 'cache_sizes': result.cache_sizes.tolist(),
                'served_caches': result.served_caches, 'caches': result.caches,
            }
            if exact:
                regret = result.approximation_regret(empty_loss, reference_loss, alpha)
                row.update(optimal_loss=reference_loss, alpha=float(alpha),
                           final_alpha_regret=float(regret[-1]), final_exact_regret=float(gap[-1]),
                           cumulative_alpha_regret=regret.tolist(), cumulative_exact_regret=gap.tolist())
            else:
                row.update(final_reference_gap=float(gap[-1]), cumulative_reference_gap=gap.tolist())
            rows.append(row)
        print(f'[e2] seed {trial}: {len(names)} algorithms finished', flush=True)
    return {
        'config': {
            'experiment': 'e2', 'mode': 'online', 'dataset': 'withmartian/routerbench',
            'dataset_path': str(Path(dataset).resolve()), 'dataset_revision': DATASET_REVISION,
            'dataset_format': Path(dataset).suffix.lower().removeprefix('.'),
            'model': records.model, 'n_queries': problem.m, 'data_seed': data_seed,
            'eval_name': eval_name, 'horizon': horizon, 'runs': runs, 'first_seed': seed,
            'algorithms': names, 'arrival': arrival, 'arrival_probabilities': problem.p.tolist(),
            'budget': storage.budget, 'budget_fraction': storage.budget_fraction,
            'storage_unit': 'UTF-8 bytes', 'storage': 'exact encoded prompt bytes + recorded answer bytes',
            'usd_per_cost_unit': usd_scale, 'cost_normalization': 'recorded USD / (2 * max workload USD)',
            'cost_source': 'recorded estimated API costs, not measured latency',
            'noise_sigma': float(noise_sigma),
            'ombsc_confidence_scale': float(ombsc_confidence_scale),
            'ombsc_confidence_note': 'scales only OMBSC confidence; scale < 1 is empirical and does not inherit the original confidence guarantee',
            'noise': 'C=c+Rademacher*min(noise_sigma,c,1-c); mean-preserving bounded synthetic noise',
            'noise_pairing': 'same per-query service-call-index streams across algorithms',
            'embedding': 'Xenova/all-MiniLM-L6-v2, quantized ONNX, CPU, masked mean/L2, max 256 tokens',
            'embedding_revision': MODEL_REVISION, 'distance': 'Euclidean distance / 2 on unit embeddings',
            'gamma': float(gamma), 'alpha': float(alpha) if exact else None, 'delta': delta,
            'epsilon': epsilon, 'vsocb_growth': vsocb_growth, 'benchmark': benchmark_name,
            'vsocb_oracle': vsocb_oracle,
            'vsocb_approximation': 'Appendix A complement oracle; min-knapsack 2-approximation (beta=1); not a semantic guarantee',
            'benchmark_note': ('full-Q exact optimum, evaluator only' if exact else
                               'full-Q offline greedy feasible reference; gaps are not optimal regret'),
            'loss': 'expected ideal cache loss, distinct from realized action cost',
            'switching': 'expected recorded generation cost for new entries; charged next round; no extra observations',
            'information': 'cost and answer size revealed only on fresh service; all raw truth is evaluator metadata',
            'cache_timestamps': 'served_caches[t] before service; caches[t] after update',
        },
        'workload': [dict(index=i, sample_id=records.sample_ids[i], eval_name=records.eval_names[i],
                          prompt=records.prompts[i], answer=records.answers[i],
                          cost_usd=float(records.cost_usd[i]), normalized_cost=float(problem.cost[i]),
                          query_bytes=int(records.query_bytes[i]), answer_bytes=int(records.answer_bytes[i]))
                     for i in range(problem.m)],
        'runs': rows,
    }


def save_results(payload, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for filename, obj in [('e2.json', {k: v for k, v in payload.items() if k != 'workload'}),
                          ('workload.json', payload['workload'])]:
        (out / filename).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    scalar_keys = [key for key, value in payload['runs'][0].items() if not isinstance(value, list)]
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_keys} for row in payload['runs'])
    curve_keys = [key for key in ('cache_losses', 'switching_costs', 'realized_serving_costs',
                                 'service_llm_calls', 'memory_used', 'cache_sizes',
                                 'cumulative_alpha_regret', 'cumulative_exact_regret',
                                 'cumulative_reference_gap') if key in payload['runs'][0]]
    with (out / 'curves.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['algorithm', 'seed', 'round', *curve_keys])
        writer.writeheader()
        for row in payload['runs']:
            for t in range(payload['config']['horizon']):
                writer.writerow(dict(algorithm=row['algorithm'], seed=row['seed'], round=t+1,
                                     **{key: row[key][t] for key in curve_keys}))
    plot_results(payload, out / 'e2.png')


def plot_results(payload, path):
    config = payload['config']
    exact = config['benchmark'] == 'exact'
    gap_key = 'cumulative_alpha_regret' if exact else 'cumulative_reference_gap'
    t = np.arange(1, config['horizon'] + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for name in config['algorithms']:
        rows = [row for row in payload['runs'] if row['algorithm'] == name]
        loss = np.array([row['cache_losses'] for row in rows])
        switches = np.array([row['switching_costs'] for row in rows])
        curves = [np.array([row[gap_key] for row in rows]), np.cumsum(loss + switches, axis=1) / t,
                  np.cumsum(switches, axis=1),
                  np.cumsum(np.array([row['service_llm_calls'] for row in rows]), axis=1)]
        for ax, curve in zip(axes.flat, curves):
            mean, std = curve.mean(axis=0), curve.std(axis=0)
            ax.plot(t, mean, label=name)
            ax.fill_between(t, mean - std, mean + std, alpha=0.12)
    labels = ['Cumulative alpha-regret' if exact else 'Gap to offline greedy reference',
              'Average ideal loss + switching', 'Cumulative switching cost', 'Cumulative fresh service calls']
    for ax, label in zip(axes.flat, labels):
        ax.set(xlabel='Online round', ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0, 0].axhline(0, color='gray', linewidth=0.6)
    fig.suptitle(f"e2: RouterBench, N={config['n_queries']}, B={config['budget']:.0f} bytes, "
                 f"noise={config['noise_sigma']:g}, OMBSC confidence={config.get('ombsc_confidence_scale', 1.):g}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=DATA_ROOT)
    parser.add_argument('--dataset', type=Path, help='defaults to DATA_DIR/routerbench_single_model.csv')
    parser.add_argument('--model', help='optional check of the model recorded in CSV')
    parser.add_argument('--queries', type=int, default=20)
    parser.add_argument('--data-seed', type=int, default=0)
    parser.add_argument('--eval-name', help='optional exact RouterBench eval_name filter')
    parser.add_argument('--horizon', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quick', action='store_true', help='2 seeds, 100 rounds')
    budgets = parser.add_mutually_exclusive_group()
    budgets.add_argument('--budget', type=float, help='fixed memory budget in UTF-8 bytes')
    budgets.add_argument('--budget-fraction', type=float, default=0.25)
    parser.add_argument('--arrival', choices=('uniform', 'zipf'), default='uniform')
    parser.add_argument('--noise-sigma', type=float, default=0.05, help='normalized-cost sub-Gaussian bound')
    parser.add_argument('--ombsc-confidence-scale', type=float, default=1.0,
                        help='OMBSC confidence-radius multiplier; does not change environment noise; <1 is empirical tuning')
    parser.add_argument('--gamma', type=float, default=1.0, help='semantic mismatch cost scaling')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--epsilon', type=float, default=0.15)
    parser.add_argument('--vsocb-growth', type=float, default=1.0)
    parser.add_argument('--vsocb-oracle', choices=('auto', 'exact', 'approx'), default='auto',
                        help='auto: exact up to 40 feasible candidates, complementary min-knapsack approximation above 40')
    parser.add_argument('--benchmark', choices=('auto', 'exact', 'greedy'), default='auto')
    parser.add_argument('--algorithms', nargs='+', choices=tuple(ALGORITHMS), default=list(DEFAULT_ALGORITHMS))
    parser.add_argument('--out', type=Path, default=Path('results/e2'))
    args = parser.parse_args()
    try:
        payload = run_experiment(dataset=args.dataset or args.data_dir / DEFAULT_CSV.name,
                                 data_dir=args.data_dir, model=args.model, n_queries=args.queries,
                                 data_seed=args.data_seed, eval_name=args.eval_name,
                                 horizon=100 if args.quick else args.horizon, runs=2 if args.quick else args.runs,
                                 seed=args.seed, budget=args.budget, budget_fraction=args.budget_fraction,
                                 algorithms=args.algorithms, arrival=args.arrival, noise_sigma=args.noise_sigma,
                                 gamma=args.gamma, alpha=args.alpha, epsilon=args.epsilon,
                                 vsocb_growth=args.vsocb_growth, benchmark=args.benchmark,
                                 vsocb_oracle=args.vsocb_oracle,
                                 ombsc_confidence_scale=args.ombsc_confidence_scale)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    save_results(payload, args.out)
    print(f"Benchmark: {payload['config']['benchmark']}; budget: {payload['config']['budget']:g} bytes")
    for name in args.algorithms:
        rows = [row for row in payload['runs'] if row['algorithm'] == name]
        print(f"{name:18s} avg objective={np.mean([r['average_objective'] for r in rows]):.6f}, "
              f"fresh calls={np.mean([r['service_llm_call_count'] for r in rows]):.1f}")
    print(f'Saved {args.out.resolve()}')


if __name__ == '__main__':
    main()
