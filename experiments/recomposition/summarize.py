#!/usr/bin/env python3
"""Summarize raw measured repetitions. Rust already omits the warmups.

Proof savings are per pair: ratio = factored / production for the same
(process, trial). The interval is a seeded percentile bootstrap over pairs.
"""
import argparse
import json
import random
import statistics
from pathlib import Path

BOOTSTRAP = 10_000

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results')
root = parser.parse_args().results


def stats(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'n': len(values)}


def bootstrap_median(values):
    rng = random.Random(0)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP))
    return [medians[int(0.025 * BOOTSTRAP)], medians[int(0.975 * BOOTSTRAP) - 1]]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize_proofs(rows):
    pairs = {}
    for row in rows:
        pairs.setdefault((row.get('process', 0), row['trial']), {})[row['variant']] = row
    pairs = [pair for pair in pairs.values() if len(pair) == 2]
    ratios = [pair['factor']['seconds'] / pair['production']['seconds'] for pair in pairs]
    savings = [pair['production']['seconds'] - pair['factor']['seconds'] for pair in pairs]
    low, high = bootstrap_median(ratios)
    result = {
        'production': stats([pair['production']['seconds'] for pair in pairs]),
        'factor': stats([pair['factor']['seconds'] for pair in pairs]),
        'paired_savings': stats(savings),
        'ratio': stats(ratios),
        'saving_percent': {'median': 100 * (1 - statistics.median(ratios)),
                           'ci95': [100 * (1 - high), 100 * (1 - low)]},
        'setup_seconds': stats([row['setup_seconds'] for row in rows]),
    }
    by_process = {}
    for pair, ratio in zip(pairs, ratios):
        by_process.setdefault(pair['factor'].get('process', 0), []).append(ratio)
    result['ratio_by_process'] = {process: stats(values) for process, values in sorted(by_process.items())}
    if all('phases' in pair['factor'] for pair in pairs):
        names = sorted({name for pair in pairs for row in pair.values() for name in row['phases']})
        phases = {}
        for name in names:
            def field(variant, key):
                return [pair[variant]['phases'].get(name, {key: 0})[key] for pair in pairs]
            phases[name] = {
                'production_seconds': stats(field('production', 'seconds')),
                'factor_seconds': stats(field('factor', 'seconds')),
                'paired_saving_seconds': stats([p - f for p, f in
                                                zip(field('production', 'seconds'), field('factor', 'seconds'))]),
                'production_matrix_passes': statistics.median(field('production', 'matrix_passes')),
                'factor_matrix_passes': statistics.median(field('factor', 'matrix_passes')),
            }
        result['phases'] = phases
    return result


summary = {}
for step in range(1, 4):
    records = read(root / f'measure-{step}.jsonl')
    data = {'plan': next(r for r in records if r['event'] == 'plan'),
            'oracle': next(r for r in records if r['event'] == 'oracle')}
    for event in ['fresh', 'carried']:
        rows = [r for r in records if r['event'] == event]
        if rows:
            data[event] = {key: stats([r[key] for r in rows]) for key in rows[0] if key.endswith('_seconds')}
    rows = [r['timings'] for r in records if r['event'] == 'openings_split']
    data['openings'] = {key: stats([sum(r[key]) if isinstance(r[key], list) else r[key] for r in rows])
                        for key in rows[0]}
    proofs = root / f'proofs-{step}.jsonl'
    if proofs.exists():
        data['proofs'] = summarize_proofs(read(proofs))
    summary[step] = data
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({step: data.get('proofs', {}) for step, data in summary.items()}, indent=2))
