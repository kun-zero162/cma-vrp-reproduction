"""Run the full model plus every single-component ablation and aggregate the results.

Produces the Table 2 comparison of the paper: one row per configuration, with the delta
against the full model. Each row is a real train + test subprocess run, so a switch that
is cosmetic rather than functional shows up as a row identical to `full`.

Run from CMAVRP:
    python tools/ablation_table.py --datasets twitter --epochs 12 --cpu
    python tools/ablation_table.py --datasets twitter weibo --seeds 3759,1 --epochs 50
    python tools/ablation_table.py --aggregate_only        # rebuild tables, no training
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # CMAVRP/
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from configs.config_setting import setting_config as config  # noqa: E402

ALL_ABLATIONS = ('use_vrf', 'use_gcl_inter', 'use_gcl_intra', 'use_cmgaf', 'use_accil',
                 'use_vsca')
# what the paper reports in Tables 1-2, in the order the columns are printed
METRICS = ('auc', 'accuracy', 'mac_f1', 'f1_fake', 'recall_real', 'precision_fake')
PY = sys.executable


def tag_for(label, seed, multi):
    """Filesystem/run-name tag; matches the tag train.py and test.py were given."""
    return f'{label}_s{seed}' if multi else label


def switches_for(label):
    """'full' -> nothing disabled; 'no_use_vsca' -> that one switch off."""
    if label == 'full':
        return ()
    if not label.startswith('no_'):
        raise SystemExit(f'--configs entries must be "full" or "no_<switch>", got {label}')
    sw = label[3:]
    if sw not in ALL_ABLATIONS:
        raise SystemExit(f'unknown switch in {label}; choose from {ALL_ABLATIONS}')
    return (sw,)


def run(cmd, log_path):
    """One subprocess; returns (ok, tail_of_output)."""
    with open(log_path, 'w', encoding='utf-8') as f:
        p = subprocess.run(cmd, cwd=_HERE, stdout=f, stderr=subprocess.STDOUT)
    with open(log_path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    return p.returncode == 0, text[-600:]


def metrics_path(dataset, backend, tag):
    return os.path.join(_HERE, 'results',
                        f'cmavrp-{dataset}-{backend}{("-" + tag) if tag else ""}',
                        'metrics.json')


def one_config(py, dataset, backend, label, tag, switches, seed, args):
    """Train + test one ablation; returns its metrics dict or None."""
    log_dir = os.path.join(_HERE, 'logs', 'ablation')
    os.makedirs(log_dir, exist_ok=True)
    # dataset in the filename: two datasets in one invocation used to overwrite
    # each other's transcripts, leaving the first table's rows unauditable.
    stem = f'{dataset}-{backend}-{tag}'
    ab = ['--ablate'] + list(switches) if switches else []
    common = ['--datasets', dataset, '--vrp_backend', backend,
              '--batch_size', str(args.batch_size), '--seed', str(seed)] + ab

    train_cmd = [py, 'train.py', '--epochs', str(args.epochs), '--early_stop',
                 str(args.early_stop), '--tag', tag] + common
    if args.cpu:
        train_cmd.append('--cpu')
    test_cmd = [py, 'test.py', '--tag', tag, '--split', 'test'] + common
    if args.cpu:
        test_cmd.append('--cpu')

    ok, tail = run(train_cmd, os.path.join(log_dir, f'train-{stem}.log'))
    if not ok:
        print(f'    !! train FAILED ({label}, seed={seed})\n{tail}')
        return None
    ok, tail = run(test_cmd, os.path.join(log_dir, f'test-{stem}.log'))
    if not ok:
        print(f'    !! test FAILED ({label}, seed={seed})\n{tail}')
        return None
    mp = metrics_path(dataset, backend, tag)
    if not os.path.isfile(mp):
        print(f'    !! no metrics.json at {mp}')
        return None
    with open(mp, encoding='utf-8') as f:
        m = json.load(f)
    m['_tag'] = tag
    return m


def fmt(values):
    """mean (sd) over seeds, as the paper's tables imply."""
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return '  -  '
    if len(vals) == 1:
        return f'{vals[0]:.4f}'
    return f'{statistics.mean(vals):.4f}±{statistics.pstdev(vals):.4f}'


def aggregate(rows, full_row):
    """rows: [{label, metric -> (mean, sd)}]; adds a delta column against full."""
    out = []
    for r in rows:
        line = {'config': r['config'], 'seeds': r['seeds']}
        for m in METRICS:
            line[m] = r[m]
            if full_row and isinstance(r[m].get('mean'), (int, float)) \
                    and isinstance(full_row[m].get('mean'), (int, float)):
                line[m + '_delta'] = r[m]['mean'] - full_row[m]['mean']
            else:
                line[m + '_delta'] = None
        out.append(line)
    return out


def write_outputs(dataset, table, out_dir, note):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f'ablation_{dataset}')
    with open(base + '.json', 'w', encoding='utf-8') as f:
        json.dump({'dataset': dataset, 'note': note, 'metrics': METRICS,
                   'rows': table}, f, indent=2)
    with open(base + '.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['config'] + [m for m in METRICS] + [f'{m}_delta' for m in METRICS])
        for r in table:
            w.writerow([r['config']] + [r[m] for m in METRICS]
                       + [r[m + '_delta'] for m in METRICS])
    with open(base + '.md', 'w', encoding='utf-8') as f:
        f.write(f'# Ablation table - {dataset}\n\n{note}\n\n')
        f.write('| config | ' + ' | '.join(METRICS) + ' |\n')
        f.write('|---|' + '---|' * len(METRICS) + '\n')
        for r in table:
            cells = []
            for m in METRICS:
                d = r[m + '_delta']
                # the delta is the number the ablation section is actually about
                txt = f"{r[m]['mean']:.4f}" if isinstance(r[m].get('mean'), (int, float)) else '-'
                if r[m].get('n', 0) > 1:
                    txt = f"{txt} ± {r[m]['sd']:.4f}"
                cells.append(txt + (f' ({d:+.4f})' if d is not None and abs(d) > 1e-9 else ''))
            f.write(f"| {r['config']} | " + ' | '.join(cells) + ' |\n')
    return base + '.md'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['twitter'],
                   choices=['twitter', 'pheme', 'weibo'])
    p.add_argument('--vrp_backend', default=config.vrp_backend, choices=['mock', 'hf', 'agent'])
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--early_stop', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=config.batch_size)
    p.add_argument('--seeds', default=str(config.seed),
                   help='comma separated, e.g. 3759,1,2 (paper reports single runs)')
    p.add_argument('--configs', nargs='*', default=None,
                   help='subset of full/no_<switch>; switches: ' + '/'.join(ALL_ABLATIONS))
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--reuse', action='store_true',
                   help='reuse existing metrics.json instead of re-running')
    args = p.parse_args()
    args.seeds_multi = ',' in args.seeds
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    labels = args.configs or (['full'] + [f'no_{a}' for a in ALL_ABLATIONS])

    note = ('**Warning:** generated on this machine\'s stand-in data/backends, so the '
            'absolute numbers are plumbing evidence, not a reproduction of the paper\'s '
            'Table 2. What this *does* establish is that every switch changes what is '
            'trained and evaluated, and by how much on this configuration.')

    for dataset in args.datasets:
        print(f'=== {dataset} ===')
        rows = []
        for label in labels:
            switches = switches_for(label)
            per_seed = []
            for seed in seeds:
                tag = tag_for(label, seed, args.seeds_multi)
                mp = metrics_path(dataset, args.vrp_backend, tag)
                if args.reuse and os.path.isfile(mp):
                    with open(mp, encoding='utf-8') as f:
                        m = json.load(f)
                    print(f'  [reuse] {label} seed={seed}')
                else:
                    print(f'  [run] {label} seed={seed} ...', flush=True)
                    m = one_config(PY, dataset, args.vrp_backend, label, tag, switches,
                                   seed, args)
                    if m is None:
                        continue
                per_seed.append(m)
            if not per_seed:
                rows.append({'config': label, 'seeds': 0,
                             **{m: {'mean': None, 'sd': None, 'n': 0} for m in METRICS}})
                continue
            row = {'config': label, 'seeds': len(per_seed)}
            for m in METRICS:
                vs = [x.get(m) for x in per_seed if isinstance(x.get(m), (int, float))]
                row[m] = {'mean': statistics.mean(vs) if vs else None,
                          'sd': statistics.pstdev(vs) if len(vs) > 1 else 0.0,
                          'n': len(vs)}
            rows.append(row)
            print('    ' + '  '.join(f'{m}={fmt([row[m]["mean"]])}' for m in METRICS[:3]))

        full_row = next((r for r in rows if r['config'] == 'full'), None)
        table = aggregate(rows, full_row)
        md = write_outputs(dataset, table, os.path.join(_HERE, 'results'), note)
        print(f'\n{note}\n')
        hdr = '| config | ' + ' | '.join(METRICS) + ' |'
        print(hdr)
        print('|' + '---|' * (len(METRICS) + 1))
        for r in table:
            cells = []
            for m in METRICS:
                d = r[m + '_delta']
                txt = f"{r[m]['mean']:.4f}" if r[m]['mean'] is not None else '-'
                cells.append(txt + (f' ({d:+.3f})' if d is not None and abs(d) > 1e-9 else ''))
            print(f"| {r['config']} | " + ' | '.join(cells) + ' |')
        print(f'\n[saved] {md}')


if __name__ == '__main__':
    main()
