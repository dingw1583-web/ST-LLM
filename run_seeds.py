import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from config import DATASET_CONFIGS, get_config, CHECKPOINT_ROOT


def build_command(dataset, cuda, seed, checkpoint, num_workers):
    return [sys.executable, str(Path(__file__).with_name('train.py')),
            '--dataset', dataset, '--cuda', str(cuda), '--seed', str(seed),
            '--save_path', str(checkpoint), '--num_workers', str(num_workers)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=tuple(DATASET_CONFIGS))
    parser.add_argument('--cuda', required=True)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    seeds = args.seeds or get_config(args.dataset).SEEDS
    if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        parser.error('Seeds must be unique integers in [0, 2**32)')
    output = (args.output or CHECKPOINT_ROOT/args.dataset/'seed_runs').resolve()
    rows = []
    failed = False
    for seed in seeds:
        directory = output/f'seed_{seed}'
        checkpoint = directory/'model.pth'
        command = build_command(args.dataset, args.cuda, seed, checkpoint, args.num_workers)
        print(' '.join(command), flush=True)
        if args.dry_run:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        metadata = checkpoint.with_suffix('.run.json')
        metadata.unlink(missing_ok=True)
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        with (directory/'train.log').open('w', encoding='utf-8') as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = json.loads(metadata.read_text()) if process.returncode == 0 and metadata.exists() else {}
        success = process.returncode == 0 and record.get('seed') == seed and 'best_validation_accuracy' in record
        failed |= not success
        rows.append({'seed': seed, 'status': 'completed' if success else 'failed',
                     'returncode': process.returncode,
                     'best_validation_accuracy': record.get('best_validation_accuracy') if success else None,
                     'checkpoint': str(checkpoint), 'log': str(directory/'train.log')})
        with (output/'runs.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if not args.dry_run:
        values = [r['best_validation_accuracy'] for r in rows if r['status'] == 'completed']
        (output/'summary.json').write_text(json.dumps({
            'requested_seeds': list(seeds), 'successful_runs': len(values),
            'mean_best_validation_accuracy': statistics.mean(values) if values else None,
            'sample_std_best_validation_accuracy': statistics.stdev(values) if len(values)>1 else None,
            'scope': 'Best validation accuracies, not test-set results. Each seed controls initialization, split and loader order.'
        }, indent=2), encoding='utf-8')
    return 2 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
