"""Aggregate NEW training outputs; never overwrites curated paper summaries."""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output).resolve()
    if 'paper_summary' in output.parts:
        p.error('Choose a new-run output location, not paper_summary')
    if output.exists():
        p.error('Refusing to overwrite existing summary')
    groups = defaultdict(list)
    for path in sorted(Path(args.input_root).rglob('metrics.csv')):
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        run = rows[0]
        key = (path.parent.parent.name, run['method'], run['K'], run['world_size'], run['model'])
        groups[key].append((run['seed'], len(rows),
                            max(float(r['test_acc']) for r in rows),
                            sum(float(r['epoch_time']) for r in rows)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['experiment', 'method', 'K', 'world_size', 'model', 'seeds',
                         'epochs_per_run', 'peak_accuracy_mean', 'peak_accuracy_sample_sd',
                         'training_seconds_mean', 'training_seconds_sample_sd'])
        for key, runs in sorted(groups.items()):
            seeds = [r[0] for r in runs]
            if len(set(seeds)) != len(seeds):
                raise ValueError('Duplicate seeds in group: ' + str(key))
            accuracy = [r[2] for r in runs]
            times = [r[3] for r in runs]
            writer.writerow([*key, ';'.join(seeds), ';'.join(str(r[1]) for r in runs),
                             statistics.mean(accuracy), statistics.stdev(accuracy) if len(runs)>1 else '',
                             statistics.mean(times), statistics.stdev(times) if len(runs)>1 else ''])


if __name__ == '__main__':
    main()
