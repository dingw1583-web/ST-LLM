import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy import stats


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    previous = 0.0
    for rank,index in enumerate(order):
        previous = max(previous, min(1.0, (len(pvalues)-rank)*pvalues[index]))
        adjusted[index] = previous
    return adjusted.tolist()


def read_runs(root, split):
    result = {}
    statuses = None
    if (Path(root)/'runs.csv').exists():
        with (Path(root)/'runs.csv').open(newline='',encoding='utf-8-sig') as f:
            statuses = {int(row['seed']):row['status'] for row in csv.DictReader(f)}
    for path in sorted(Path(root).glob('seed_*/'+split+'.json')):
        row = json.loads(path.read_text())
        if row['split'] != split:
            raise ValueError('Result split differs from requested split')
        if statuses is not None and statuses.get(row['seed']) != 'completed':
            raise ValueError('Result belongs to a failed/unregistered run')
        if row['seed'] in result:
            raise ValueError('Duplicate seed')
        result[row['seed']] = row
    return result


def compare(a,b,metric):
    if set(a)!=set(b) or len(a)<2:
        raise ValueError('Paired comparison requires identical sets of at least two seeds')
    differences = []
    for seed in sorted(a):
        x,y=a[seed],b[seed]
        for field in ('dataset','split','sample_ids','labels','train_sample_ids','validation_sample_ids','sample_group_ids','train_group_ids','validation_group_ids'):
            if x[field]!=y[field]:
                raise ValueError(f'Unpaired data for seed {seed}: {field}')
        differences.append(x['metrics'][metric]-y['metrics'][metric])
    d=np.asarray(differences,dtype=float)
    if not np.isfinite(d).all():
        raise ValueError('Nonfinite metrics')
    mean=float(d.mean()); se=float(d.std(ddof=1)/np.sqrt(len(d)))
    p= (1.0 if mean==0 else 0.0) if se==0 else float(stats.ttest_1samp(d,0).pvalue)
    margin=float(stats.t.ppf(.975,len(d)-1)*se)
    return dict(n=len(d),seeds=sorted(a),mean_difference=mean,ci95=[mean-margin,mean+margin],p_value=p)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ours')
    parser.add_argument('--comparisons',type=Path,help='JSON list of ours,baseline,split objects; all rows share one Holm family')
    parser.add_argument('--baseline',nargs='+')
    parser.add_argument('--split',default='test')
    parser.add_argument('--metrics',nargs='+',default=['accuracy','macro_f1'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.comparisons:
        jobs=json.loads(args.comparisons.read_text())
    elif args.ours and args.baseline:
        jobs=[dict(ours=args.ours,baseline=baseline,split=args.split) for baseline in args.baseline]
    else:
        parser.error('Supply --comparisons or both --ours and --baseline')
    if not jobs:
        raise ValueError('No comparisons supplied')
    rows=[]
    for job in jobs:
        ours=read_runs(job['ours'],job.get('split','test'))
        other=read_runs(job['baseline'],job.get('split','test'))
        for metric in args.metrics:
            rows.append(dict(ours=job['ours'],baseline=job['baseline'],split=job.get('split','test'),metric=metric,**compare(ours,other,metric)))
    for row,adjusted in zip(rows,holm([r['p_value'] for r in rows])):
        row['p_holm']=adjusted
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'family':'All baseline/metric comparisons in this invocation','results':rows},indent=2))


if __name__=='__main__':
    main()
