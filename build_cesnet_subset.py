import argparse
import csv
import hashlib
import json
from pathlib import Path
from paired_data import temporal_rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True,help='Candidate CSV: path,label,week,flow_id')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--classes',type=Path,help='Explicit one-label-per-line list; otherwise use all Week-1 classes')
    parser.add_argument('--max-per-class-week',type=int,help='Optional cap; omitted means retain all candidates')
    parser.add_argument('--seed',type=int,default=32)
    args=parser.parse_args()
    if args.max_per_class_week is not None and args.max_per_class_week<1:
        parser.error('Cap must be positive')
    with args.input.open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    mapping=temporal_rows(rows)
    labels=sorted(set(args.classes.read_text().splitlines())) if args.classes else sorted(mapping)
    labels=[label for label in labels if label]
    if len(labels)!=75 or not set(labels).issubset(mapping):
        raise ValueError('Supply exactly 75 classes present in Week 1')
    groups={}
    for row in rows:
        if row['label'] in labels:
            groups.setdefault((row['label'],int(row['week'])),[]).append(row)
    selected=[]
    for label in labels:
        for week in range(1,5):
            candidates=groups.get((label,week),[])
            if not candidates:
                raise ValueError(f'Missing class/week: {label}/{week}')
            candidates.sort(key=lambda r:hashlib.sha256((str(args.seed)+':'+r['flow_id']).encode()).hexdigest())
            for row in candidates[:args.max_per_class_week]:
                selected.append(dict(path=str((args.input.resolve().parent/row['path']).resolve()),label=label,week=week,flow_id=row['flow_id']))
    selected.sort(key=lambda r:(r['week'],r['label'],r['flow_id']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['path','label','week','flow_id']); writer.writeheader();writer.writerows(selected)
    args.output.with_suffix('.provenance.json').write_text(json.dumps(dict(seed=args.seed,classes=labels,
        max_per_class_week=args.max_per_class_week,input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        counts_by_week={str(w):sum(r['week']==w for r in selected) for w in range(1,5)},
        note='Explicit deterministic subset recipe; not evidence of the original published subset.'),indent=2))


if __name__=='__main__':
    main()
