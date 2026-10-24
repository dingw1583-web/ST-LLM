import argparse
import csv
import json
from pathlib import Path
import numpy as np
from config import get_config, DATA_ROOT, DATASET_CONFIGS
from paired_data import paired_segments, file_digest, temporal_rows


def write_split(header_path, graph_path, records, cfg):
    import dgl
    import torch
    from utils import construct_graph
    if not records:
        raise ValueError('Empty split: ' + str(header_path))
    Path(header_path).parent.mkdir(parents=True, exist_ok=True)
    ids = [r['sample_id'] for r in records]
    graphs = [construct_graph(bytes=packet, w_size=cfg.PMI_WINDOW_SIZE, k=1)
              for r in records for packet in r['payload']]
    labels = np.array([r['label'] for r in records], dtype=np.int64)
    np.savez_compressed(header_path, data=np.stack([r['header'] for r in records]), label=labels, sample_ids=np.array(ids), group_ids=np.array([r["source_sha256"] for r in records]))
    dgl.save_graphs(str(graph_path), graphs, {'glabel': torch.tensor(labels)})
    Path(str(graph_path) + '.ids.json').write_text(json.dumps(ids), encoding='utf-8')


def prepare(args):
    cfg = get_config(args.dataset)
    ledger, splits = [], {}
    def check_sources():
        assignments = {}
        for r in ledger:
            assignments.setdefault(r['source_sha256'], set()).add(r['split'])
        labels_by_source = {}
        for r in ledger:
            labels_by_source.setdefault(r['source_sha256'], set()).add(r['label'])
        if any(len(labels)>1 for labels in labels_by_source.values()):
            raise ValueError('Identical source bytes have conflicting class labels')
        if any(len(groups)>1 for groups in assignments.values()):
            raise ValueError('Source content crosses splits; adjust manifest or sample boundary before training')
    if args.dataset == 'cesnet-quic22':
        if not args.manifest:
            raise ValueError('CESNET requires an explicit --manifest CSV')
        manifest = Path(args.manifest).resolve()
        with manifest.open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        mapping = temporal_rows(rows)
        if len(mapping) != cfg.NUM_CLASSES:
            raise ValueError(f'Expected {cfg.NUM_CLASSES} Week-1 classes; found {len(mapping)}. Supply the documented 75-class subset.')
        for week in range(1, 5):
            splits['week' + str(week)] = []
        for row in sorted(rows, key=lambda r: (int(r['week']), r['label'], r['flow_id'])):
            path = (manifest.parent / row['path']).resolve()
            digest = file_digest(path)
            group = 'week' + str(int(row['week']))
            for window, h, y, indices in paired_segments(path, cfg):
                record = dict(sample_id=row['flow_id'] + ':' + str(window), source_flow_id=row['flow_id'], source=str(path), source_sha256=digest, label=mapping[row['label']], split=group, packet_indices=indices)
                ledger.append(record)
                splits[group].append(dict(record, header=h, payload=y))
        base = DATA_ROOT / 'CESNET-QUIC22'
        base.mkdir(parents=True, exist_ok=True)
        (base/'labels.json').write_text(json.dumps(mapping, indent=2))
        check_sources()
        for split, records in splits.items():
            write_split(base/(split+'.npz'), base/(split+'.bin'), records, cfg)
    else:
        splits = {'train_pool': [], 'test': []}
        for label, folder in cfg.DIR_PATH_DICT.items():
            records = []
            for path in sorted(Path(folder).glob('*.npz')):
                digest = file_digest(path)
                source_id = str(label) + '/' + path.name
                for window, h, y, indices in paired_segments(path, cfg, temporal=args.dataset == 'iscx-tor'):
                    record = dict(sample_id=source_id+':'+str(window), source_flow_id=source_id, source=str(path.resolve()), source_sha256=digest, label=label, packet_indices=indices)
                    records.append(dict(record, header=h, payload=y))
            records = records[:cfg.MAX_SEG_PER_CLASS]
            cut = len(records)//10 + 1
            for split, subset in (('test', records[:cut]), ('train_pool', records[cut:])):
                for r in subset:
                    r['split'] = split
                    ledger.append({k:v for k,v in r.items() if k not in ('header', 'payload')})
                splits[split].extend(subset)
        check_sources()
        write_split(cfg.HEADER_TRAIN_DATA, cfg.TRAIN_GRAPH_DATA, splits['train_pool'], cfg)
        write_split(cfg.HEADER_TEST_DATA, cfg.TEST_GRAPH_DATA, splits['test'], cfg)
        base = Path(cfg.HEADER_TRAIN_DATA).parent
    (base/'split_manifest.json').write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    print({split:len(records) for split,records in splits.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=tuple(DATASET_CONFIGS))
    parser.add_argument('--manifest', help='CESNET CSV relative paths resolve from this file')
    prepare(parser.parse_args())
