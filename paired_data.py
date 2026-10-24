import hashlib
import numpy as np
from packet_input import read_packets


def paired_segments(path, cfg, temporal=False):
    f = read_packets(path)
    headers, payloads = f['header'], f['payload']
    if len(headers) != len(payloads):
        raise ValueError(f'Unpaired packet arrays: {path}')
    if temporal:
        times = np.asarray(f['time'], dtype=float)
        if len(times) != len(headers) or not len(times) or not np.isfinite(times).all() or (np.diff(times) < 0).any():
            raise ValueError(f'Invalid packet timestamps: {path}')
        windows = np.floor((times - times[0]) / 60).astype(int)
    else:
        windows = np.zeros(len(headers), dtype=int)
    records = []
    for window in np.unique(windows):
        indices = [int(i) for i in np.flatnonzero(windows == window) if len(payloads[i]) and len(headers[i])]
        if not indices or len(indices) > cfg.ANOMALOUS_FLOW_THRESHOLD:
            continue
        selected = indices[:cfg.FLOW_PAD_TRUNC_LENGTH]
        h = np.full((cfg.FLOW_PAD_TRUNC_LENGTH, cfg.HEADER_BYTE_PAD_TRUNC_LENGTH), cfg.PAD_TRUNC_DIGIT, dtype=np.int64)
        y = np.full((cfg.FLOW_PAD_TRUNC_LENGTH, cfg.BYTE_PAD_TRUNC_LENGTH), cfg.PAD_TRUNC_DIGIT, dtype=np.int64)
        for row, i in enumerate(selected):
            raw = list(headers[i])
            masked = raw if f["header_masked"] else raw[:12] + raw[24:]
            for target, values in ((h, masked), (y, list(payloads[i]))):
                values = values[:target.shape[1]]
                target[row, :len(values)] = values
        records.append((int(window), h, y, selected))
    return records


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def temporal_rows(rows):
    required = {'path', 'label', 'week', 'flow_id'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Manifest requires path,label,week,flow_id')
    seen = set()
    for r in rows:
        if int(r['week']) not in (1, 2, 3, 4) or not r['flow_id'] or r['flow_id'] in seen:
            raise ValueError('Invalid week or missing/duplicate original flow_id')
        seen.add(r['flow_id'])
    labels = sorted({r['label'] for r in rows if int(r['week']) == 1})
    if not labels or any(r['label'] not in labels for r in rows):
        raise ValueError('Validation/test classes must exist in Week 1')
    return {label: i for i, label in enumerate(labels)}


def grouped_split(group_ids, seed):
    from collections import Counter
    groups = sorted(set(group_ids))
    if len(groups)<2:
        raise ValueError('At least two independent source flows are required')
    counts=Counter(group_ids)
    chosen=set(); count=0
    for i in np.random.default_rng(seed).permutation(len(groups))[:-1]:
        group=groups[i]; chosen.add(group); count+=counts[group]
        if count>=len(group_ids)//2:
            break
    return ([i for i,g in enumerate(group_ids) if g in chosen],
            [i for i,g in enumerate(group_ids) if g not in chosen])
