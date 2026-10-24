import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def metrics(labels, predictions):
    result = {'accuracy': float(accuracy_score(labels, predictions))}
    for average in ('macro', 'weighted'):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=average, zero_division=0)
        result.update({average+'_precision': float(precision), average+'_recall': float(recall), average+'_f1': float(f1)})
    return result


def run_evaluation(model, loader, device):
    import torch
    labels, predictions = [], []
    model.eval()
    with torch.inference_mode():
        for header, payload, target in loader:
            logits = model(header.to(device), payload.to(device))
            labels.extend(target.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    if not labels:
        raise ValueError('Empty evaluation split')
    return labels, predictions


def load_model(cfg, checkpoint, device, variant=None):
    import torch
    from model import MixTemporalGNN
    path = Path(checkpoint)
    metadata = json.loads(path.with_suffix('.run.json').read_text())
    recorded_variant = metadata['settings']['variant']
    if variant is not None and variant != recorded_variant:
        raise ValueError('Requested variant differs from checkpoint metadata')
    for key in ('NUM_CLASSES','FLOW_PAD_TRUNC_LENGTH','GPT_LAYERS','HDF_U','EMBEDDING_SIZE','H_FEATS'):
        if metadata['settings'][key] != getattr(cfg,key):
            raise ValueError('Checkpoint/config mismatch: '+key)
    model = MixTemporalGNN(num_classes=cfg.NUM_CLASSES, embedding_size=cfg.EMBEDDING_SIZE, h_feats=cfg.H_FEATS,
                           dropout=cfg.DROPOUT, classifier_dropout=cfg.CLASSIFIER_DROPOUT,
                           gpt_layers=cfg.GPT_LAYERS, hdf_u=cfg.HDF_U, variant=recorded_variant).to(device)
    model.load_state_dict(torch.load(path, map_location=device), strict=True)
    model.eval()
    return model, metadata
