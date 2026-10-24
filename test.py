import argparse
import json
from pathlib import Path
from config import get_config, DATASET_CONFIGS
from evaluation import load_model, run_evaluation, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=tuple(DATASET_CONFIGS))
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--cuda', required=True)
    parser.add_argument('--split', choices=('test','week2','week3','week4'), default='test')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from dgl.dataloading import GraphDataLoader
    from dataloader import MixTrafficFlowDataset4DGL
    from utils import set_seed, get_device, mix_collate_fn
    cfg = get_config(args.dataset)
    set_seed(args.seed)
    device = get_device(index=args.cuda)
    checkpoint = args.checkpoint or cfg.MIX_MODEL_CHECKPOINT
    model, metadata = load_model(cfg, checkpoint, device)
    if metadata['seed'] != args.seed:
        raise ValueError('Seed differs from checkpoint metadata')
    if args.dataset == 'cesnet-quic22':
        split = 'week3' if args.split == 'test' else args.split
        prefix = {'week2':'VAL','week3':'TEST','week4':'WEEK4'}[split]
    else:
        if args.split != 'test':
            raise ValueError('Week selection only applies to CESNET')
        split, prefix = 'test','TEST'
    ds = MixTrafficFlowDataset4DGL(getattr(cfg,prefix+'_GRAPH_DATA'),getattr(cfg,'HEADER_'+prefix+'_DATA'))
    if ds.flow_length != cfg.FLOW_PAD_TRUNC_LENGTH:
        raise ValueError("Prepared packet count differs from the checkpoint configuration")
    loader = GraphDataLoader(ds,batch_size=cfg.BATCH_SIZE,shuffle=False,collate_fn=mix_collate_fn,num_workers=cfg.NUM_WORKERS)
    labels, predictions = run_evaluation(model,loader,device)
    result = dict(dataset=args.dataset,split=split,seed=args.seed,variant=metadata['settings']['variant'],
                  checkpoint=str(checkpoint),metrics=metrics(labels,predictions),sample_ids=ds.sample_ids,sample_group_ids=ds.group_ids,
                  labels=labels,predictions=predictions,train_sample_ids=metadata['train_sample_ids'],
                  validation_sample_ids=metadata['validation_sample_ids'],train_group_ids=metadata['train_group_ids'],validation_group_ids=metadata['validation_group_ids'])
    output = args.output or Path(checkpoint).with_suffix('.'+split+'.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['metrics'],indent=2))


if __name__ == '__main__':
    main()
