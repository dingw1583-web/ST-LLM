import argparse

import torch
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import classification_report

from utils import set_seed, get_device, mix_collate_fn
from dataloader import MixTrafficFlowDataset4DGL
from model import MixTemporalGNN
from config import *


torch.autograd.set_detect_anomaly(True)


def test():
    model = MixTemporalGNN(num_classes=config.NUM_CLASSES, embedding_size=config.EMBEDDING_SIZE, h_feats=config.H_FEATS,
                           dropout=config.DROPOUT, downstream_dropout=config.DOWNSTREAM_DROPOUT).to(device)
    model.load_state_dict(torch.load(opt.checkpoint or config.MIX_MODEL_CHECKPOINT, map_location={'cuda:0': 'cuda:' + str(opt.cuda),
                                                                                'cuda:1': 'cuda:' + str(opt.cuda),
                                                                                'cuda:2': 'cuda:' + str(opt.cuda),
                                                                                'cuda:3': 'cuda:' + str(opt.cuda)}))
    model.eval()


    dataset = MixTrafficFlowDataset4DGL(header_path=config.HEADER_TEST_DATA,
                                        payload_path=config.TEST_GRAPH_DATA)

    dataloader = GraphDataLoader(dataset, batch_size=32, shuffle=False, collate_fn=mix_collate_fn,
                                 num_workers=config.NUM_WORKERS, pin_memory=False)

    label_preds = []
    label_ids = []
    with torch.no_grad():
        for header_data, payload_data, labels in dataloader:
            header_data = header_data.to(device, non_blocking=True)
            payload_data = payload_data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model(header_data, payload_data)
            pred_label = pred.argmax(1).detach().cpu().numpy()
            label_preds.extend(pred_label)
            label_ids.extend(labels.detach().cpu().numpy())

    print(classification_report(label_ids, label_preds, digits=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--cuda", type=str, help="cuda", required=True)
    opt = parser.parse_args()

    config = get_config(opt.dataset)

    device = get_device(index=opt.cuda)
    set_seed(config.SEED if opt.seed is None else opt.seed)
    test()
