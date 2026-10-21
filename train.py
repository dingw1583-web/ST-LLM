import os, sys, argparse, contextlib, json
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from dataloader import MixTrafficFlowDataset4DGL
from model import MixTemporalGNN
from optim import GradualWarmupScheduler
from utils import show_time, set_seed, get_device, mix_collate_fn
from config import *


class EarlyStopping:

    def __init__(self, patience: int = 6):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, metric: float):
        score = -metric
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

@contextlib.contextmanager
def silent_forward(model):

    _stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = _stdout

@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for hdr, pay, labels in loader:
        hdr, pay, labels = hdr.to(device), pay.to(device), labels.to(device)
        with silent_forward(model):
            logits = model(hdr, pay)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def _build_subset_ds(ds, only_label=None):

    if only_label is None:
        return ds
    idx = [i for i in range(len(ds)) if ds[i][2] == only_label]
    return torch.utils.data.Subset(ds, idx)

def _assert_test_not_in_train(cfg, args):

    if not args.test_header or not args.test_graph:
        return
    train_set = {Path(cfg.HEADER_TRAIN_DATA).resolve(),
                 Path(cfg.TRAIN_GRAPH_DATA).resolve()}
    test_set  = {Path(args.test_header).resolve(),
                 Path(args.test_graph).resolve()}
    inter = train_set & test_set
    if inter:
        raise ValueError(f"🚨 测试集文件与训练集重复: {list(inter)}")

def train(config, device, args):
    _assert_test_not_in_train(config, args)


    full_ds = MixTrafficFlowDataset4DGL(header_path=config.HEADER_TRAIN_DATA,
                                        payload_path=config.TRAIN_GRAPH_DATA)


    full_ds = _build_subset_ds(full_ds, args.only_label)
    if args.only_label is not None:
        print(f"📌 模拟事故现场：仅使用标签 {args.only_label}，样本数 {len(full_ds)}")


    n = len(full_ds)
    train_len = int(0.5 * n)
    val_len = n - train_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(args.seed))

    g = torch.Generator().manual_seed(args.seed)
    train_loader = GraphDataLoader(train_ds, batch_size=config.BATCH_SIZE,
                                   shuffle=True, collate_fn=mix_collate_fn,
                                   num_workers=config.NUM_WORKERS, generator=g)
    val_loader = GraphDataLoader(val_ds, batch_size=config.BATCH_SIZE,
                                 shuffle=False, collate_fn=mix_collate_fn,
                                 num_workers=config.NUM_WORKERS, generator=g)


    model = MixTemporalGNN(num_classes=config.NUM_CLASSES,
                           embedding_size=config.EMBEDDING_SIZE,
                           h_feats=config.H_FEATS,
                           dropout=config.DROPOUT,
                           downstream_dropout=config.DOWNSTREAM_DROPOUT).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = len(train_loader) * config.MAX_EPOCH
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, total_steps=steps)

    best_val_acc, best_state = -1.0, None


    for epoch in range(1, config.MAX_EPOCH + 1):
        model.train(); epoch_loss = correct = total = 0
        for hdr, pay, labels in train_loader:
            hdr, pay, labels = hdr.to(device), pay.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with silent_forward(model):
                logits = model(hdr, pay)
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step(); scheduler.step()
            epoch_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        tr_loss, tr_acc = epoch_loss / total, correct / total
        val_loss, val_acc, v_preds, v_labels = evaluate(model, val_loader, criterion, device)

        cm = confusion_matrix(v_labels, v_preds, labels=list(range(config.NUM_CLASSES)))
        print(f"\nConfusion Matrix (Epoch {epoch}):\n{cm}")
        print(f"{show_time()} │ Epoch {epoch:03d}/{config.MAX_EPOCH} │ "
              f"train {tr_loss:.4f}/{tr_acc:.3f} │ val {val_loss:.4f}/{val_acc:.3f} │ "
              f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


    save_path = Path(args.save_path or config.MIX_MODEL_CHECKPOINT)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_path)
    save_path.with_suffix(".run.json").write_text(json.dumps({
        "dataset": args.dataset, "seed": args.seed,
        "best_validation_accuracy": best_val_acc,
        "checkpoint": str(save_path), "train_indices": train_ds.indices,
        "validation_indices": val_ds.indices,
        "scope": "Validation accuracy; not test accuracy. Indices refer to the selected dataset."
    }, indent=2), encoding="utf-8")
    print("✅ Best model saved to", save_path)


    if args.test_header and args.test_graph:
        test_path_h, test_path_g = Path(args.test_header), Path(args.test_graph)
        if not test_path_h.exists() or not test_path_g.exists():
            print("⚠️  测试集路径不存在，跳过测试阶段"); return
        try:
            test_ds = MixTrafficFlowDataset4DGL(header_path=str(test_path_h), payload_path=str(test_path_g))
        except Exception as e:
            print("⚠️  加载测试集失败：", e); return
        test_loader = GraphDataLoader(test_ds, batch_size=config.BATCH_SIZE,
                                      shuffle=False, collate_fn=mix_collate_fn,
                                      num_workers=config.NUM_WORKERS, generator=g)
        model.load_state_dict(best_state); model.to(device); model.eval()
        t_loss, t_acc, t_preds, t_labels = evaluate(model, test_loader, criterion, device)
        print(f"\n🔥 Test Accuracy: {t_acc:.3f}")
        print("Test Confusion Matrix:\n", confusion_matrix(t_labels, t_preds,
                                                          labels=list(range(config.NUM_CLASSES))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["iscx-vpn", "iscx-nonvpn", "iscx-tor", "iscx-nontor", "ustc-tfc2016"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cuda", required=True, help="GPU index, e.g. 0")
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default="", help="自定义模型保存路径")
    parser.add_argument("--only_label", type=int, default=None, help="仅保留该类数据进行训练")
    parser.add_argument("--test_header", type=str, default="", help="测试集 header 文件路径")
    parser.add_argument("--test_graph", type=str, default="", help="测试集 graph 文件路径")
    args = parser.parse_args()


    config = get_config(args.dataset)

    config.NUM_WORKERS = args.num_workers if args.num_workers >= 0 else config.NUM_WORKERS

    device = get_device(index=args.cuda)
    args.seed = config.SEED if args.seed is None else args.seed
    config.SEED = args.seed
    set_seed(args.seed)

    train(config, device, args)
