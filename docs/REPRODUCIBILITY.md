# Reproducibility Notes

## Model and preprocessing

The model combines a bidirectional LSTM header encoder, a GraphSAGE payload encoder, cross-gated fusion, the first six pretrained GPT-2 Transformer blocks, and a downstream bidirectional LSTM classifier. The default freezing rule trains LayerNorm parameters in earlier blocks and non-MLP parameters in the final retained block; token and position embeddings remain frozen. See `model.py` for the exact parameter selection.

Default representations use 50 packets per sample, 150 payload bytes and 40 header bytes per packet, padding value 256, and PMI window size 5. The Tor route uses time-window segmentation; other routes use per-flow segmentation. Keep header/payload sample identities, packet selections, labels, and ordering aligned: the existing branches filter packet contents separately, and the loader does not independently verify original flow identity.

## Data splits and optimization

Preprocessing retains at most `MAX_SEG_PER_CLASS` samples per class. Within each class, the first `floor(N/10) + 1` retained samples form the test set and the rest form the training pool. Preprocessing follows filesystem iteration order; preserve the input file ordering or a fixed manifest to reproduce the split. Keep related samples from the same original session in one split when preparing data.

`train.py` randomly divides the prepared training pool into equally sized training and validation subsets, differing by one sample when the count is odd. The run seed controls this split. The trainer currently uses AdamW with learning rate `3e-4`, weight decay `1e-4`, label smoothing `0.05`, and OneCycleLR. These settings are fixed in `train.py`; similarly named configuration fields do not override them. Configured gradient accumulation is not implemented. Dataset-specific batch size, dropout, epochs, and worker count are read from configuration.

## Seeds and repeated runs

`Config.SEEDS` is `(32, 42, 52, 62, 72, 82, 92, 102, 112, 122)` and is inherited by all five configurations. Dataset-specific single-run defaults remain unchanged, including VPN's 66. Select a seed with `train.py --seed` or use `run_seeds.py` for the configured list:

```bash
python run_seeds.py --dataset iscx-vpn --cuda 0 --dry-run
python run_seeds.py --dataset iscx-vpn --cuda 0
python run_seeds.py --dataset iscx-vpn --cuda 0 --seeds 32 42 --output checkpoints/custom_runs
```

Replace the dataset argument with `iscx-nonvpn`, `iscx-tor`, `iscx-nontor`, or `ustc-tfc2016` as needed. Each seed runs in an independent process and controls initialization, dropout, the train/validation split, loader randomness, global RNGs, and subprocess `PYTHONHASHSEED`. The preprocessing train/test split is not regenerated. GPU and library operations may still affect determinism.

## Outputs and evaluation

Each run writes `model.pth`, `model.run.json`, and `train.log` under `checkpoints/<dataset>/seed_runs/seed_<seed>/`. Metadata records the seed, best validation accuracy, and train/validation indices. `runs.csv` and `summary.json` summarize successful runs using validation mean and sample standard deviation; failed runs receive no performance value. Reusing an output directory overwrites its run files.

Evaluate checkpoints separately on the held-out test set:

```bash
python test.py --dataset iscx-vpn --cuda 0 --seed 32 --checkpoint checkpoints/iscx-vpn/seed_runs/seed_32/model.pth
```

The validation summary is not a test-set estimate. This source update does not introduce trained weights or new performance measurements.

## Environment

Use compatible PyTorch, DGL, CUDA, and Transformers builds. `train.py` defaults `HF_ENDPOINT` to `https://hf-mirror.com`; an explicitly set environment value takes precedence. Cache GPT-2 weights if network access is unavailable. Source checks validate syntax and configuration only; they do not validate end-to-end GPU execution.
