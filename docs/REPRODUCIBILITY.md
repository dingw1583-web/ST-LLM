# Reproducibility Notes

## Model and preprocessing

The model combines a bidirectional LSTM header encoder, a GraphSAGE payload encoder, cross-gated fusion, the first six pretrained GPT-2 Transformer blocks, temporal mean pooling, and a two-layer fully connected classifier. There is no recurrent module after GPT-2. The Bi-LSTM in the header encoder is retained. The default freezing rule trains LayerNorm parameters in earlier blocks and non-MLP parameters in the final retained block; token and position embeddings remain frozen. See `model.py` for the exact parameter selection.

Default representations use 30 packets per sample, 150 payload bytes and 40 header bytes per packet, padding value 256, and PMI window size 5. The Tor route uses time-window segmentation; other routes use per-flow segmentation. Keep header/payload sample identities, packet selections, labels, and ordering aligned: the existing branches filter packet contents separately, and the loader checks packet counts and label ordering but cannot independently verify original flow identity.

## Data splits and optimization

Preprocessing retains at most `MAX_SEG_PER_CLASS` samples per class. Within each class, the first `floor(N/10) + 1` retained samples form the test set and the rest form the training pool. Preprocessing follows filesystem iteration order; preserve the input file ordering or a fixed manifest to reproduce the split. Keep related samples from the same original session in one split when preparing data.

`train.py` randomly divides the prepared training pool into equally sized training and validation subsets, differing by one sample when the count is odd. The run seed controls this split. The trainer reads optimization and model settings from the selected dataset configuration. The reply-aligned defaults use Adam, a peak learning rate of `1e-2`, a minimum of `1e-4`, a warm-up ratio of `0.1`, 120 epochs, batch size 512, and dropout 0.2. Weight decay and label smoothing default to zero because the response table does not specify them. A linear warm-up followed by cosine decay implements the stated learning-rate range; that scheduler shape is an explicit implementation choice, not a claim about unavailable historical runs. There is one optimizer step per batch. Edit a dataset configuration to supply validated dataset-specific settings.

## Seeds and repeated runs

`Config.SEEDS` is `(32, 42, 52, 62, 72, 82, 92, 102, 112, 122)` and is inherited by all five configurations. Every dataset now defaults to seed 32, with the same ten-seed list available for repeated runs. Select a seed with `train.py --seed` or use `run_seeds.py` for the configured list:

```bash
python run_seeds.py --dataset iscx-vpn --cuda 0 --dry-run
python run_seeds.py --dataset iscx-vpn --cuda 0
python run_seeds.py --dataset iscx-vpn --cuda 0 --seeds 32 42 --output checkpoints/custom_runs
```

Replace the dataset argument with `iscx-nonvpn`, `iscx-tor`, `iscx-nontor`, or `ustc-tfc2016` as needed. Each seed runs in an independent process and controls initialization, dropout, the train/validation split, loader randomness, global RNGs, and subprocess `PYTHONHASHSEED`. The preprocessing train/test split is not regenerated. GPU and library operations may still affect determinism.

## Outputs and evaluation

Each run writes `model.pth`, `model.run.json`, and `train.log` under `checkpoints/<dataset>/seed_runs/seed_<seed>/`. Metadata records the seed, effective model/training settings, best validation accuracy, and train/validation indices. `runs.csv` and `summary.json` summarize successful runs using validation mean and sample standard deviation; failed runs receive no performance value. Reusing an output directory overwrites its run files.

Evaluate checkpoints separately on the held-out test set:

```bash
python test.py --dataset iscx-vpn --cuda 0 --seed 32 --checkpoint checkpoints/iscx-vpn/seed_runs/seed_32/model.pth
```

The validation summary is not a test-set estimate. This source update does not introduce trained weights or new performance measurements.

## Environment

Use compatible PyTorch, DGL, CUDA, and Transformers builds. `train.py` defaults `HF_ENDPOINT` to `https://hf-mirror.com`; an explicitly set environment value takes precedence. Cache GPT-2 weights if network access is unavailable. Source checks validate syntax and configuration only; they do not validate end-to-end GPU execution.

## Alignment and checkpoint compatibility

See [Manuscript Alignment](ALIGNMENT.md) for the exact source changes, the original configured values, and unresolved differences in the response letter. Changing the classifier and packet limit requires regenerating processed data and retraining. Previous post-GPT-2 Bi-LSTM checkpoints are incompatible. Historical accuracy, parameter counts, FLOPs, memory, latency, and throughput must not be attributed to this revised model without rerunning the corresponding measurements.
