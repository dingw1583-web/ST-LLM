# Reproducibility Notes

See [Experiment Guide](EXPERIMENTS.md) for paired preprocessing, the CESNET subset manifest, week-specific evaluation, controlled LLM variants, test-set aggregation, paired statistics and deployment measurements.

The default representation uses 30 packets per sample, 150 payload bytes and 40 header bytes per packet, padding value 256, and PMI window size 5. The model uses a header Bi-LSTM, one-layer mean-aggregator GraphSAGE with hidden dimension 128, cross-gated fusion producing 256 dimensions, a separate 768-dimensional projection, six GPT-2 blocks, temporal mean pooling and a two-layer classifier. The default HDF rule trains final-block attention and LayerNorms while freezing FFNs.

Configuration defaults are Adam, peak learning rate `1e-2`, minimum `1e-4`, linear warm-up then cosine decay, warm-up ratio 0.1, 120 epochs, batch size 512, and dropout 0.2. Weight decay and label smoothing are zero. These implement the response table as defaults for new runs; they are not recovered historical logs.

The ten seeds are `32, 42, 52, 62, 72, 82, 92, 102, 112, 122`. The runner sets subprocess `PYTHONHASHSEED`; the trainer seeds Python, NumPy, PyTorch, CUDA and DGL. The source-group split uses NumPy's seeded generator. GPU/library operations can still affect determinism. Each run records settings and the actual selected sample IDs, group IDs and train/validation indices. A seed alone does not reproduce different cached datasets.

Checkpoint directories now include the variant: `checkpoints/<dataset>/<variant>/seed_runs/seed_<seed>/`. `model.run.json` identifies the model version and variant; `test.py` requires matching metadata and loads state dictionaries strictly. Test results and validation summaries are stored separately. Rerunning an output directory replaces its run files; use a separate output directory for distinct experiments.

Regenerate all old artifacts and retrain. Paired packet selection, complete Tor windows, isolated graph nodes and grouped validation partitioning differ from the previous implementation. Previous published values must be remeasured before they can be claimed for this version. The original post-GPT-2 recurrent-head checkpoints remain incompatible.

Use compatible PyTorch, DGL, CUDA and Transformers builds. `train.py` defaults `HF_ENDPOINT` to `https://hf-mirror.com` unless explicitly set. Pretrained variants require cached/downloadable GPT-2 weights. Source tests do not validate full GPU execution.
