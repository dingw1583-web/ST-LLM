# Alignment with the reviewer response

The current source provides the main six-block model, HDF, No Pretrain and Full Tuning entry points, a manifest-driven CESNET 75-class temporal route, paired preprocessing, ten-seed test evaluation, paired t-tests with Holm correction, and CUDA deployment profiling. See [Experiment Guide](EXPERIMENTS.md) for exact commands and limitations.

## Architecture and settings

The extra Bi-LSTM after GPT-2 has been removed; the header Bi-LSTM remains. The classifier takes temporal mean-pooled 768-dimensional GPT-2 features, then `Linear(768,128)`, ReLU, dropout and `Linear(128,classes)`. GraphSAGE is one layer with mean aggregation and hidden dimension 128. Fusion produces 256 dimensions before a separate 768-dimensional projection. HDF U=1 preserves final-block attention and LayerNorm tuning while FFNs remain frozen.

The response letter's page-15 MLP wording must be changed to MHA. Its fusion dimension statements and Table 1 must distinguish fusion output 256 from GPT-2 input 768. Mean pooling and linear-warm-up/cosine scheduling are explicit implementation choices that should be described in the manuscript.

All routes inherit response-table defaults: 30 packets, 150 payload bytes, 40 header bytes, batch 512, 120 epochs, Adam, learning rate 1e-2 to 1e-4, warm-up 0.1, dropout 0.2, and the ten configured seeds. These are reproducible defaults, not proof of historical experimental settings. `PREVIOUS_SETTINGS.json` preserves the earlier configuration snapshot, not an experimental record.

## Changes requiring updated manuscript wording

Files are sorted and modalities now share packet selection and sample IDs. Training/validation splitting is source-group-aware to prevent segments of one source crossing the boundary, targeting a 50/50 sample ratio. With multiple segments per source this is approximately equal rather than exactly equal. Tor now includes the last 60-second window. Positive-PMI graphs retain isolated byte nodes including padding. These corrections require regenerated data and fresh experiments; do not describe them as preserving historical results.

CESNET uses explicit Week-1 training, Week-2 validation and Week-3/4 tests. The original 126,029-flow subset list is unavailable, so the new deterministic subset builder does not claim to reconstruct it. Temporal class selection and source week metadata must be documented with the actual manifest.

## Evidence limits

No trained checkpoints or fabricated results are included. New source entry points do not validate the reported accuracy, 548-second training time, 603.9-MiB memory, 17.2-ms latency or 2560.5-samples/s throughput. Measure the revised model and all comparators under the documented protocol before updating result tables. Baseline statistics require genuine paired per-seed results; the utility never invents these inputs.

Full Layer, FPT, feature-removal ablations and external baseline training are not implemented by the three provided variant switches. Do not claim every experimental figure can be reproduced solely from this release. Local checks cover source/configuration and synthetic correctness; GPU training/profiling remain to be run with the required environment and data.
