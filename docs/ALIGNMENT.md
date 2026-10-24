# Alignment with the reviewer response

## Model architecture

- Removed the extra two-layer bidirectional LSTM after GPT-2.
- GPT-2 now returns packet representations of shape `[batch, packets, 768]`. Temporal mean pooling yields one 768-dimensional flow representation, followed by `Linear(768, 128)`, ReLU, dropout, and `Linear(128, classes)`.
- Retained the header Bi-LSTM, one-layer mean-aggregator GraphSAGE with hidden dimension 128, and six retained GPT-2 blocks.
- Cross-gated fusion outputs 256 dimensions; the separate projection produces the 768-dimensional GPT-2 input. These are distinct dimensions.
- The `PFA` implementation is unchanged. Under U=1, the final retained block's attention and the LayerNorms are trainable, while FFNs are frozen. The response letter's page-15 statement about tuning the MLP is a text error; it must not be implemented in code.
- Temporal mean pooling specifies how a packet sequence enters the two-layer classifier. The response does not independently specify the pooling operation; this is an explicit implementation choice for this aligned release.

## Training settings

The response supplies a shared settings table, not separate original training logs. The following per-dataset configuration matrix implements that table as reproducible defaults for new runs:

| Dataset | Packets | Batch size | Epochs | Optimizer | Peak LR | Min LR | Warm-up | Dropout | Default seed |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ISCX-VPN | 30 | 512 | 120 | Adam | 1e-2 | 1e-4 | 0.1 | 0.2 | 32 |
| ISCX-NonVPN | 30 | 512 | 120 | Adam | 1e-2 | 1e-4 | 0.1 | 0.2 | 32 |
| ISCX-Tor | 30 | 512 | 120 | Adam | 1e-2 | 1e-4 | 0.1 | 0.2 | 32 |
| ISCX-NonTor | 30 | 512 | 120 | Adam | 1e-2 | 1e-4 | 0.1 | 0.2 | 32 |
| USTC-TFC2016 | 30 | 512 | 120 | Adam | 1e-2 | 1e-4 | 0.1 | 0.2 | 32 |

Shared values are intentionally identified as reported defaults. They are not evidence that the original experiments used identical settings. Dataset-specific values can be supplied in their configuration classes when validated logs become available. The previous configuration matrix and actual trainer overrides are preserved in `PREVIOUS_SETTINGS.json`; that snapshot also is not an experimental record.

Adam, loss smoothing, weight decay, learning-rate bounds, warm-up ratio, batch size, epochs, and model dropout now read the effective configuration. Linear warm-up followed by cosine decay is the documented scheduler choice because the response gives only a range and warm-up ratio. Weight decay and label smoothing default to zero; neither is specified in the response table. A run prints and saves its effective settings in `model.run.json` (or the selected checkpoint stem's `.run.json`).

## Splits and seeds

The per-class preprocessing test split and the seeded 50/50 train/validation split are unchanged. All configurations use the ten seeds 32, 42, 52, 62, 72, 82, 92, 102, 112, and 122. Preprocessing continues to follow filesystem order. The loader rejects incompatible packet counts and mismatched label sequences; matching counts and labels alone do not prove original flow identity.

## Migration and evidence limits

Regenerate both header and payload artifacts with the 30-packet configuration, then retrain the new classifier. Checkpoints containing the old post-GPT-2 `rnn.*` parameters do not match the revised head. Do not silently discard those parameters or load them with `strict=False` to claim compatibility.

No new training results are supplied. The architecture change can affect accuracy, parameters, FLOPs, memory, latency, and throughput; all such reported values require measurement with this model. Local checks cover Python parsing, configuration/routing, schedule boundaries and edge cases, and unchanged HDF source structure. Full model execution requires compatible PyTorch, DGL, Transformers, GPT-2 weights, and datasets.
