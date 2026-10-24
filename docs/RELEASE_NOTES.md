# Release Notes

## Reviewer-response alignment

Removed the post-GPT-2 Bi-LSTM, added temporal mean pooling and a direct two-layer classifier, and wired training to the response-table configuration. The HDF implementation is unchanged. Previous configurations are retained as a source snapshot, and current settings are recorded per run. Regenerate data and retrain; this release does not validate historical performance. See [Alignment Details](ALIGNMENT.md).

## English documentation update

- Translated Chinese runtime messages, exceptions, and command-line help into English in `train.py`, `model.py`, and `dataloader.py`.
- Reorganized the README around the model, environment, data preparation, training, and evaluation.
- Documented ISCX-VPN, ISCX-NonVPN, ISCX-Tor, ISCX-NonTor, and USTC-TFC2016 with equal coverage and commands for every configuration.
- Moved detailed class mappings and repeated-run procedures to `DATASETS.md` and `REPRODUCIBILITY.md`.
- At that documentation-only stage, model computations and training settings were unchanged. The later alignment changes are described above.

## Included functionality

The repository contains packet extraction, PMI graph construction, training/evaluation, and an optional repeated-run script. Local roots are configurable through `ST_LLM_DATA_ROOT` and `ST_LLM_CHECKPOINT_ROOT`. Empty categorization templates are supplied for all five configurations. Original captures, filename manifests, generated datasets, and trained weights are not distributed.

## Validation

Python syntax, class mappings, and seed wiring are checked by `python scripts/check_source.py`. The translation changes only displayed text and preserves the executable structure. No model training or inference measurements were performed as part of this documentation update.
