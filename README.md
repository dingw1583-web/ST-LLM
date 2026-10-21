# ST-LLM

Spatial-temporal encrypted traffic classification with packet-header encoding, byte-level payload graphs, cross-gated fusion, and a partially frozen GPT-2 backbone.

![ST-LLM architecture](figures/ST-LLM.png)

## Model

The implementation combines:

1. A byte embedding and bidirectional LSTM encoder for packet headers.
2. A GraphSAGE encoder for payload byte graphs constructed from pointwise mutual information.
3. A cross-gated filter that combines the two packet representations.
4. A projection into a GPT-2 backbone retaining its first six Transformer blocks.
5. A downstream bidirectional LSTM and classification head.

The default adaptation rule tunes LayerNorm parameters in the earlier retained blocks and non-MLP parameters in the final retained block. Token and position embedding weights are frozen. See `model.py` for the exact implementation.

## Datasets

| CLI dataset name | Class count | Data directory | Categorization files |
| --- | ---: | --- | --- |
| `iscx-vpn` | 6 | `data/ISCX-VPN/` | `DATA/ISCX-VPN/` |
| `iscx-nonvpn` | 6 | `data/ISCX-NonVPN/` | `DATA/ISCX-NonVPN/` |
| `iscx-tor` | 8 | `data/ISCX-Tor/` | `DATA/ISCX-Tor/` |
| `iscx-nontor` | 8 | `data/ISCX-NonTor/` | `DATA/ISCX-NonTor/` |
| `ustc-tfc2016` | 20 | `data/USTC-TFC2016/` | `DATA/USTC-TFC2016/` |

`DATA/` contains empty categorization templates for all supported datasets, whereas lowercase `data/` is the default location for local traffic and generated artifacts. Original capture-filename manifests are omitted from this public repository because they can contain IP addresses and traffic metadata. Templates do not contain packet captures. The preprocessing scripts read the class directories configured in `config.py`; they do not automatically move captures based on the lists.

### USTC-TFC2016 label mapping

The configuration uses one 20-class task, with labels 0–9 for benign applications and 10–19 for malware families. Directory names are case-sensitive.

| Label | Class directory | Group |
| ---: | --- | --- |
| 0 | `Benign_BitTorrent` | Benign |
| 1 | `Benign_Facetime` | Benign |
| 2 | `Benign_FTP` | Benign |
| 3 | `Benign_Gmail` | Benign |
| 4 | `Benign_MySQL` | Benign |
| 5 | `Benign_Outlook` | Benign |
| 6 | `Benign_Skype` | Benign |
| 7 | `Benign_SMB` | Benign |
| 8 | `Benign_Weibo` | Benign |
| 9 | `Benign_WorldOfWarcraft` | Benign |
| 10 | `Malware_Cridex` | Malware |
| 11 | `Malware_Geodo` | Malware |
| 12 | `Malware_Htbot` | Malware |
| 13 | `Malware_Miuref` | Malware |
| 14 | `Malware_Neris` | Malware |
| 15 | `Malware_Nsis-ay` | Malware |
| 16 | `Malware_Shifu` | Malware |
| 17 | `Malware_Tinba` | Malware |
| 18 | `Malware_Virut` | Malware |
| 19 | `Malware_Zeus` | Malware |

The 20 USTC categorization files shipped here are empty placeholders. Populate them with your actual capture filenames if you use them as manifests. Obtain captures separately and split them into bidirectional TCP flows before placing them in their class directories. A single dataset capture can contain many flows; the converter does not perform flow splitting.

The new USTC configuration and CLI routing are included, but USTC training, evaluation, and hyperparameter validation have not been performed for this release. Its model and training settings are explicitly listed in `USTCTFC2016Config`; these initial values should be validated on your data before an experiment.

## Repository contents

| Path | Purpose |
| --- | --- |
| `config.py` | Model, training, dataset paths, and class mappings |
| `model.py` | ST-LLM model |
| `dataloader.py` | Header-array and payload-graph dataset loader |
| `pcap2npy.py` | Extract both headers and payloads from per-flow TCP PCAPs |
| `preprocess.py` | Create flow arrays and byte graphs |
| `train.py` | Training with a validation split |
| `test.py` | Evaluation of a saved checkpoint |
| `utils.py`, `optim.py` | Processing, collation, graph construction, scheduler utilities |
| `pcap2header.py` | Separate header-only utility; not needed for the main pipeline |
| `pcapng2pcap.py` | Optional format-conversion utility with a configurable source path |
| `DATA/` | Categorization lists and USTC placeholders |
| `docs/RELEASE_NOTES.md` | Cleanup scope and validation limitations |

## Environment

Use a Python environment with mutually compatible PyTorch, DGL, CUDA, and Transformers versions. Install PyTorch and DGL builds matching your machine first, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch dgl
python -m pip install -r requirements.txt
python -c "import torch, dgl, transformers; print(torch.__version__, dgl.__version__, transformers.__version__); print('CUDA:', torch.cuda.is_available())"
```

The PyTorch/DGL command above installs the packages selected by your package index; it does not guarantee CUDA support. Check CUDA availability and confirm that DGL can execute on the selected device before GPU training. On Windows, activate the environment with `.venv\Scripts\activate`.

The model loads pretrained `gpt2` through Transformers. The first model initialization requires access to those weights unless they are already cached. `train.py` supplies `https://hf-mirror.com` as the default Hugging Face endpoint; set `HF_ENDPOINT` yourself to override it.

This release was checked for Python syntax and configuration consistency. A complete dependency environment and GPU training were not available during release preparation.

## Data preparation

### 1. Set the local data location

By default, data and checkpoints are relative to the repository root. You can override their roots:

```bash
export ST_LLM_DATA_ROOT=/absolute/path/to/data
export ST_LLM_CHECKPOINT_ROOT=/absolute/path/to/checkpoints
```

For USTC, place per-flow captures in directories such as `data/USTC-TFC2016/Benign_BitTorrent/` and `data/USTC-TFC2016/Malware_Cridex/`. Each class directory must exist. Class-to-label assignments come from `config.py`.

### 2. Extract packet bytes

```bash
python pcap2npy.py --dataset ustc-tfc2016
```

This writes a per-flow NPZ next to each `.pcap`, containing both header and payload sequences. It skips an existing NPZ with the same filename stem. Use TCP captures compatible with the converter's packet fields.

### 3. Build arrays and graphs

```bash
python preprocess.py --dataset ustc-tfc2016
```

The outputs include `train.npz`, `test.npz`, `header_train.npz`, `header_test.npz`, and their graph files under the selected dataset directory. The main model consumes header arrays and payload graphs; header graphs are generated by the existing pipeline but are not used by the main loader.

Defaults are 50 packets per sample, 150 payload bytes per packet, 40 header bytes per packet, padding value 256, and PMI window size 5. Tor uses time-window segmentation; the other dataset routes, including USTC, use the existing per-flow segmentation path.

For any other dataset, replace `ustc-tfc2016` in the commands with the corresponding CLI name in the dataset table.

## Training

```bash
python train.py --dataset ustc-tfc2016 --cuda 0
```

To choose the checkpoint location:

```bash
python train.py --dataset ustc-tfc2016 --cuda 0 --save_path checkpoints/ustc_custom.pth
```

The supplied trainer divides the prepared training set into equal training and validation subsets using the selected run seed (`--seed`, or the dataset's configured `SEED`). It uses AdamW with learning rate `3e-4`, weight decay `1e-4`, label smoothing `0.05`, and OneCycleLR. These values are currently set in `train.py`; similarly named fields in `config.py` do not override them. Configured gradient accumulation is not implemented in this trainer.

The default USTC checkpoint is `checkpoints/mix_model_ustc_tfc2016.pth`.

## Ten-seed experiments

`Config.SEEDS` contains `(32, 42, 52, 62, 72, 82, 92, 102, 112, 122)`. USTC uses 32 as its single-run default and inherits this ten-seed list. Existing dataset-specific single-run defaults are retained, including VPN's 66. Run a single chosen seed with:

```bash
python train.py --dataset ustc-tfc2016 --cuda 0 --seed 32
```

Run all ten seeds in independent processes with:

```bash
python run_seeds.py --dataset ustc-tfc2016 --cuda 0
```

Preview the ten commands without starting training:

```bash
python run_seeds.py --dataset ustc-tfc2016 --cuda 0 --dry-run
```

Each run writes `model.pth`, `model.run.json`, and `train.log` under `checkpoints/ustc-tfc2016/seed_runs/seed_<seed>/`. The run metadata records the seed, best validation accuracy, and train/validation indices. `runs.csv` and `summary.json` report successful runs and their validation mean/sample standard deviation. Failed runs are not assigned performance values. Existing files at the same run paths are overwritten on rerun; use `--output /path/to/new_runs` to keep a separate experiment.

The seed controls model initialization, dropout, training/validation split, DataLoader generators, global RNGs, and subprocess `PYTHONHASHSEED`. Consequently, this protocol varies the training/validation split as well as optimization randomness. The preprocessing train/test split is not regenerated by the runner. Determinism can still depend on GPU/library operations. Evaluate each checkpoint separately on the same held-out test set; the summary of best validation accuracy is not a test-set performance estimate.

```bash
python test.py --dataset ustc-tfc2016 --cuda 0 --seed 32 --checkpoint checkpoints/ustc-tfc2016/seed_runs/seed_32/model.pth
```

## Evaluation

```bash
python test.py --dataset ustc-tfc2016 --cuda 0
```

Evaluation reads the configured header test array, payload test graph, and `MIX_MODEL_CHECKPOINT`. If you trained with a custom `--save_path`, pass that path with `test.py --checkpoint /path/to/model.pth`. The script prints per-class precision, recall, and F1.

## Reproducibility notes

Keep header and payload sample identities, packet selection, labels, and ordering aligned. The existing preprocessing branches filter packet contents separately; check their outputs before experiments. The supplied loader does not fully establish cross-modality flow identity. Dataset splitting currently follows filesystem iteration order, so retain a fixed data manifest for reproducible splits. Split at the original-session level when multiple samples originate from one session.

No trained checkpoints, PCAPs, full processed datasets, or new experimental performance results are included. Upload preparation does not establish training correctness or deployment efficiency.

## Repository and GitHub upload

Extract this repository first. Upload its contents, including `README.md`, the Python files, `DATA/`, and `figures/`; uploading only the ZIP will not expose the source as a normal repository.

For Git, create your empty target repository, then run the following from this project's root after replacing the remote URL:

```bash
git init
git add .
git commit -m "Prepare ST-LLM release with USTC-TFC2016 configuration"
git branch -M main
git remote add origin https://github.com/dingw1583-web/ST-LLM.git
git push -u origin main
```

The included `.gitignore` excludes local captures, generated data, checkpoints, caches, and environments. The source, categorization files, figure, and license remain trackable. A GitHub Actions workflow checks source syntax and class mappings; it does not train the model.

## License

This project is distributed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
