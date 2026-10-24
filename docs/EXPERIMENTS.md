# Reproducing experiments

These entry points generate new measurements. No published scores, checkpoints, original CESNET file list, or original profiling logs are supplied. Regenerate data and retrain after this update; do not attribute previous results to this version without measurement.

## Paired preprocessing

For the five existing dataset routes:

```bash
python preprocess.py --dataset iscx-vpn
```

Files are sorted by name within each class. Header and payload use the same nonempty-packet indices. The script saves headers, payload graphs, ordered sample IDs, source SHA-256 group IDs, packet indices, and `split_manifest.json`. Padding graphs retain a node for the padding byte even when no positive-PMI edges exist. The loader rejects missing/mismatched identity lists; old artifacts must be regenerated.

The existing per-class first `floor(N/10)+1` test rule is retained. Identical source bytes crossing train/test cause an error. Tor segmentation now uses consecutive 60-second windows including the final window. A boundary that divides segments of one original flow across train/test is rejected; curate the inputs/boundary before running, rather than silently accepting leakage.

Training/validation partitioning uses a seeded source-group split targeting half of the samples. All segments and byte-identical copies of a source stay together. With one sample per source this gives equal sizes up to one sample; otherwise sizes may differ. This replaces the previous unconstrained sample-level random split. Use the saved IDs for all paired baselines and update the manuscript's split description accordingly.

## CESNET-QUIC22 subset and temporal split

Prepare an explicit candidate CSV with `path,label,week,flow_id`. Each path points to one already separated bidirectional flow, in PCAP/PCAPNG or NPZ form. `flow_id` is a globally unique original-flow/session identifier and cannot be repeated across weeks. Week values must come from the dataset's acquisition metadata; the script does not infer them from file names. All four weeks must refer to the same fixed class vocabulary.

For NPZ inputs, arrays `header` and `payload` contain corresponding packet-byte sequences. Legacy NPZ headers use the existing IPv4-style identifier removal. Set scalar `header_masked=True` only if identifiers have already been removed. PCAP inputs support IPv4/IPv6 and UDP/TCP; the reader removes IP address bytes and transport ports before truncation, excludes link-layer headers and IPv6 extension headers, and takes the transport payload. The old TCP-specific `pcap2npy.py` is not the QUIC extraction route. Input captures must already be split into flows; this script does not perform flow reconstruction. Flow-summary tables without packet bytes cannot be used as byte-graph inputs.

```csv
path,label,week,flow_id
flows/example_w1.pcap,application_a,1,original_flow_001
flows/example_w2.npz,application_a,2,original_flow_002
```

The example only illustrates the schema; it is not a usable 75-class dataset. Build the subset using exactly 75 Week-1 classes. `classes.txt` lists one actual class label per line. An optional cap selects deterministically by the hash of seed and flow ID, separately for each class/week; omit the cap to retain all candidates. The builder requires each selected class in every week and records its recipe and input checksum.

```bash
python build_cesnet_subset.py --input candidates.csv --classes classes.txt --seed 32 --output cesnet_subset.csv
python preprocess.py --dataset cesnet-quic22 --manifest cesnet_subset.csv
python run_seeds.py --dataset cesnet-quic22 --cuda 0 --variant hdf
```

Week 1 trains the model, Week 2 selects the checkpoint, and Weeks 3 and 4 are separately held-out future tests. No random split replaces this temporal partition. `labels.json` fixes the numeric class mapping. Paths are resolved relative to the manifest. Manifest content and source checksums are saved. The script verifies source identity/content separation; the accuracy of the supplied week metadata remains the dataset curator's responsibility.

This recipe does not reconstruct the reported 126,029-flow subset: reproducing that exact subset requires the original file list and selection settings. Do not select classes or caps using test performance.

## LLM controls

```bash
python run_seeds.py --dataset iscx-vpn --cuda 0 --variant hdf
python run_seeds.py --dataset iscx-vpn --cuda 0 --variant no-pretrain
python run_seeds.py --dataset iscx-vpn --cuda 0 --variant full-tuning
```

- `hdf`: pretrained six-block GPT-2; final-block attention and LayerNorms train; FFNs and embeddings are frozen. This default HDF rule is preserved.
- `no-pretrain`: a fresh GPT-2 configuration with six blocks, identical parameter shapes and the same traffic encoders/classifier, trained from scratch. No pretrained weights are downloaded for this variant.
- `full-tuning`: pretrained six-block GPT-2 with all backbone parameters unfrozen, including positional embeddings. Token embeddings remain unused because the model accepts `inputs_embeds`.

No Pretrain and Full Tuning use full backbone training; HDF uses selective training. Total model capacity matches, but trainable-parameter counts differ by design. All use the same optimizer/schedule, dataset artifacts, and run seed. Checkpoint metadata records the variant. A genuinely controlled comparison also requires any external baseline to consume the same saved train/validation/test identities. Full Layer, FPT, and feature-removal variants are not provided by these three switches.

## Test metrics and statistics

The runner evaluates each best-validation checkpoint on the independent test split. CESNET evaluates Week 2, Week 3 and Week 4 separately. Each seed saves labels, predictions, IDs, split and variant metadata, accuracy, macro precision/recall/F1, and weighted precision/recall/F1. `test_summary.json` contains per-split means and sample standard deviations. `summary.json` remains explicitly validation-only. Failed evaluations are marked failed and excluded; report the actual successful run count and do not describe an incomplete set as ten runs.

```bash
python test.py --dataset iscx-vpn --cuda 0 --seed 32 --checkpoint checkpoints/iscx-vpn/hdf/seed_runs/seed_32/model.pth
python compare_runs.py --ours checkpoints/iscx-vpn/hdf/seed_runs --baseline checkpoints/iscx-vpn/no-pretrain/seed_runs checkpoints/iscx-vpn/full-tuning/seed_runs --metrics accuracy macro_f1 --output results/paired.json
```

`compare_runs.py` enforces identical seed sets, labels, sample IDs, source groups, train IDs and validation IDs. It computes a two-sided paired t-test, the t-based 95% CI for the mean difference, and Holm-adjusted p-values. Scores and differences are fractions; multiply differences by 100 to report percentage points. A zero-variance difference is handled explicitly. With ten successful pairs the CI uses 9 degrees of freedom; other counts are recorded rather than disguised as ten.

All comparisons in one invocation form one Holm family. To correct across datasets as well, provide `--comparisons family.json` containing a list of `{"ours":"...","baseline":"...","split":"test"}` objects, and the shared `--metrics`. External baselines must export the same per-seed JSON schema as `test.py`; no third-party baseline training code or baseline measurements are bundled. Select the strongest baseline by the stated protocol rather than corrected p-values. Both macro and weighted F1 are reported because the original F1 averaging convention must be stated in the manuscript.

## Deployment measurements

```bash
python benchmark.py --dataset iscx-vpn --checkpoint checkpoints/iscx-vpn/hdf/seed_runs/seed_32/model.pth --cuda 0 --batch-size 1 --warmup 20 --repeats 100 --output results/latency.json
python benchmark.py --dataset iscx-vpn --checkpoint checkpoints/iscx-vpn/hdf/seed_runs/seed_32/model.pth --cuda 0 --batch-size 512 --warmup 20 --repeats 100 --output results/throughput.json
```

GPU synchronization brackets each forward pass after warm-up. Latency is per batch, with mean/p50/p95; batch size 1 gives single-sample latency. Throughput divides the actual processed sample count by total timed forward seconds, including partial batches. One sample is one preprocessed flow representation. Input transfer, preprocessing and disk loading are excluded from timing; this is not end-to-end network-service latency. The script cycles deterministically through the test loader, so small test sets may repeat samples.

Memory reports the CUDA allocator's peak allocated and peak reserved memory in MiB after resetting peak counters, plus the warm-up allocation and total device capacity. Peak allocation includes model weights and current device inputs; it is not total GPU capacity or system-wide `nvidia-smi` usage. Reserved memory includes allocator caching. GPU model, software versions, total/trainable parameter counts, warm-up and batch settings are recorded. Training metadata separately records wall time of the train/validation loop including logging, excluding preprocessing and model loading.

Use identical hardware, precision, input definition and timing scope for all compared models. Renaming packets/s to samples/s does not make packet and flow inputs comparable. This script measures LLM4ETC variants; other architectures require their own adapters under the same protocol. No historical latency, memory, throughput or training-time values are inserted by the scripts.

## Verification limits

Local source checks and synthetic tests cover routing, paired packet filtering, source grouping, temporal validation, statistics and padded-graph handling. End-to-end training and CUDA profiling require PyTorch, DGL, Transformers, weights, packet data and a CUDA device. Those measurements have not been run in the editing environment.
