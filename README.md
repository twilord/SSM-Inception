# SSM-Inception Minimal Reproducibility Package

This directory provides a lightweight reproducibility package for
SSM-Inception. The released implementation focuses on the DSADS experiment and
includes the proposed model, the reported DSADS subject split, data
preprocessing, model/training configuration, a three-fold training and
evaluation entry point, and the hardware measurement protocol.

This package is intentionally scoped. It does **not** contain public dataset
files, trained weights, results from other datasets, baseline implementations,
internal experiment logs, or manuscript source files.

## Contents

```text
OpenSource/
├── configs/dsads.yaml             # experiment and model configuration
├── ssm_inception/model.py         # S4D + DS-Inception + SE model
├── ssm_inception/data.py          # DSADS loader and leakage-safe test split
├── train.py                       # 3-fold training, checkpointing, evaluation
├── benchmark.py                   # Params/FLOPs/latency/RSS/throughput
└── environment.yml                # reference software environment
```

## Dataset layout

Download and extract DSADS separately. The loader expects the original layout:

```text
DSADS/
├── a01/p1/s01.txt
├── a01/p1/s02.txt
├── ...
└── a19/p8/s60.txt
```

If `feature1/files.txt` and `feature1/labels.txt` are present, they are used as
an explicit ordered manifest and cross-checked against directory-derived
activity labels. Otherwise, files are discovered in deterministic sorted
order. The data itself must not be committed to this repository.

## Environment

```bash
conda env create -f environment.yml
conda activate ssm-inception-dsads
```

## Training and evaluation

```bash
python train.py \
  --config configs/dsads.yaml \
  --data-root /path/to/DSADS \
  --device cuda:0
```

The command writes one validation-selected checkpoint per fold, fold-level and
aggregate Accuracy/Macro-F1, normalization statistics, the exact split
manifest, and test predictions under `outputs/dsads/`.

The expected data audit is 9,120 total windows, split into 7,980 development
windows (subjects 1–6 and 8) and 1,140 independent test windows (subject 7).

## Diagnostic hardware benchmark

Run from this directory on the target device for a controlled DSADS diagnostic
measurement:

```bash
python benchmark.py \
  --checkpoint outputs/dsads/fold_0/checkpoint.pt \
  --device cpu \
  --threads 4 \
  --warmup 100 \
  --iterations 5000
```

The helper above uses an explicitly controlled CPU thread count and the released
DSADS input shape. It is therefore a diagnostic benchmark, not a direct
reproduction of the WISDM measurements reported in Table VI. The reported
Table VI measurements retained the default PyTorch CPU thread configuration.
See [PROTOCOL.md](PROTOCOL.md) and [HARDWARE_PROTOCOL.md](HARDWARE_PROTOCOL.md)
before reporting or comparing results.

## Scope and licensing

This package exposes the material needed to inspect the released
SSM-Inception implementation and reproduce the DSADS model setting described
above. Add the project license selected by the authors and the final paper
citation before public release. DSADS is distributed by its original provider
under its own terms and is not included.
