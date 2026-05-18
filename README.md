# Domain-Generalized DJSCC Benchmark (CIFAR-10)

Pluggable benchmark for Deep Joint Source-Channel Coding under channel-domain
shift. **Algorithms** (the method) and **modes** (single- vs. multi-source
training) are independent knobs, so any algorithm composes with any mode.

## Algorithm vs. Mode

An **algorithm** defines how the per-channel loss is computed (just MSE for
DJSCC-WIT; an algorithm could also add regularizers, channel-aware inputs,
etc.). A **mode** decides the training-time scope over channels:

- `single_source` — one channel per step.
- `multi_source`  — all selected channels per step; per-channel losses are
  averaged. With every channel selected and a plain MSE algorithm, this is
  **Basic Domain-Generalized DJSCC**.

Concretely:

| Setting | Meaning |
|---|---|
| `algorithm.name: djscc_wit` + `mode: single_source` + `train_snr_db: 10.0` | Reproduces Bourtsoulatze et al. 2019 / chunbaobao repo (DJSCC-WIT). |
| `algorithm.name: djscc_wit` + `mode: single_source` (no fixed SNR) | Single-source baseline with uniform SNR sampling. |
| `algorithm.name: djscc_wit` + `mode: multi_source` + `train_channels: all` | Basic Domain-Generalized DJSCC. |
| `algorithm.name: dg_djscc` | Alias of the row above (`mode: multi_source` is the default). |

## Layout

```
dg_djscc/
├── main.py                       # CLI: --mode {train, eval, train_eval}
├── engine.py                     # Harness: setup, eval, loop, ckpt, W&B
├── algorithms/
│   ├── base.py                   # BaseAlgorithm: mode-aware train_step
│   ├── djscc_wit.py              # DJSCC-WIT (MSE forward) + dg_djscc alias
│   └── __init__.py               # Registry + build_algorithm(cfg)
├── models/
│   ├── baseline.py               # Our default wider CNN + GroupNorm/PReLU
│   ├── chunbaobao.py             # Exact arch from chunbaobao/Deep-JSCC-PyTorch
│   └── __init__.py               # Registry + build_model(cfg)
├── channels/                     # AWGN, Rayleigh, Rician (E[|h|^2]=1)
├── data/                         # CIFAR-10 loaders
├── utils/                        # Metrics, ckpt, seeding, W&B wrapper
├── configs/                      # YAML configs (one per experiment)
├── scripts/run_baselines.sh      # Sweep helper
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt
wandb login      # or:  wandb offline

# DG-DJSCC (multi-source djscc_wit on all 3 channels)
python main.py --config configs/dg_djscc_cifar10.yaml --mode train_eval

# DJSCC-WIT reproduction (single-source on AWGN, fixed SNR=10 dB)
python main.py --config configs/djscc_wit_awgn_snr10.yaml --mode train_eval

# DJSCC-LPP
python main.py --config configs/djscc_lpp_awgn_snr10.yaml --mode train_eval

# Single-source baselines (one channel, uniform-SNR training)
python main.py --config configs/baseline_awgn_only.yaml --mode train_eval
python main.py --config configs/baseline_rayleigh_only.yaml --mode train_eval
python main.py --config configs/baseline_rician_only.yaml --mode train_eval

# Run them all in sequence
bash scripts/run_baselines.sh
```

## Baselines
1. DeepJSCC‑L++: "Robust and Bandwidth‑Adaptive Wireless Image Transmission”, IEEE Globecom 2023, by Chenghong Bian, Yulin Shao, and Deniz Gündüz.
2. DeepJSCC-WIT: "Deep Joint Source-Channel Coding for Wireless Image Transmission", in IEEE Transactions on Cognitive Communications and Networking, E. Bourtsoulatze, D. Burth Kurka and D. Gündüz.
3. 

## Datasets

All datasets live under a single shared root so they're downloaded **once**
and reused across runs. The location is resolved in this order:

1. ``data.root`` in the YAML config, if set.
2. The ``DJSCC_DATA_ROOT`` environment variable, if set.
3. Default: ``<repo>/datasets/``.

CIFAR-10 lands in ``<root>/cifar10/``. A small file lock prevents racing
downloads when multiple processes start at once.

Override examples:

```bash
# Point at a shared HPC scratch dir for the whole session
export DJSCC_DATA_ROOT=/scratch/$USER/datasets
python main.py --config configs/dg_djscc_cifar10.yaml --mode train_eval

# Or per-run via the YAML's data.root
```

## Experiments shipped

| Config | Algorithm | Mode | Backbone | Trained on | Tests on |
|---|---|---|---|---|---|
| `dg_djscc_cifar10.yaml` | `dg_djscc` | `multi_source` | `baseline` | AWGN + Rayleigh + Rician | all three (all ID) |
| `djscc_wit_awgn_snr10.yaml` | `djscc_wit` | `single_source` | `chunbaobao` | AWGN @ fixed 10 dB | AWGN (ID), others (OOD) |
| `baseline_awgn_only.yaml` | `djscc_wit` | `single_source` | `baseline` | AWGN (uniform SNR) | AWGN (ID), others (OOD) |
| `baseline_rayleigh_only.yaml` | `djscc_wit` | `single_source` | `baseline` | Rayleigh (uniform SNR) | Rayleigh (ID), others (OOD) |
| `baseline_rician_only.yaml` | `djscc_wit` | `single_source` | `baseline` | Rician (uniform SNR) | Rician (ID), others (OOD) |

## Adding a new algorithm

Subclass `BaseAlgorithm` and implement **only the per-channel forward**:

```python
# algorithms/my_method.py
from .base import BaseAlgorithm

class MyMethod(BaseAlgorithm):
    name = "my_method"

    def compute_per_channel_loss(self, x, channel_name, channel, snr_db):
        z = self.model.encode(x)
        x_hat = self.model.decode(channel(z, snr_db))
        loss = ... # your loss formula, with regularizers etc.
        return loss, x_hat
```

Register it in `algorithms/__init__.py`. Done. The base class handles:

- single-source vs. multi-source training (the loop, the loss averaging),
- optional fixed training SNR via `algorithm.train_snr_db`,
- in-domain / OOD tagging for evaluation,
- per-channel PSNR logging.

If your method should default to multi-source (like a DG variant), set
`default_mode = "multi_source"` on the class.

## Model backbones

Both backbones expose `encode(x)` and `decode(z_tilde)` so any algorithm can
use either. Pick via `model.name` in YAML.

| Name | Notes |
|---|---|
| `baseline` | Wider CNN (64/128 planes) with GroupNorm+PReLU and `PowerNorm` (E[\|z\|²]=1). Our default; stronger for DG-DJSCC. |
| `chunbaobao` | Exact arch from chunbaobao/Deep-JSCC-PyTorch (16/32 planes, total-energy normalization). Paper-faithful for Bourtsoulatze. |

The `baseline` model's `cr` counts real values; the `chunbaobao` model's
`cr` counts complex symbols by default (override with `cr_convention: real`).
Or pin `c_inner` directly for either backbone.

## CLI flags

| Flag | Purpose |
|---|---|
| `--config PATH` | Path to YAML config (required). |
| `--mode {train,eval,train_eval}` | Default `train_eval`. (Distinct from the algorithm's `mode` — this is the run mode.) |
| `--ckpt PATH` | Checkpoint for `eval` mode. Defaults to `<out_dir>/best.pt`. |
| `--wandb-project NAME` / `--wandb-entity NAME` / `--wandb-mode {online,offline,disabled}` | W&B overrides. |
| `--no-wandb` | Disable W&B regardless of config. |
| `--debug-cuda` | Synchronous CUDA launches; use for crash diagnosis. |

## What W&B logs

**Per training step**: `train/loss`, `train/snr_db`, `train/mode`,
`train/psnr_avg`, `train/psnr_<channel>` for each trained channel,
`train/snr_db_fixed` (if pinned).

**Per epoch** (validation at SNR=10 dB): `val/psnr_<channel>@10dB`,
`val/psnr_<channel>_id@10dB` / `_ood@10dB`, `val/in_domain_psnr@10dB`,
`val/ood_psnr@10dB`, `val/generalization_gap@10dB`.

**Full eval sweep**: per-(SNR, channel) scalars under `eval/psnr_*`,
`eval/psnr_<channel>_id` / `_ood`, `eval/psnr_in_domain`, `eval/psnr_ood`,
`eval/generalization_gap`. W&B Tables `eval/psnr_table` and
`eval/msssim_table` with `(ID)`/`(OOD)` tagged columns.

**Checkpoints**: best model uploaded as a W&B Artifact named
`<experiment.name>-best`, with `in_domain` and `algorithm` recorded so
later evaluation can re-tag columns correctly.
