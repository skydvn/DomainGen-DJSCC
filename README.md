# Domain-Generalized DJSCC Benchmark (CIFAR-10)

Pluggable benchmark for Deep Joint Source-Channel Coding under channel-domain
shift. Algorithms live in `algorithms/`; the engine is method-agnostic.

## Layout

```
dg_djscc/
├── main.py                       # CLI: --mode {train, eval, train_eval}
├── engine.py                     # Harness: setup, eval, loop, ckpt, W&B
├── algorithms/                   # One file per training method
│   ├── base.py                   #   BaseAlgorithm interface
│   ├── dg_djscc.py               #   Basic Domain-Generalized DJSCC
│   ├── single_source.py          #   Train-on-one-channel baseline
│   └── __init__.py               #   Registry + build_algorithm(cfg)
├── channels/                     # AWGN, Rayleigh, Rician (E[|h|^2]=1)
├── models/                       # DJSCC encoder/decoder + PowerNorm
├── data/                         # CIFAR-10 loaders
├── utils/                        # Metrics, ckpt, seeding, W&B wrapper
├── configs/                      # YAML configs (1 per experiment)
├── scripts/run_baselines.sh      # Sweep helper
└── README.md
```

## How algorithms plug in

Each YAML config picks an algorithm by name and passes its knobs under
`algorithm.*`:

```yaml
algorithm:
  name: dg_djscc          # registered in algorithms/__init__.py
  train_channels: all     # or 'awgn' / ['awgn', 'rician']
```

```yaml
algorithm:
  name: single_source
  train_channel: awgn     # exactly one
```

`engine.py` calls `build_algorithm(cfg, model, channels, optimizer)` and then
hands every mini-batch to `algorithm.train_step(x, snr_db)`. Engine never sees
algorithm-specific logic.

## Adding a new algorithm

1. Create `algorithms/my_method.py`:

   ```python
   from .base import BaseAlgorithm

   class MyMethod(BaseAlgorithm):
       name = "my_method"

       def in_domain_channels(self):
           return list(self.channels.keys())

       def train_step(self, x, snr_db):
           # ... compute loss, .backward(), .step()
           return loss.detach(), {"train/loss": loss.item(), ...}
   ```

2. Register it in `algorithms/__init__.py`:

   ```python
   from .my_method import MyMethod
   ALGORITHMS = {"dg_djscc": DGDJSCC, "single_source": SingleSource,
                 "my_method": MyMethod}
   ```

3. Add a YAML with `algorithm.name: my_method` and any extra knobs under
   `algorithm.*`. Engine, main, channels, models, data — all unchanged.

The base class also exposes `on_train_start()` and `on_epoch_end(epoch)` hooks
for algorithms that need EMA, schedules, curricula, or meta-learning state.

## Experiments shipped

| Config | Algorithm | Trained on | Tests on |
|---|---|---|---|
| `dg_djscc_cifar10.yaml` | `dg_djscc` | AWGN + Rayleigh + Rician | all three (all ID) |
| `baseline_awgn_only.yaml` | `single_source` | AWGN | AWGN (ID), others (OOD) |
| `baseline_rayleigh_only.yaml` | `single_source` | Rayleigh | Rayleigh (ID), others (OOD) |
| `baseline_rician_only.yaml` | `single_source` | Rician | Rician (ID), others (OOD) |

## Quickstart

```bash
pip install -r requirements.txt
wandb login   # or `wandb offline`

# DG-DJSCC (main method): train + evaluate
python main.py --config configs/dg_djscc_cifar10.yaml --mode train_eval

# Single-source baseline
python main.py --config configs/baseline_awgn_only.yaml --mode train_eval

# Full sweep (all baselines + DG-DJSCC)
bash scripts/run_baselines.sh
```

## CLI flags

| Flag | Purpose |
|---|---|
| `--config PATH` | Path to YAML config (required). |
| `--mode {train,eval,train_eval}` | Default `train_eval`. |
| `--ckpt PATH` | Checkpoint for eval mode. Defaults to `<out_dir>/best.pt`. |
| `--wandb-project NAME` / `--wandb-entity NAME` / `--wandb-mode {online,offline,disabled}` | W&B overrides. |
| `--no-wandb` | Disable W&B regardless of config. |
| `--debug-cuda` | Synchronous CUDA launches; use for crash diagnosis. |

## What W&B logs

**During training** (per `train.log_every`): `train/loss`, `train/snr_db`,
`train/psnr_avg`, `train/psnr_<channel>`.

**Per epoch** (validation at SNR=10 dB): `val/psnr_<channel>@10dB`,
`val/psnr_<channel>_id@10dB` or `_ood@10dB`, `val/in_domain_psnr@10dB`,
`val/ood_psnr@10dB`, `val/generalization_gap@10dB`.

**During eval** (full SNR sweep): per-(SNR, channel) scalars under
`eval/psnr_*`, `eval/psnr_<channel>_id` / `_ood`, `eval/psnr_in_domain`,
`eval/psnr_ood`, `eval/generalization_gap`. Plus W&B Tables
`eval/psnr_table` and `eval/msssim_table` with `(ID)`/`(OOD)` tagged columns.

**Checkpoints**: best model uploaded as a W&B Artifact named
`<experiment.name>-best`, with `in_domain` and `algorithm` recorded in its
metadata so eval mode can re-tag columns even when run later from a different
config.
