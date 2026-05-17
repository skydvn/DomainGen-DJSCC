#!/usr/bin/env bash
# Run all single-source baselines and DG-DJSCC on CIFAR-10.
#
# Each run trains on one source channel (or all, for DG-DJSCC) and evaluates
# on every channel — giving you the full source-vs-target generalization matrix
# in W&B. Pass extra args (e.g. --no-wandb, --debug-cuda) and they'll forward.
#
# Usage:
#   bash scripts/run_baselines.sh                     # all online
#   bash scripts/run_baselines.sh --no-wandb          # local only
#   ONLY=awgn bash scripts/run_baselines.sh           # just one source
set -euo pipefail

# Run from repo root regardless of where the script was called from.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

EXTRA_ARGS=("$@")

declare -a CONFIGS=(
  "configs/baseline_awgn_only.yaml"
  "configs/baseline_rayleigh_only.yaml"
  "configs/baseline_rician_only.yaml"
  "configs/djscc_wit_awgn_snr10.yaml"
  "configs/dg_djscc_cifar10.yaml"
)

if [[ -n "${ONLY:-}" ]]; then
  CONFIGS=("configs/baseline_${ONLY}_only.yaml")
fi

for cfg in "${CONFIGS[@]}"; do
  echo
  echo "=============================================================="
  echo " Running: ${cfg}"
  echo "=============================================================="
  python main.py --config "${cfg}" --mode train_eval "${EXTRA_ARGS[@]}"
done

echo
echo "All runs finished. Compare results in W&B (group by tag in the dashboard)."
