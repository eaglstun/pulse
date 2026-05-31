#!/usr/bin/env bash
#
# param_sweep.sh — run a fixed set of PULSE parameter variations on ONE input
# image and collect the results under a single folder with a shared prefix.
#
# Every variation uses the same seed and step count, so any difference in the
# output is attributable to the one parameter being changed. This is the script
# used to generate the comparison grids in readme_resources/experiments/ (see
# the README's "Parameter sweep (worked example)" section).
#
# The five variations:
#   baseline       defaults (100*L2+0.05*GEOCROSS, trainable noise, W+)
#   tile_latent    -tile_latent                       (one shared latent, W space)
#   geocross_high  -loss_str "100*L2+1.0*GEOCROSS"    (20x stronger realism prior)
#   no_geocross    -loss_str "100*L2"                 (realism prior removed)
#   noise_fixed    -noise_type fixed                  (noise frozen, not optimized)
#
# Usage:
#   scripts/param_sweep.sh <input_image.png> [output_dir] [prefix]
#
# Examples:
#   scripts/param_sweep.sh input/demo.png
#   scripts/param_sweep.sh input/demo4.png readme_resources/experiments demo4
#
# Environment overrides:
#   PULSE_PYTHON   python interpreter to use         (default: python)
#   SWEEP_SEED     random seed shared by all runs     (default: 42)
#   SWEEP_STEPS    optimization steps per run         (default: 100)
#
# Note: activate the project env first (e.g. `conda activate pulse`) so the
# default `python` has torch/torchvision, or point PULSE_PYTHON at one that does.
set -euo pipefail

IMG="${1:?usage: scripts/param_sweep.sh <input_image.png> [output_dir] [prefix]}"
OUT_DIR="${2:-readme_resources/experiments}"
PREFIX="${3:-$(basename "${IMG%.*}")}"
PY="${PULSE_PYTHON:-python}"
SEED="${SWEEP_SEED:-42}"
STEPS="${SWEEP_STEPS:-100}"

[ -f "$IMG" ] || { echo "input image not found: $IMG" >&2; exit 1; }
mkdir -p "$OUT_DIR"

# run.py processes every *.png in its input dir, so isolate this one image in a
# temp dir. Clean up the temp dir on any exit.
TMP_IN="$(mktemp -d)"
trap 'rm -rf "$TMP_IN"' EXIT
cp "$IMG" "$TMP_IN/"
IMG_NAME="$(basename "$IMG")"

# run_variation <name> [extra run.py args...]
run_variation() {
  local name="$1"; shift
  local out line
  out="$(mktemp -d)"
  line="$("$PY" run.py -input_dir "$TMP_IN" -seed "$SEED" -steps "$STEPS" \
            -output_dir "$out" "$@" 2>&1 | grep BEST || true)"
  cp "$out/$IMG_NAME" "$OUT_DIR/${PREFIX}_${name}.png"
  rm -rf "$out"
  printf '%-14s | %s\n' "$name" "$line"
}

echo "Sweeping $IMG  (seed=$SEED steps=$STEPS)  ->  $OUT_DIR/${PREFIX}_*.png"
run_variation baseline
run_variation tile_latent   -tile_latent
run_variation geocross_high -loss_str "100*L2+1.0*GEOCROSS"
run_variation no_geocross   -loss_str "100*L2"
run_variation noise_fixed   -noise_type fixed
echo "Done -> $OUT_DIR/${PREFIX}_*.png"
