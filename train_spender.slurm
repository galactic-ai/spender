#!/bin/bash
#SBATCH -J spender                  # overridden per-job by submit_all.sh (-J)
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 30:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -A AST25022       # <-- put your TACC project/allocation code here
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=nikhil.garuda@austin.utexas.edu

# Params passed in via:  sbatch --export=ALL,LAT=..,DATADIR=..,TAG=..,OUT=..
: "${LAT:?set LAT}"; : "${DATADIR:?set DATADIR}"; : "${TAG:?set TAG}"; : "${OUT:?set OUT}"
ZMAX="${ZMAX:-0.6}"
BATCH="${BATCH:-256}"
EXTRA="${EXTRA:-}"                  # e.g. EXTRA='-e' for the continuum (emission-masked) run

set -eo pipefail

# environment (mirrors your manual steps)
cd /work/11006/nikhilgaruda/ls6/python-envs/spender_venv
source bin/activate
cd /work/11006/nikhilgaruda/ls6/research/spender
mkdir -p logs

echo "host $(hostname)  gpu ${CUDA_VISIBLE_DEVICES:-NA}"
nvidia-smi -L || true
echo "RUN: latents=$LAT data=$DATADIR tag=$TAG zmax=$ZMAX batch=$BATCH out=$OUT extra='$EXTRA'"

python train/train_DESI.py "$DATADIR" "$OUT" \
    -n "$LAT" -a -s -c -zmax "$ZMAX" -b "$BATCH" -t "$TAG" -v $EXTRA

echo "DONE $OUT"
