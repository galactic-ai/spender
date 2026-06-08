#!/bin/bash
# Submit the 4 spender training jobs (each gets its own A100 on gpu-a100-small).
# Run from /work/11006/nikhilgaruda/ls6/research/spender :  bash submit_all.sh
set -euo pipefail
mkdir -p logs
S="sbatch train_spender.slurm"

# DESI ---------------------------------------------------------------
$S -J sp_desi10 --export=ALL,LAT=10,DATADIR=./DATA/,TAG=chunk1024,OUT=spender_asc_run_10latent_zmax.pt
$S -J sp_desi15 --export=ALL,LAT=15,DATADIR=./DATA/,TAG=chunk1024,OUT=spender_asc_run_15latent_zmax.pt

# Prospector+Cue mocks (data in ./prospector_data, tag cueprospector1024) ----
$S -J sp_cue10  --export=ALL,LAT=10,DATADIR=./prospector_data,TAG=cueprospector1024,OUT=spender_cue_10latent_zmax.pt
$S -J sp_cue15  --export=ALL,LAT=15,DATADIR=./prospector_data,TAG=cueprospector1024,OUT=spender_cue_15latent_zmax.pt

echo "submitted 4 jobs; check: squeue -u \$USER"
