#!/bin/bash
# Submit the 4 spender training jobs (each gets its own A100 on gpu-a100-small).
# Run from /work/11006/nikhilgaruda/ls6/research/spender :  bash submit_all.sh
# NOTE: all sbatch options MUST come BEFORE the script name; anything after the
# script name is passed as args to the script (this is what broke the first run).
set -euo pipefail
mkdir -p logs

# DESI ---------------------------------------------------------------
sbatch -J sp_desi10 --export=ALL,LAT=10,DATADIR=./DATA/,TAG=chunk1024,OUT=spender_asc_run_10latent_zmax.pt train_spender.slurm
sbatch -J sp_desi15 --export=ALL,LAT=15,DATADIR=./DATA/,TAG=chunk1024,OUT=spender_asc_run_15latent_zmax.pt train_spender.slurm

# Prospector+Cue mocks (data in ./prospector_data, tag cueprospector1024) ----
sbatch -J sp_cue10  --export=ALL,LAT=10,DATADIR=./prospector_data,TAG=cueprospector1024,OUT=spender_cue_10latent_zmax.pt train_spender.slurm
sbatch -J sp_cue15  --export=ALL,LAT=15,DATADIR=./prospector_data,TAG=cueprospector1024,OUT=spender_cue_15latent_zmax.pt train_spender.slurm

# DESI continuum (emission-masked, EXTRA=-e); encode BOTH DESI+mocks through these ----
sbatch -J sp_cont10 --export=ALL,LAT=10,DATADIR=./DATA/,TAG=chunk1024,OUT=spender_desi_cont_10latent_zmax.pt,EXTRA=-e train_spender.slurm
sbatch -J sp_cont15 --export=ALL,LAT=15,DATADIR=./DATA/,TAG=chunk1024,OUT=spender_desi_cont_15latent_zmax.pt,EXTRA=-e train_spender.slurm

echo "submitted 6 jobs; check: squeue -u \$USER"
