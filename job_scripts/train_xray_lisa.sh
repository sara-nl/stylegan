#!/bin/bash

#SBATCH -t 4-00:00:00
#SBATCH -N 1
#SBATCH -p gpu

# Copy dataset to scratch
mkdir "$TMPDIR"/CHESTXRAY
cp -r /nfs/managed_datasets/CHESTXRAY/tfrecords "$TMPDIR"/CHESTXRAY

# Go to workdir --> CHANGE THIS TO YOUR OWN HOME DIRECTORY <--
workdir= /home/veefkind
cd $workdir

# Load environment CHANGE THIS TO YOUR LOCATION DIRECTORY
source script/install_prog_gan_env.sh

# Run training
cd stylegan
python train.py