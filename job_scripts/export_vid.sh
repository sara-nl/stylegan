#!/bin/bash

#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -p gpu

# Go to workdir --> CHANGE THIS TO YOUR OWN HOME DIRECTORY <--
workdir= /home/veefkind
cd $workdir

# Load environment CHANGE THIS TO YOUR LOCATION DIRECTORY
source script/install_prog_gan_env.sh

# Run training
cd stylegan
python export_video.py