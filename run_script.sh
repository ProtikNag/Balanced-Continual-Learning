#!/bin/bash

# -------------------------------------------------------
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128

# -------------------------------------------------------
# The following is for running on JLSE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax__

python main.py

conda deactivate
