#!/bin/bash

if [ ! -d ~/miniconda3 ]; then
    echo "=== [Installing conda...] ==="

    mkdir -p ~/miniconda3
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    
    ls ~/miniconda3

    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    bash ~/miniconda3/bin/activate
    
    export PATH=$PATH:~/miniconda3/bin/
fi

conda init
conda env create --file environment.yml
conda activate ISC-Football
# python -m ipykernel install --user --name=ISC-Football