#!/bin/bash

# Installing conda env
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -p ~
# bash ~/Miniconda3-latest-Linux-x86_64.sh

# conda create -n SoccerNetv3
# conda activate SoccerNetv3
# conda install opencv pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# pip install -r requirements.txt

# Downloading SoccerNet-v3D annotation datasets
mkdir -p data/raw
curl -L -o data/raw/SNv3D.csv https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/SNv3D.csv
curl -L -o data/raw/SoccerNet-v3D.zip https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/SoccerNet-v3D.zip
unzip data/raw/SoccerNet-v3D.zip -d data/raw
rm data/raw/SoccerNet-v3D.zip 

# Downloading SoccerNet-v3D benchmark weights (YOLOv11 model) provided by the paper
# mkdir -p weights/benchmark
# curl -L -o weights/benchmark/yolo-sn-ball-opt.pt https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/yolo-sn-ball-opt.pt
# mkdir -p data/train
# mkdir -p data/test
# curl -L -o data/train/SNv3D-train.txt https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/SNv3D-train.txt
# curl -L -o data/test/SNv3D-test.txt https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/SNv3D-test.txt
