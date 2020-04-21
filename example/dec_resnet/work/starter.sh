#!/bin/bash
#conda install pytorch-gpu torchvision pip -y
#pip install -r requirements.txt
python setup.py

# move weights to current directory
# Mind the size of the weights file
cp ../results/*.pth .

export CUDA_VISIBLE_DEVICES=1

# Mind the last '/' of datapath 
## sample dataset
#python3 main.py --train --data ../data/sample/Retina/
#python3 main.py --data ../data/sample/Retina/  

## real dataset
python3 main.py --train --data ../data/real/Retina/ # > ../logs/training.log
python3 main.py --data ../data/real/Retina/ > ../logs/testing.log 

# mv weights and plot
mv *.pth ../results/
mv *.png ../results/

rm -rf __pycache__
