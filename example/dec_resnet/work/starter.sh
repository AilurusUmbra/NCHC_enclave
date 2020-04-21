#!/bin/bash
#conda install pytorch-gpu torchvision pip -y
#pip install -r requirements.txt
python setup.py

# move weights to current directory
# Mind the size of the weights file
cp ../results/*.pkl .

export CUDA_VISIBLE_DEVICES=1

# Mind the last '/' of datapath 
## sample dataset
#python main.py --train --data ../data/sample/Retina/
#python main.py --data ../data/sample/Retina/  

## real dataset
python main.py --train --data ../data/real/Retina/ # > ../logs/training.log
python main.py --data ../data/real/Retina/ > ../logs/testing.log 

# mv weights and plot
mv *.pkl ../results/
mv *.png ../results/

rm -rf __pycache__
