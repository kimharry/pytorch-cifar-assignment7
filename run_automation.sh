#!/bin/bash

python main.py --num_epochs 100 --activation relu --data_preprocessing per_channel_mean_std --weight_init kaiming &
python main.py --num_epochs 100 --activation relu --data_preprocessing per_channel_mean_std --weight_init xavier &
python main.py --num_epochs 100 --activation relu --data_preprocessing per_channel_mean_std --weight_init gaussian &

wait

python main.py --num_epochs 100 --activation relu --data_preprocessing per_channel_mean --weight_init kaiming &
python main.py --num_epochs 100 --activation relu --data_preprocessing mean_img --weight_init kaiming &

wait

python main.py --num_epochs 100 --activation leaky_relu --data_preprocessing per_channel_mean_std --weight_init kaiming &
python main.py --num_epochs 100 --activation tanh --data_preprocessing per_channel_mean_std --weight_init kaiming &

wait
