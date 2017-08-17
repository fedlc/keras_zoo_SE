#!/bin/bash

# execute with ./run.sh free_gpu_device experiment_name (example: ./run.sh 1 prova_exp)
# check free gpu devices with nvidia-smi
# dont forget to change configuration file and user paths to save experiments

# CUDA_VISIBLE_DEVICES=$1 python train.py --config_path config/camvid_adversarial_semseg.py --exp_name $2 --shared_path /home/user/experiments --local_path /home/user/experiments


# CUDA_VISIBLE_DEVICES=$1 python train.py --config_path config/camvid.py --exp_name $2 --shared_path /home/flucchesi/experiments --local_path /home/flucchesi/experiments
CUDA_VISIBLE_DEVICES=$1 python train.py --config_path config/camvid.py --exp_name $2 --shared_path /datatmp/flucchesi/experiments --local_path /datatmp/flucchesi/experiments
