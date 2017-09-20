#!/bin/bash

# execute with ./run.sh free_gpu_device experiment_name (example: ./run.sh 1 prova_exp)
# check free gpu devices with nvidia-smi
# dont forget to change configuration file and user paths to save experiments

# CUDA_VISIBLE_DEVICES=$1 python train.py --config_path config/camvid_adversarial_semseg.py --exp_name $2 --shared_path /home/user/experiments --local_path /home/user/experiments


cd ../runs/
rm -rf $2_$3
sleep 1s
mkdir $2_$3
cd ..
cp -r keras_zoo_SE/ runs/$2_$3/
sleep 2s
cd runs/$2_$3/keras_zoo_SE/


CUDA_VISIBLE_DEVICES=$1 python train.py -c config/cityscapes.py -e $2 -s /home/flucchesi/experiments -l /home/flucchesi/experiments
#CUDA_VISIBLE_DEVICES=$1 python train.py -c config/cityscapes.py -e $2 -s /datatmp/flucchesi/experiments -l /datatmp/flucchesi/experiments
