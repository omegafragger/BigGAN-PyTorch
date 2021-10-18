#!/bin/bash

#SBATCH --gres=gpu:1 #SBATCH --cpus-per-task=5
#SBATCH --job-name="BigGAN CIFAR"
#SBATCH --time=10-00:00:00

#SBATCH --account=engs-tvg
#SBATCH --reservation=tvg202110
#SBATCH --qos=tvg
#SBATCH --partition=short

source activate py36

CUDA_VISIBLE_DEVICES=0,1 python train.py \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--ensemble_path /data/coml-ecr/shug5721/ensemble/ --lamda1 1.0 --lamda2 3.0 \
--weights_root /data/coml-ecr/shug5721/biggan_models/biggan_cifar_3.0/weights \
--samples_root /data/coml-ecr/shug5721/biggan_models/biggan_cifar_3.0/samples