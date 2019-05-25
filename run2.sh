#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python train.py \
--adadelta \
--trainRoot tool/train \
--valRoot tool/val \
--cuda \
--random_sample \
--displayInterval 10 \
--valInterval 500 \
--batchSize 400 \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nepoch 200 \
--nh 256 \
--use_aug \
--use_noise \
--pretrained /home/gaoxiao/code/crnn.pytorch/expr/IAM_GEN_256_valIAM_aug_noise/0.77122.pth \
--trainName "IAM_GEN1_256_valIAM_aug_noise"
