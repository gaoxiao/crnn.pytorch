#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python train.py \
--adadelta \
--trainRoot tool/train \
--valRoot tool/val \
--cuda \
--random_sample \
--displayInterval 10 \
--valInterval 200 \
--batchSize 400 \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nepoch 200 \
--nh 512 \
--use_aug \
--use_noise \
--pretrained /home/gaoxiao/code/crnn.pytorch/expr/IAM_2LSTM_nh512_mag8/0.77600.pth \
--trainName "IAM_512_valIAM_aug_noise"