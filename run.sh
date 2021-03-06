#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py \
--adadelta \
--trainRoot tool/train2 \
--valRoot tool/val2 \
--cuda \
--random_sample \
--displayInterval 10 \
--valInterval 100 \
--batchSize 400 \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nepoch 200 \
--nh 512 \
--pretrained /home/gaoxiao/code/crnn.pytorch/expr/IAM_2LSTM_nh512_mag8/0.77600.pth \
--trainName "IAM&AllFont_finetune_IAMval"