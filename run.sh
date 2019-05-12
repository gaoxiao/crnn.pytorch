#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py \
--adadelta \
--trainRoot tool/train \
--valRoot tool/val \
--cuda \
--random_sample \
--displayInterval 10 \
--batchSize 100 \
--nepoch 200 \
--valInterval 100 \
--alphabet " !\"#$%&'()*+-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--pretrained /home/xiao/code/crnn.pytorch/expr/netCRNN_accu_0.3332432432432432.pth \
--saveInterval 100