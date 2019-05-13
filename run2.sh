#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python train.py \
--adam \
--trainRoot tool/train \
--valRoot tool/val \
--cuda \
--random_sample \
--displayInterval 10 \
--batchSize 400 \
--alphabet " !\"#$%&'()*+-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--pretrained /home/gaoxiao/code/crnn.pytorch/expr/netCRNN_accu_0.4354054054054054.pth \
--nepoch 200
