#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python train.py \
--adadelta \
--trainRoot tool/train3 \
--valRoot tool/val \
--cuda \
--random_sample \
--displayInterval 10 \
--valInterval 100 \
--batchSize 400 \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nepoch 200 \
--nh 512 \
--pretrained "/home/xiao/code/crnn.pytorch/expr/Font/0.96115.pth" \
--use_aug \
--trainName "IAM_Font_tune_IAMval_aug"