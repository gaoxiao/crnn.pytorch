#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python train.py \
--adadelta \
--trainRoot tool/train1 \
--valRoot tool/val1 \
--cuda \
--random_sample \
--displayInterval 10 \
--valInterval 100 \
--batchSize 400 \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nepoch 200 \
--nh 512 \
--pretrained "/home/gaoxiao/code/crnn.pytorch/expr/IAM&AllFont_finetune_IAMval/0.82463.pth" \
--use_aug \
--trainName "IAM&AllFont_finetune_IAMval_aug"