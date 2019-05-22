#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python train.py \
--adadelta \
--trainRoot tool/train1 \
--valRoot tool/val1 \
--cuda \
--random_sample \
--displayInterval 10 \
--valInterval 20 \
--batchSize 400 \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nepoch 200 \
--nh 512 \
--trainName "Font"