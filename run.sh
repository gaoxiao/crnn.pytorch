#!/usr/bin/env bash

python train.py --adadelta \
--trainRoot tool/train \
--valRoot tool/val \
--cuda \
--random_sample \
--displayInterval 10 \
--batchSize 100 \
--nepoch 100 \
--valInterval 100 \
--alphabet " !\"#$%&'()*+-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--pretrained expr/netCRNN_24_100.pth \
--saveInterval 100