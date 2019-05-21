#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python demo.py \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nh 512 \
--pretrained '/home/gaoxiao/code/crnn.pytorch/expr/IAM_2LSTM_nh512_mag8/0.77512.pth' \
--img_path '/home/gaoxiao/Pictures/a.png'