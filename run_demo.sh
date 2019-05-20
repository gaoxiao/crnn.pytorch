#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python demo.py \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nh 256 \
--pretrained '/home/xiao/code/crnn.pytorch/expr/_IAM_2LSTM_distortion_0.665875.pth' \
--img_path '/home/xiao/data/ocr_data/IAM/word/a01/a01-000x/a01-000x-03-00.png'