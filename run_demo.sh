#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python demo.py \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nh 512 \
--pretrained '/home/xiao/code/crnn.pytorch/expr/Font_aug/0.91077.pth' \
--img_path '/home/xiao/Pictures/a.png'