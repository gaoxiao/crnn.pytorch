#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python demo.py \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nh 512 \
--pretrained '/home/gaoxiao/code/crnn.pytorch/expr/AllFont_finetune/0.95231.pth' \
--img_path '/home/gaoxiao/Pictures/a.png'

CUDA_VISIBLE_DEVICES=0 python demo.py \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nh 512 \
--pretrained '/home/gaoxiao/code/crnn.pytorch/expr/IAM&AllFont_finetune/0.83925.pth' \
--img_path '/home/gaoxiao/Pictures/a.png'

CUDA_VISIBLE_DEVICES=0 python demo.py \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nh 512 \
--pretrained '/home/gaoxiao/code/crnn.pytorch/expr/IAM&AllFont_finetune_IAMval/0.83375.pth' \
--img_path '/home/gaoxiao/Pictures/a.png'

CUDA_VISIBLE_DEVICES=0 python demo.py \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nh 512 \
--pretrained '/home/gaoxiao/code/crnn.pytorch/expr/IAM_AllFont_finetune_IAMval_aug/0.81923.pth' \
--img_path '/home/gaoxiao/Pictures/a.png'



CUDA_VISIBLE_DEVICES=0 python demo.py \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nh 512 \
--pretrained '/home/gaoxiao/code/crnn.pytorch/expr/IAM_512_aug/0.78800.pth' \
--img_path '/home/gaoxiao/Pictures/a.png'

CUDA_VISIBLE_DEVICES=0 python demo.py \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nh 512 \
--pretrained '/home/gaoxiao/code/crnn.pytorch/expr/IAM_GEN_512_valIAM_aug/0.77563.pth' \
--img_path '/home/gaoxiao/Pictures/a.png'
