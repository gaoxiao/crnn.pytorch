#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0 python demo.py \
#--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
#--nh 512 \
#--pretrained '/home/xiao/code/crnn.pytorch/expr/AllFont_finetune/0.95231.pth' \
#--img_path '/home/xiao/Pictures/a.png'
#
#CUDA_VISIBLE_DEVICES=0 python demo.py \
#--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
#--nh 512 \
#--pretrained '/home/xiao/code/crnn.pytorch/expr/IAM&AllFont_finetune_IAMval/0.83375.pth' \
#--img_path '/home/xiao/Pictures/a.png'
#
#CUDA_VISIBLE_DEVICES=0 python demo.py \
#--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
#--nh 512 \
#--pretrained '/home/xiao/code/crnn.pytorch/expr/IAM_GEN_512_valIAM_aug/0.77563.pth' \
#--img_path '/home/xiao/Pictures/a.png'


CUDA_VISIBLE_DEVICES=0 python demo.py \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nh 512 \
--pretrained '/home/xiao/model/remote/IAM_GEN_512_valIAM_aug_noise_0.73925.pth' \
--img_path '/home/xiao/Pictures/a.png'

CUDA_VISIBLE_DEVICES=0 python demo.py \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nh 512 \
--pretrained '/home/xiao/model/remote/IAM_GEN_512_valIAM_aug_noise_not_eval_0.74987.pth' \
--img_path '/home/xiao/Pictures/a.png'

CUDA_VISIBLE_DEVICES=0 python demo.py \
--alphabet " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \
--nh 512 \
--pretrained '/home/xiao/model/remote/IAM_GEN_512_valIAM_noise_0.76850.pth' \
--img_path '/home/xiao/Pictures/a.png'