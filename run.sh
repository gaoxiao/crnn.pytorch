#!/usr/bin/env bash

python train.py --adadelta \
--trainRoot tool/train \
--valRoot tool/val \
--cuda \
--random_sample \
--displayInterval 10 \
--batchSize 64 \
--nepoch 300 \
--valInterval 40 \
--saveInterval 40