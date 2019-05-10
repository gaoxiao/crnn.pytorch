#!/usr/bin/env bash

python train.py --adadelta \
--trainRoot tool/train \
--valRoot tool/val \
--cuda \
--random_sample \
--displayInterval 10 \
--batchSize 64 \
--nepoch 100 \
--valInterval 30 \
--saveInterval 10