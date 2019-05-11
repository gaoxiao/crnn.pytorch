#!/usr/bin/env bash

python train.py --adadelta \
--trainRoot tool/train \
--valRoot tool/val \
--cuda \
--random_sample \
--displayInterval 100 \
--batchSize 64 \
--nepoch 25 \
--valInterval 100 \
--saveInterval 200