import os
import re

import cv2
import lmdb  # install lmdb by "pip install lmdb"
import numpy as np
from pathlib import Path

home = str(Path.home())


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(v) == str:
                v = v.encode()
            txn.put(k.encode(), v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def create(data_path, file, path, imagePathList, labelList, vocb=None):
    file = os.path.join(data_path, file)
    with open(file) as f:
        for l in f:
            tokens = l.split(',')
            if len(tokens) < 2:
                continue
            img = tokens[0].strip()
            txt = ''.join(tokens[1:]).strip()
            txt = re.sub('[|]', '', txt)

            path = os.path.join(data_path, path)
            img = os.path.join(path, img) + '.jpg'
            imagePathList.append(img)
            labelList.append(txt)
            if vocb is not None:
                vocb.update(txt)


def COCO(trainImagePathList, trainLabelList, valImagePathList, valLabelList, vocb):
    data_path = os.path.join(home, 'data/ocr_data/coco')
    create(data_path, 'train_words_gt.txt', 'train_words', trainImagePathList, trainLabelList, vocb)
    create(data_path, 'val_words_gt.txt', 'val_words', valImagePathList, valLabelList)


def Born_Digital(trainImagePathList, trainLabelList, valImagePathList, valLabelList, vocb):
    data_path = '../dataset/train'
    file = os.path.join(data_path, 'gt.txt')
    imagePathList = []
    labelList = []
    with open(file) as f:
        for l in f:
            tokens = l.split(',')
            img = tokens[0].strip()
            img = os.path.join(data_path, img)
            txt = ''.join(tokens[1:]).strip()
            txt = re.sub('["]', '', txt)

            imagePathList.append(img)
            labelList.append(txt)
            vocb.update(txt)
            # print(img, txt)

    split = int(len(imagePathList) * 0.8)
    trainImagePathList.extend(imagePathList[:split])
    trainLabelList.extend(labelList[:split])
    valImagePathList.extend(imagePathList[split:])
    valLabelList.extend(labelList[split:])


def Gen_Handwritten(trainImagePathList, trainLabelList, valImagePathList, valLabelList, vocb):
    data_path = '/home/gaoxiao/code/handwriting-generation'
    gt_dir = os.path.join(data_path, 'gt')
    # img_dir = os.path.join(data_path, 'gen')
    img_dir = '/home/gaoxiao/code/handwriting-generation/gen'
    gt_file = '/home/gaoxiao/code/handwriting-generation/gt/ALL'

    imagePathList = []
    labelList = []
    with open(gt_file) as f:
        for l in f:
            tokens = l.split(',')
            img = tokens[0].strip() + '.png'
            img = os.path.join(img_dir, img)
            txt = ''.join(tokens[1:]).strip()

            if not os.path.isfile(img):
                print('file {} does not exist!'.format(img))
                continue

            imagePathList.append(img)
            labelList.append(txt)
            vocb.update(txt)

    split = int(len(imagePathList) * 0.95)
    trainImagePathList.extend(imagePathList[:split])
    trainLabelList.extend(labelList[:split])
    valImagePathList.extend(imagePathList[split:])
    valLabelList.extend(labelList[split:])


def IAM(trainImagePathList, trainLabelList, valImagePathList, valLabelList, vocb):
    data_path = os.path.join(home, 'data/ocr_data/IAM')
    # img_dir = os.path.join(data_path, 'gen')
    img_dir = os.path.join(data_path, 'word')
    gt_file = os.path.join(data_path, 'words.txt')

    imagePathList = []
    labelList = []
    with open(gt_file) as f:
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            line_split = line.strip().split(' ')
            assert len(line_split) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            file_name_split = line_split[0].split('-')
            img = '{}/{}-{}/{}.png'.format(file_name_split[0], file_name_split[0], file_name_split[1], line_split[0])
            img = os.path.join(img_dir, img)

            # GT text are columns starting at 9
            txt = ' '.join(line_split[8:])

            # check if image is not empty
            if not os.path.getsize(img):
                print('not found img file: {}'.format(img))
                continue

            imagePathList.append(img)
            labelList.append(txt)
            vocb.update(txt)

    split = int(len(imagePathList) * 0.95)
    trainImagePathList.extend(imagePathList[:split])
    trainLabelList.extend(labelList[:split])
    valImagePathList.extend(imagePathList[split:])
    valLabelList.extend(labelList[split:])


if __name__ == '__main__':
    vocb = set()
    trainImagePathList = []
    trainLabelList = []
    valImagePathList = []
    valLabelList = []

    COCO(trainImagePathList, trainLabelList, valImagePathList, valLabelList, vocb)
    # Born_Digital(trainImagePathList, trainLabelList, valImagePathList, valLabelList, vocb)
    # Gen_Handwritten(trainImagePathList, trainLabelList, valImagePathList, valLabelList, vocb)
    IAM(trainImagePathList, trainLabelList, valImagePathList, valLabelList, vocb)

    print('train img: {}, label: {}'.format(len(trainImagePathList), len(trainLabelList)))
    print('val img: {}, label: {}'.format(len(valImagePathList), len(valLabelList)))
    vocb = ''.join(vocb)
    vocb = sorted(vocb)
    print('vocb {}: {}'.format(len(vocb), ''.join(vocb)))

    createDataset('train', trainImagePathList, trainLabelList)
    createDataset('val', valImagePathList, valLabelList)
