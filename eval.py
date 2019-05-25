import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torch.autograd import Variable

import dataset
import models.crnn2 as crnn
import utils

home = str(Path.home())

parser = argparse.ArgumentParser()
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--alphabet', type=str, default="")
opt = parser.parse_args()

alphabet = opt.alphabet
converter = utils.strLabelConverter(alphabet)
transformer = dataset.resizeNormalize((100, 32), augmentation=False, noise=False)


def read_model(model_path, hidden_number):
    nclass = len(alphabet) + 1
    nc = 1
    model = crnn.CRNN(opt.imgH, nc, nclass, hidden_number)
    if torch.cuda.is_available():
        print('using GPU')
        model = model.cuda()
    print('loading pretrained model from %s' % model_path)

    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


def get_me_imgs(img_path):
    imgs = []
    for f in os.listdir(img_path):
        if not f.endswith('png'):
            continue
        gt = os.path.splitext(f)[0]
        f = os.path.join(img_path, f)
        imgs.append((gt, f))
    return imgs


def get_font_imgs(img_path):
    gt_file = os.path.join(img_path, 'gt/ALL.txt')
    img_dir = os.path.join(img_path, 'img')

    imgs = []

    with open(gt_file) as f:
        for l in f:
            tokens = l.split(',')
            img = tokens[0].strip() + '.png'
            img = os.path.join(img_dir, img)
            txt = ''.join(tokens[1:]).strip()

            if not os.path.isfile(img):
                print('file {} does not exist!'.format(img))
                continue
            imgs.append((txt, img))

    return imgs


def run_model(model_path, hidden_number):
    model = read_model(model_path, hidden_number)
    model.eval()

    # imgs = get_imgs(os.path.join(home, 'data/ocr_data/test_iam'))
    imgs = get_me_imgs(os.path.join(home, 'data/ocr_data/me'))
    # imgs = get_font_imgs(os.path.join(home, 'data/ocr_data/font_gen1'))

    t_cnt = 0
    f_cnt = 0

    for gt, img_path in imgs:
        image = Image.open(img_path)
        image = image.convert('L')
        image = transformer(image)

        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        preds = model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        if sim_pred != gt:
            f_cnt += 1
            # print('%-20s => %-20s, gt: %s' % (raw_pred, sim_pred, gt))
        else:
            t_cnt += 1

    print('Model: {}\nTrue: {}, False: {}'.format(model_path, t_cnt, f_cnt))


def main():
    models = {
        '/home/xiao/model/remote/IAM_GEN_512_valIAM_aug_noise_0.73925.pth': 512,
        '/home/xiao/model/remote/IAM_GEN_512_valIAM_aug_noise_not_eval_0.74987.pth': 512,
        # '/home/xiao/model/remote/IAM_GEN_512_valIAM_noise_0.76850.pth': 512,
        # '/home/xiao/model/remote/IAM_GEN_512_valIAM_0.81923.pth': 512,
        # '/home/xiao/model/remote/IAM_512_valIAM_aug_noise_0.74775.pth': 512,
        # '/home/xiao/model/remote/IAM_GEN_256_valIAM_aug_noise_0.75262.pth': 256,
        '/home/xiao/model/remote/IAM_GEN_256_valIAM_aug_noise_0.75738.pth': 256,

        # '/home/xiao/code/crnn.pytorch/expr/Font_aug/0.91077.pth': 512,
        # '/home/xiao/code/crnn.pytorch/expr/IAM_dist_randColor/0.72300.pth': 512,
    }
    for model_path in models:
        run_model(model_path, models[model_path])


if __name__ == '__main__':
    main()
