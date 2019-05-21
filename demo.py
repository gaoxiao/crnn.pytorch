import argparse

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import functional as F
from torchvision.transforms import transforms

import dataset
import models.crnn2 as crnn
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--pretrained', default='/home/gaoxiao/code/crnn.pytorch/expr/IAM_2LSTM_nh512_mag8/0.77512.pth',
                    help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str,
                    default=" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€")
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--img_path', type=str,
                    default='/home/gaoxiao/data/ocr_data/IAM/word/a01/a01-000x/a01-000x-03-00.png')
opt = parser.parse_args()

model_path = opt.pretrained
img_path = opt.img_path
alphabet = opt.alphabet
nclass = len(alphabet) + 1
nc = 1
model = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
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

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32), augmentation=False)
image = Image.open(img_path)
print(image.format, image.size, image.mode)
im2arr = np.array(image)
print(im2arr.shape)

image = image.convert('L')
print(image.format, image.size, image.mode)
im2arr = np.array(image)
print(im2arr.shape)

transform_train = transforms.Compose([
    lambda img: F.adjust_contrast(img, 5),
    transforms.Resize((32, 100), Image.BILINEAR),
])
image2 = transform_train(image)
image2.save('tmp.png')

transform_train = transforms.Compose([
    lambda img: F.adjust_contrast(img, 5),
    transformer,
])

image = transform_train(image)

if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
