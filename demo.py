import argparse

import Augmentor
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import transforms

import utils
import dataset
from PIL import Image

import models.crnn2 as crnn


parser = argparse.ArgumentParser()
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--pretrained', default='/home/xiao/code/crnn.pytorch/expr/_IAM_2LSTM_distortion_0.665875.pth', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default=" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€")
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--img_path', type=str, default='/home/xiao/data/ocr_data/IAM/word/a01/a01-000x/a01-000x-03-00.png')
opt = parser.parse_args()

# model_path = './data/crnn.pth'
model_path = opt.pretrained
img_path = opt.img_path
alphabet = " \"#$%&'()*+-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_\`abcdefghijklmnopqrstuvwxyz~¡¢£©®°ÇÉ•€★" \

nclass = len(opt.alphabet) + 1
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

# model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path)
print(image.format, image.size, image.mode)
im2arr = np.array(image)
print(im2arr.shape)

image = image.convert('L')
print(image.format, image.size, image.mode)
im2arr = np.array(image)
print(im2arr.shape)

p = Augmentor.Pipeline()
# p.gaussian_distortion(probability=0.4, grid_width=7, grid_height=6
#                       , magnitude=6, corner="ul", method="in", mex=0.5, mey=0.5, sdx=0.05, sdy=0.05)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
transform_train = transforms.Compose([
    transforms.Resize((32, 100), Image.BILINEAR),
    p.torch_transform(),
])

image2 = transform_train(image)
image2.save('tmp.png')

image = transformer(image)

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
