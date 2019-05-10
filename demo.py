import numpy as np
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn

# model_path = './data/crnn.pth'
model_path = '/home/xiao/code/crnn.pytorch/expr/netCRNN_99_40.pth'
img_path = './data/a.png'
# img_path = '/home/xiao/code/crnn.pytorch/dataset/train/word_865.png'
# alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
alphabet = " !$%&'()*+-./0123456789:?@ABCDEFGHIJKLMNOPQRSTUVWXYZ\_abcdefghijklmnopqrstuvwxyz£®Ç€"

model = crnn.CRNN(32, 1, len(alphabet) + 1, 256)
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
