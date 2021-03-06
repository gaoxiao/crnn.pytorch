from __future__ import division
from __future__ import print_function

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

import dataset
import models.crnn2 as crnn
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--trainName', required=True, help='Train name')
parser.add_argument('--trainRoot', required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=128, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default="0123456789abcdefghijklmnopqrstuvwxyz")
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--train_accuracy', action='store_true', help='Whether to calculate train accurate (slow)')
parser.add_argument('--use_aug', action='store_true', help='Use augmentation')
parser.add_argument('--use_noise', action='store_true', help='Use noise')
parser.add_argument('--valInterval', type=int, default=40, help='Interval to be displayed')
opt = parser.parse_args()
print(opt)

PREFIX = opt.trainName
writer = SummaryWriter(log_dir=os.path.join('runs', PREFIX))

model_dir = os.path.join(opt.expr_dir, PREFIX)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio, augmentation=opt.use_aug,
                                    noise=opt.use_noise))
test_dataset = dataset.lmdbDataset(
    root=opt.valRoot, transform=dataset.resizeNormalize((opt.imgW, opt.imgH), augmentation=False, noise=False))

nclass = len(opt.alphabet) + 1
nc = 1

converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    crnn.load_state_dict(torch.load(opt.pretrained))
print(crnn)
total_params = sum(p.numel() for p in crnn.parameters())
print('Total parameter number: {}'.format(total_params))

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

best_accuracy = 0.0


def val(net, dataset, criterion, idx, max_iter=20):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds.cpu(), text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
        # writer.add_text('Text', '%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt), idx)

    accuracy = n_correct / float(max_iter * opt.batchSize)

    writer.add_scalars('data/loss', {'val': loss_avg.val()}, idx)
    writer.add_scalars('data/accuracy', {'val': accuracy}, idx)

    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

    global best_accuracy
    # save model
    if best_accuracy < accuracy:
        best_accuracy = accuracy
        if best_accuracy > 0.35:
            model_path = os.path.join(model_dir, '{:.5f}.pth'.format(best_accuracy))
            print('At epoch {}, iter {}, writing model file to {}'.format(epoch, i, model_path))
            torch.save(crnn.state_dict(), model_path)


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds.cpu(), text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()

    n_correct = 0
    if opt.train_accuracy:
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    return cost, n_correct


for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0
    all_correct = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost, n_correct = trainBatch(crnn, criterion, optimizer)
        all_correct += n_correct
        loss_avg.add(cost)
        i += 1

        idx = epoch * len(train_loader) + i
        writer.add_scalars('data/loss', {'train': loss_avg.val()}, idx)

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()
        if i % opt.valInterval == 0:
            val(crnn, test_dataset, criterion, epoch * len(train_loader), max_iter=20)

    if opt.train_accuracy:
        accuracy = all_correct / float(len(train_loader) * opt.batchSize)
        writer.add_scalars('data/accuracy', {'train': accuracy}, epoch * len(train_loader))

writer.close()
