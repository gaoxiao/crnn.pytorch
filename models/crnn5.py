import torch.nn as nn

# Single LSTM
# Param#=6236388 before.
# Param#=997990 (or 6236902) now.
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, useDropout=False):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.dropout = nn.Dropout(0.5)
        self.useDropout = useDropout

    def forward(self, input):
        # **output** of shape `(seq_len, batch, num_directions * hidden_size)`
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        if self.useDropout:
            output = self.dropout(output)
        output = output.view(T, b, -1)

        return output

# Small CNN
class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]  # kernel size
        ss = [1, 1, 1, 1, 1, 1, 1]  # stride size
        ps = [1, 1, 1, 1, 1, 1, 0]  # padding size
        nm = [64, 64, 64, 64, 64, 128, 128]  # CNN channel number

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0, True)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3, True)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4, True)
        convRelu(5, True)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(6, True)  # 256x1x26

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(128, nh, nclass, useDropout=False))
        self.dropout = nn.Dropout(0.8)

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        conv = self.dropout(conv)

        # rnn features
        output = self.rnn(conv)

        return output
