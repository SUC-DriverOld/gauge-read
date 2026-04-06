import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, inputs):
        recurrent, _ = self.rnn(inputs)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, "imgH has to be a multiple of 16"

        def activation():
            return nn.LeakyReLU(0.2, inplace=True) if leakyRelu else nn.ReLU(True)

        cnn = nn.Sequential()

        cnn.add_module("conv0", nn.Conv2d(nc, 64, 3, 1, 1))
        cnn.add_module("relu0", activation())
        cnn.add_module("pooling0", nn.MaxPool2d(2, 2, 0))

        cnn.add_module("conv1", nn.Conv2d(64, 128, 3, 1, 1))
        cnn.add_module("relu1", activation())
        cnn.add_module("pooling1", nn.MaxPool2d(2, 2, 0))

        cnn.add_module("conv2", nn.Conv2d(128, 256, 3, 1, 1))
        cnn.add_module("batchnorm2", nn.BatchNorm2d(256))
        cnn.add_module("relu2", activation())

        cnn.add_module("conv3", nn.Conv2d(256, 256, 3, 1, 1))
        cnn.add_module("relu3", activation())
        cnn.add_module("pooling2", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))

        cnn.add_module("conv4", nn.Conv2d(256, 512, 3, 1, 1))
        cnn.add_module("batchnorm4", nn.BatchNorm2d(512))
        cnn.add_module("relu4", activation())

        cnn.add_module("conv5", nn.Conv2d(512, 512, 3, 1, 1))
        cnn.add_module("relu5", activation())
        cnn.add_module("pooling3", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))

        cnn.add_module("conv6", nn.Conv2d(512, 512, 2, 1, 0))
        cnn.add_module("batchnorm6", nn.BatchNorm2d(512))
        cnn.add_module("relu6", activation())

        self.cnn = cnn
        self.rnn = nn.Sequential(BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass))

    def forward(self, inputs):
        # conv features
        conv = self.cnn(inputs)

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)

        return output
