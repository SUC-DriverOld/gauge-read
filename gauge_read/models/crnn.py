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

        cnn = nn.Sequential()

        conv_specs = [
            # (out_channels, kernel_size, stride, padding, use_batchnorm)
            (64, 3, 1, 1, False),
            (128, 3, 1, 1, False),
            (256, 3, 1, 1, True),
            (256, 3, 1, 1, False),
            (512, 3, 1, 1, True),
            (512, 3, 1, 1, False),
            (512, 2, 1, 0, True),
        ]

        pool_specs = {0: (2, 2, 0), 1: (2, 2, 0), 3: ((2, 2), (2, 1), (0, 1)), 5: ((2, 2), (2, 1), (0, 1))}

        for i, (n_out, k, s, p, use_bn) in enumerate(conv_specs):
            n_in = nc if i == 0 else conv_specs[i - 1][0]
            cnn.add_module(f"conv{i}", nn.Conv2d(n_in, n_out, k, s, p))
            if use_bn:
                cnn.add_module(f"batchnorm{i}", nn.BatchNorm2d(n_out))
            if leakyRelu:
                cnn.add_module(f"relu{i}", nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f"relu{i}", nn.ReLU(True))

            if i in pool_specs:
                k_pool, s_pool, p_pool = pool_specs[i]
                pool_idx = list(pool_specs.keys()).index(i)
                cnn.add_module(f"pooling{pool_idx}", nn.MaxPool2d(k_pool, s_pool, p_pool))

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
