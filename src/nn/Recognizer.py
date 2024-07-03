from torch import nn

class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels
    ):
        super(VGGFeatureExtractor, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )

    def forward(self, input):
        return self.ConvNet(input)


class BidirectionalLSTM(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_size
    ):
        super(BidirectionalLSTM, self).__init__()

        self.lstm = nn.LSTM(
            in_features, hidden_size,
            bidirectional = True,
            batch_first = True
        )
        self.out = nn.Linear(hidden_size * 2, out_features)

    def forward(self, x):
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        return self.out(output)


class RecognizerNetwork(nn.Module):
    def __init__(self, cfg):
        super(RecognizerNetwork, self).__init__()
        self.feature_extractor = VGGFeatureExtractor(cfg.img_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((None, 1))
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 512, 512),
            BidirectionalLSTM(512, len(cfg.dict_), 512),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.avgpool(features.permute(0, 3, 1, 2)).squeeze(3)
        rnn_out =  self.rnn(features)
        return rnn_out
