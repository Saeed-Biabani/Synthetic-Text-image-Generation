from torch.nn.utils import spectral_norm
from torch.nn import functional as nnF
from torch import nn
import torch

class ConditionalBatchNorm(nn.Module):
    def __init__(
        self,
        lstm_size,
        emb_size,
        out_size,
        batch_size,
        channels = 1,
        height = 1,
        width = 3
    ):
        super(ConditionalBatchNorm, self).__init__()
        self.lstm_size = lstm_size
        self.emb_size = emb_size
        self.out_size = out_size

        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width

        self.betas = nn.Parameter(torch.zeros(self.batch_size, self.channels))
        self.gammas = nn.Parameter(torch.ones(self.batch_size, self.channels))
        self.eps = 1.0e-5

        self.fc_gamma = nn.Sequential(
            nn.Linear(self.lstm_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
        )

        self.fc_beta = nn.Sequential(
            nn.Linear(self.lstm_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def create_cbn_input(self, lstm_emb):
        delta_betas = self.fc_beta(lstm_emb)
        delta_gammas = self.fc_gamma(lstm_emb)
        return delta_betas, delta_gammas

    def forward(self, feature, lstm_emb):
        self.batch_size, self.channels, self.height, self.width = feature.data.shape

        delta_betas, delta_gammas = self.create_cbn_input(lstm_emb)

        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()

        betas_cloned += delta_betas
        gammas_cloned += delta_gammas

        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)

        betas_expanded = torch.stack([betas_cloned] * self.height, dim=2)
        betas_expanded = torch.stack([betas_expanded] * self.width, dim=3)

        gammas_expanded = torch.stack([gammas_cloned] * self.height, dim=2)
        gammas_expanded = torch.stack([gammas_expanded] * self.width, dim=3)

        feature_normalized = (feature - batch_mean) / torch.sqrt(batch_var + self.eps)

        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return out


class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResLayer, self).__init__()
        self.__block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            nn.BatchNorm2d(out_channels),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        )
        self.residual = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        initial = torch.zeros_like(x)
        initial = initial.copy_(x)
        
        x = self.__block(x)

        return x + self.residual(initial)


class ResDownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDownLayer, self).__init__()
        self.__block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            nn.BatchNorm2d(out_channels),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        )
        self.residual = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        initial = torch.zeros_like(x)
        initial = initial.copy_(x)

        x = self.__block(x)

        return x + nnF.avg_pool2d(self.residual(initial), 2, 2)


class ResUpLayer(nn.Module):
    def __init__(self, in_channels, out_channels, cbn_in, emb_size, cbn_hidden, batch_size):
        super(ResUpLayer, self).__init__()
        self.cbn_1 = ConditionalBatchNorm(cbn_in, emb_size, cbn_hidden, batch_size)
        self.cbn_2 = ConditionalBatchNorm(cbn_in, emb_size, cbn_hidden, batch_size)
        self.act_1 = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
        self.conv_1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
        self.act_2 = nn.ReLU()
        self.conv_2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.residual = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, noise, emb, prev):
        emb_cat = torch.cat((emb, noise), dim=1)
        
        x = self.cbn_1(prev, emb_cat)
        x = self.act_1(x)
        x = self.upsample(x)
        x = self.conv_1(x)
        x = self.cbn_1(x, emb_cat)
        x = self.act_2(x)
        x = self.conv_2(x)

        return x + self.residual(nnF.interpolate(prev, scale_factor = 2, mode = "nearest"))