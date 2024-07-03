from torch.nn.utils import spectral_norm
from ..attention import SelfAttention
from ..layers import ResDownLayer
from torch import nn

class DiscriminatorNetwork(nn.Module):
    def __init__(self):
        super(DiscriminatorNetwork, self).__init__()
        self.resblock1 = ResDownLayer(1, 16)
        self.resblock2 = ResDownLayer(16, 32)
        self.resblock3 = ResDownLayer(32, 64)
        self.resblock4 = ResDownLayer(64, 128)
        self.resblock5 = ResDownLayer(128, 128)
        self.resblock6 = ResDownLayer(128, 256)

        self.self_attention = SelfAttention(64)
        self.flatten = nn.Flatten(1)
        self.global_sum_pooling = nn.LPPool2d(norm_type=1, kernel_size=(1, 3))
        self.dense = spectral_norm(nn.Linear(256, 1))

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.self_attention(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.global_sum_pooling(x)
        x_flat = self.flatten(x)
        return self.dense(x_flat).view(-1)
