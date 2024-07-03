from einops.layers.torch import Rearrange
from torch.nn.utils import spectral_norm
from ..attention import SelfAttention
from .Alphabet import AlphabetNetwork
from ..layers import ResUpLayer
from torch import nn
import torch


class GeneratorNetwork(nn.Module):
    def __init__(self, cfg):
        super(GeneratorNetwork, self).__init__()
        self.cfg = cfg
        self.embedding = AlphabetNetwork(
            len(cfg.dict_),
            cfg.emb_size,
            cfg.hiddne_size,
            cfg.padding_index
        )
        self.expand =  nn.Sequential(
            spectral_norm(nn.Linear(16, 1*3*256)),
            Rearrange("b (c h w) -> b c h w", h = 1, w = 3)
        )
        
        self.layer1 = ResUpLayer(256, 256, 128, cfg.cbn_mlp_dim, 1, cfg.batch_size)
        self.layer2 = ResUpLayer(256, 128, 128, cfg.cbn_mlp_dim, 1, cfg.batch_size)
        self.layer3 = ResUpLayer(128, 128, 128, cfg.cbn_mlp_dim, 1, cfg.batch_size)
        self.layer4 = ResUpLayer(128, 64, 128, cfg.cbn_mlp_dim, 1, cfg.batch_size)
        self.layer5 = ResUpLayer(64, 32, 128, cfg.cbn_mlp_dim, 1, cfg.batch_size)
        self.layer6 = ResUpLayer(32, 16, 128, cfg.cbn_mlp_dim, 1, cfg.batch_size)
        
        self.attention = SelfAttention(64)
        self.actf = nn.ReLU()
        self.conv = spectral_norm(nn.Conv2d(16, 1, 3, padding = 1, bias = False))
        self.bn = nn.BatchNorm2d(1)
        self.tanh = nn.Tanh()
    
    def forward(self, noise, labels):
        alphabet_emb = self.embedding(labels)
        noise_splits = torch.split(noise, split_size_or_sections = self.cfg.chunk_size, dim = 1)
        prev = self.expand(noise_splits[0])
        
        prev = self.layer1(noise_splits[1], alphabet_emb, prev)
        prev = self.layer2(noise_splits[2], alphabet_emb, prev)
        prev = self.layer3(noise_splits[3], alphabet_emb, prev)
        prev = self.layer4(noise_splits[4], alphabet_emb, prev)
        prev = self.attention(prev)
        prev = self.layer5(noise_splits[5], alphabet_emb, prev)
        prev = self.layer6(noise_splits[6], alphabet_emb, prev)
        
        out = self.actf(prev)
        out = self.conv(out)
        return self.tanh(self.bn(out))
