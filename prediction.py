from src.utils.labelConverter import LabelConverter
from src.nn.Generator import GeneratorNetwork
from src.utils.misc import generate_noise
from src.utils.PlotUtils import PlotPred
import config as cfg
import argparse
import random
import torch
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help="model weights")
parser.add_argument("-i", "--inp", help="input word")
args = parser.parse_args()

class ImageGenerator:
    def __init__(self, weights = "../../generator_e682.pth"):
        self.device = cfg.device
        self.gen = GeneratorNetwork(cfg).to(self.device)
        self.gen.load_state_dict(torch.load(weights))
        self.plt = PlotPred()

    def __get_noise__(self):
        return generate_noise(cfg.z_dim, 1, self.device)

    def __plot__(self, img, query):
        img = img[0].cpu().detach().numpy().transpose((1, 2, 0))
        img = (img + 1) / 2
        p = self.plt.plot(img*255, query)
        p.save(f"generated_image_{query}.jpg")

    def __call__(self, query):
        labelsconverted = LabelConverter([query], cfg.dict_).to(self.device)
        noise = self.__get_noise__()
        fake = self.gen(noise, labelsconverted)
        self.__plot__(fake, query)

if __name__ == "__main__":
    generator = ImageGenerator(args.weight)
    generator(args.inp)