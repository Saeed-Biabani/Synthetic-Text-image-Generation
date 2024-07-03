import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def ShowImage(img):
    img = img.cpu().detach().numpy()
    img = np.moveaxis(img, 0, -1)
    
    plt.imshow(img, cmap = "gray")
    plt.savefig("inja.png")

def SaveGrid(imgs, labels, fname):
    for i, (img, label) in enumerate(zip(imgs, labels)):
        img = img.cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img + 1.0) * 127.5
        img = img.astype(np.uint8)
        plt.subplot(4, 4, i+1)
        plt.imshow(img, cmap = "gray")
        plt.title(label)
        plt.axis("off")

    path_ = os.path.join("log", "plots", fname)
    
    plt.savefig(path_)
    plt.close()

def generate_noise(z_len, batch_size, device):
    return torch.randn((batch_size, z_len)).to(device)

def printParams(model, text):
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        params_num.append(np.prod(p.size()))
    print(text.format(sum(params_num)))
