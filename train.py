from src.utils.transforms import Resize, Rescale, Normalization
from src.nn.Discriminator import DiscriminatorNetwork
from src.utils.labelConverter import LabelConverter
from src.utils.dataProvider import DataGenerator
from src.nn.Recognizer import RecognizerNetwork
from src.nn.Generator import GeneratorNetwork
from torchvision.transforms import Compose
from src.utils.misc import generate_noise
from torch.utils.data import DataLoader
from src.utils.misc import printParams
from src.utils.misc import SaveGrid
from src.utils.calculator import *
import config as cfg
from torch import nn
import torchvision
import torch
import tqdm

def printConfigVars(module, fname):
    pa = [item for item in dir(module) if not item.startswith("__")]
    for item in pa:
        value = eval(f'{fname}.{item}')
        if str(type(value)) not in ("<class 'module'>", "<class 'function'>"):
            print(f"{fname}.{item} : {eval(f'{fname}.{item}')}")

device = cfg.device

trainds = DataGenerator(
    root = cfg.ds_path["train_ds"],
    transforms = Compose([
        Resize((cfg.img_h, cfg.img_w)),
        Rescale(1./255),
        Normalization()
    ])
); trian_dataloader = DataLoader(trainds, cfg.batch_size, True)

print(f"Len DS : {len(trainds)}")
printConfigVars(cfg, 'cfg')

dis = DiscriminatorNetwork().to(device)
printParams(dis, "Discriminator Trainable Params : {:,}")
dis_opt = torch.optim.Adam(dis.parameters(), lr=cfg.learning_rate, betas=cfg.betas)

gen = GeneratorNetwork(cfg).to(device)
printParams(gen, "Generator Trainable Params : {:,}")
gen_opt = torch.optim.Adam(gen.parameters(), lr=cfg.learning_rate, betas=cfg.betas)

rec = RecognizerNetwork(cfg).to(device)
printParams(rec, "Recognizer Trainable Params : {:,}\n")
rec_opt = torch.optim.Adam(rec.parameters(), lr=cfg.learning_rate, betas=cfg.betas)

fixed_noise = generate_noise(cfg.z_dim, 64, device)

stddev = 1

for epoch in range(cfg.epochs):
    loop = tqdm.tqdm(trian_dataloader)
    for batch_indx, (imgs, labels, len_gt) in enumerate(loop):
        imgs = imgs.to(device)
        labelsconverted = LabelConverter(labels, cfg.dict_).to(device)
        len_gt = torch.LongTensor(len_gt).to(device)
        ctc_labels = labelsconverted.transpose(0, 1).to(device)
        
        noise = torch.distributions.normal.Normal(0, stddev)
        stddev -= 0.00000125
        
        dis_opt.zero_grad()
        rec_opt.zero_grad()

        z_dis = generate_noise(cfg.z_dim, imgs.size(0), device)
        gen_out = gen(z_dis, labelsconverted)
        
        dis_out_fake = dis(gen_out + noise.sample(gen_out.shape).to(device))
        dis_out_real = dis(imgs + noise.sample(gen_out.shape).to(device))
        
        rec_out_dis = rec(imgs)
        dis_loss = dis_criterion(dis_out_fake, dis_out_real)
        rec_loss = calc_ctc_loss(rec_out_dis, ctc_labels, len_gt)
        
        dis_loss.backward()
        rec_loss.backward()

        dis_opt.step()
        rec_opt.step()
        
        gen_opt.zero_grad()

        # z = generate_noise(cfg.z_dim, imgs.size(0), device)
        gen_out = gen(z_dis, labelsconverted)
        rec_out = rec(gen_out)
        dis_out = dis(gen_out + noise.sample(gen_out.shape).to(device))

        ctc = calc_ctc_loss(rec_out, ctc_labels, len_gt)
        gen_loss = gen_criterion(dis_out, ctc)

        gen_loss.backward()

        gen_opt.step()
        
        __log = {
            "epoch" : epoch + 1,
            "ocr_loss" : rec_loss.item(),
            "discriminator_loss" : dis_loss.item(),
            "generator_loss" : gen_loss.item(),
        }
        loop.set_postfix(__log)
        
        if batch_indx % 200 == 0:
            with torch.no_grad():
                fake = gen(fixed_noise, labelsconverted)
                
                fake_grid = torchvision.utils.make_grid(
                    [fake[:16]], normalize = True
                )
                
                SaveGrid(fake_grid, labels[:16], f"plot_fake_{epoch+1}_{batch_indx+1}.png")
    torch.save(gen.state_dict(), "generator.pth")
    torch.save(dis.state_dict(), "discriminator.pth")
    torch.save(rec.state_dict(), "recognizer.pth")