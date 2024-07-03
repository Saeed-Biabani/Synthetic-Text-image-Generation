import numpy as np
import torch
import json
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_ds_info(ds_path_list):
    max_str_len = 0
    total = []
    for path_ in ds_path_list:
        with open(
            os.path.join(
                path_, "labels.json"
            ), 'r'
        ) as file_:
            labels = json.load(file_)
            total += list(labels.values())

    max_str_len = len(
        max(total, key = lambda x : len(x))
    )
    dict_ = ''.join(
        np.unique(list(''.join(total)))
    ).replace(' ', '') + '_'

    return ' '+dict_, max_str_len

ds_path = {
    "train_ds" : "../../../Desktop/Roshd/OCR/DataGeneration/puGanDS" 
}
dict_, max_str_len = extract_ds_info(ds_path.values())
batch_size = 64
img_h = 64
img_w = 192
img_channel = 1
emb_size = 112
z_dim = 112
chunk_size = 16
padding_index = 0
hiddne_size = emb_size // 2
cbn_mlp_dim = 512
learning_rate = 2e-4
betas = (0.5, 0.999)
epochs = 1000
