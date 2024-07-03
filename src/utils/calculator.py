import torch.nn.functional as nnF
import torch

def calc_ctc_loss(preds, targets, tg_len):
    preds_size = torch.IntTensor([preds.size(1)] * preds.size(0))
    preds = preds.log_softmax(2).permute(1, 0, 2)
    cost = nnF.ctc_loss(preds, targets, preds_size, tg_len, zero_infinity = True, blank = 0)
    return cost

def dis_criterion(fake_op, real_op):
    return torch.mean(nnF.relu(1.0 - real_op)) + torch.mean(nnF.relu(1.0 + fake_op))

def gen_criterion(dis_preds, ctc_loss):
    return ctc_loss - torch.mean(dis_preds)