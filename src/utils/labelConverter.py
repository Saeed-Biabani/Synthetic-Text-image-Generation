import torch

def convert_word(word, encoding_dict):
    return [encoding_dict.find(c) for c in word]

def LabelConverter(labels, encoding_dict):
    max_len = len(max(labels, key=len))
    final = []
    for label in labels:
        encoding = convert_word(label, encoding_dict) + [0] * (max_len - len(label))
        final.append(encoding)

    return torch.LongTensor(final).transpose(0, 1)