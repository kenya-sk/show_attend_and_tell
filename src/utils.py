import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pickle
import nltk
nltk.download('punkt')
import torchvision.transforms as transforms
from torch.autograd import Variable


class Vocabulary(object):
    """
    Vocabulary class manage words and corresponding indexes.
    If input the not registered word, return <unk>.

    word2idx : dictionary which converts word to index.
    idx2word : dictionary which converts index to word.
    idx      : keep current index.

    method:
        add_word: To register a new word in vocabulay class.
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if word not in self.word2idx.keys():
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        else:
            return self.word2idx[word]
        
    def __len__(self):
        return self.idx


class AverageMeter(object):
    """
    AverageMeter class manage statistics value about one variable.

    val : current value.
    avg : average of past input values.
    sum : summation of past input values.
    cnt : the number of element size.

    method:
        reset  : To set all statistics value to 0.
        update : To update all statistics value by input value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def tensor2numpy(x):
    """
    Convert tensor to numpy and variable move GPU to CPU. 
    """

    return x.data.cpu().numpy()


def to_variable(x):
    """
    If the device can be used GPU, to move the variable.
    """

    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def set_transform(resize=(256,256), crop_size=(224,224), horizontal_flip=False, normalize=True):
    compose_lst = []
    compose_lst.append(transforms.Resize(resize)) if resize is not None
    compose_lst.append(transforms.RandomCrop(crop_size)) if crop_size is not None
    compose_lst.append(transforms.RandomHorizontalFlip()) if horizontal_flip
    compose_lst.append(transforms.ToTensor())
    compose_lst.append(transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))) if normalize

    transform = transforms.Compose(compose_lst)
    
    return transform


def decode_caption(captions_lst, idx2word):
    def decoded(cap, idx2word):
        decoded_lst = []
        for wid in range(len(cap)):
            word = idx2word[cap[wid]]
            if word == '<end>':
                break
            decoded_lst.append(word)
        decoded_cap = "".join(decoded_lst)
        return decoded_cap

    decoded_cap_lst = []
    for caption in captions_lst:
        # convert tensor to numpy
        caption = caption.data.cpu().numpy()
        decoded_cap_lst.append(decoded(caption, idx2word))

    return decoded_cap_lst


def clip_gradient(optimizer, grad_clip):
    for params in optimizer.param_groups:
        for param in params["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, decay_rate):
    """
    decay learning rate by specified rate.
    
    optimizer  : optimizer whose learning rate must be decay
    decay_rate : rate in interval (0.0, 1.0) to multiply learning rate
    """

    for params in optimizer.param_groups:
        params["lr"] = params["lr"] * decay_rate


        
