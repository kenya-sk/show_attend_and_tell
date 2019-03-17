import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable

from utils import caption_tensor


# Encoder-Decoder model data loder
class ImageDataset(data.Dataset):
    def __init__(self, root_dirc, file_path, vocab, transform=None):
        self._root_dirc = root_dirc
        self._target_df = pd.read_csv(file_path)
        self._vocab = vocab
        self._transform = transform
        
    def __len__(self):
        return len(self._target_df)
    
    def __getitem__(self, idx):
        img_path = self._target_df["path"][idx]
        
        img_path = os.path.join(self._root_dirc, img_path)
        image= Image.open(img_path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        caption = self._target_df["caption"][idx]

        # Convert caption (string) to word ids.
        caption_lst = []
        caption_lst.append(self._vocab('<start>'))
        caption_lst.extend([self._vocab(cap) for cap in caption])
        caption_lst.append(self._vocab('<end>'))
        target = torch.Tensor(caption_lst)
        return image, target


def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions = zip(*data)
    images = torch.stack(images, dim=0)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(tuple(images), 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]  
    return torch.Tensor(images), targets, torch.Tensor(lengths)

    
def get_image_loader(root_dirc, file_path, vocab, transform, batch_size, shuffle, num_workers):
    dataset = ImageDataset(root_dirc, file_path, vocab, transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader