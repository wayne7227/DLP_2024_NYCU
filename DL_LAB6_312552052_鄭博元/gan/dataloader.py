import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.utils as vutils
import os
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    with open('../train.json', "r") as f:
        data_json = json.load(f)

    with open("../objects.json", "r") as f:
        object_json = json.load(f)
    
    img_path_list, img_label_list = [], []

    for img_path, img_label in data_json.items():
        img_path_list.append(os.path.join("../iclevr", img_path))
        label = np.zeros(len(object_json),dtype=np.float32)
        for l in img_label:
            label[object_json[l]] = 1
        img_label_list.append(label)

    return img_path_list, img_label_list

class CLEVRLoader(torch.utils.data.Dataset):
    def __init__(self, transform_list):
        self.transform = transform_list
        self.img_path_list, self.label_list = get_data()

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.img_path_list[index]).convert("RGB"))
        label = self.label_list[index]

        return img, label

def dataloader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CLEVRLoader(transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    #print(len(dataloader.dataset))
    #print(dataset.__getitem__(0))

    return dataloader


    