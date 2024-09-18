import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import json
import numpy as np
import torchvision.transforms as T


def save_images(images, path, nrow):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    torchvision.utils.save_image(grid)


class iclevr(torch.utils.data.Dataset):
    def __init__(self, root, jfile, object, transform):
        self.root = root

        self.jfile = jfile
        with open(self.jfile, 'r') as f:
            self.raw_data = json.load(f)
        self.data = []
        for i in self.raw_data:
            self.data.append([i, self.raw_data[i]])

        self.object = object
        with open(self.object, 'r') as f:
            self.mapping = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]

        img_path = os.path.join(self.root, data[0])
        img = self.transform(Image.open(img_path).convert('RGB'))

        raw_label = data[1]
        label = multilabel([self.mapping[i] for i in raw_label])

        return img, label
        
def getDataloader(dataset, batch):
    return DataLoader(dataset=dataset, batch_size=batch, shuffle=True)
    
def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def multilabel(label):
    output = torch.zeros(24)
    for i in label:
        output[i] = 1
    return output

def get_test_label(test_json, object):
    with open(test_json, 'r') as f:
        data = json.load(f)
    with open(object, 'r') as f:
        mapping = json.load(f)
    output = torch.tensor([])
    for i in range(len(data)):
        raw = data[i]
        output = torch.cat((output, multilabel([mapping[l] for l in raw]).view(1, -1)), dim=0)
    return output


if __name__ == '__main__':
    a = torch.randn(5,3,64,64)
    a = a*255
    save_images(a,'./a.png',3)
