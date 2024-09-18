import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from PIL import Image
from torchvision import transforms


class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):
        assert mode in {"train", "valid", "test"}
        self.root = root
        self.mode = mode
        self.transform = transform
        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")
        self.filenames = self._read_split() # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")
        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)
        sample = dict(image=image, mask=mask, trimap=trimap)
        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)
        image = Image.fromarray(sample["image"])
        mask = Image.fromarray(sample["mask"])
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))
        seed = np.random.randint(2147483647)
        if self.transform is not None:
            torch.manual_seed(seed)
            sample['image'] = self.transform['image'](image)
            torch.manual_seed(seed)
            sample['mask'] = self.transform['mask'](mask)
        sample['mask'] = (sample['mask'] > 0).float()
        sample = dict(image=sample['image'], mask=sample['mask'])
        return sample

def load_dataset(data_path, mode, transform):
    dataset = SimpleOxfordPetDataset(root=data_path, mode=mode, transform=transform)
    return dataset

def create_split_files(data_path):
    images_directory = os.path.join(data_path, "images")
    annotations_directory = os.path.join(data_path, "annotations")

    all_files = [f.split('.')[0] for f in os.listdir(images_directory) if f.endswith('.jpg')]
    random.shuffle(all_files)

    trainval_split = int(0.9 * len(all_files))
    trainval_files = all_files[:trainval_split]
    test_files = all_files[trainval_split:]

    with open(os.path.join(annotations_directory, 'trainval.txt'), 'w') as f:
        for item in trainval_files:
            f.write(f"{item} 1\n")

    with open(os.path.join(annotations_directory, 'test.txt'), 'w') as f:
        for item in test_files:
            f.write(f"{item} 1\n")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision import transforms

    data_path = '../dataset/oxford-iiit-pet'

    if not os.path.exists(os.path.join(data_path, 'annotations', 'trainval.txt')) or not os.path.exists(os.path.join(data_path, 'annotations', 'test.txt')):
        create_split_files(data_path)
    
    # Define transformations for image and mask
    transform_image = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48, 0.45, 0.4], std=[0.23, 0.235, 0.228])
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Create a dictionary for transformations
    Tran = {'image': transform_image, 'mask': transform_mask}
    
    # Load the dataset
    dataset = load_dataset(data_path, 'train', Tran)
    
    # Get the first sample from the dataset and print the mask size
    sample = dataset[0]
    print(sample['mask'].size())
    
    # Display the mask using matplotlib
    mask_np = sample['mask'].detach().cpu().numpy().transpose(1, 2, 0)
    plt.imshow(mask_np, cmap='gray')
    plt.show()

