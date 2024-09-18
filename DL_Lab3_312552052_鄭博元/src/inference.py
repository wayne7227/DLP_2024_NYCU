import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import random  # 加入 random 模块
import matplotlib.pyplot as plt

from utils import dice_score, visualize_predictions
from oxford_pet import load_dataset
from evaluate import test
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet

def set_seed(seed):
    """设置随机种子，以确保每次运行时的结果一致"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', required=True, help='Path to the stored model weights')
    parser.add_argument('--data_path', '-d', type=str, required=True, help='Path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--print', type=int, default=1, help='Print the result')
    parser.add_argument('--plot', type=int, default=1, help='Plot example')
    parser.add_argument('--model_type', type=str, choices=['unet', 'resnet34_unet'], required=True, help='Model type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')  # 加入 seed 参数

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # 设置随机种子
    set_seed(args.seed)

    test_transforms = {
        'image': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'mask': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    }

    test_dataset = load_dataset(args.data_path, mode='test', transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Instantiate the model architecture
    if args.model_type == 'unet':
        model = UNet()
    else:
        model = ResNet34UNet()

    # Load the model state dictionary
    model.load_state_dict(torch.load(args.model))
    model = model.cuda()

    if args.print:
        # Evaluate the test dataset
        test_score = test(model, test_loader)
        print(f'Experiment results: Dice score {test_score:.4f}')

    if args.plot:
        # Plot random example
        visualize_predictions(model, test_dataset, display_plot=True)
