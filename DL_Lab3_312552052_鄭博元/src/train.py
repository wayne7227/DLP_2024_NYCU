import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torch.optim import RAdam
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from oxford_pet import load_dataset
from utils import dice_score
from evaluate import evaluate
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet

def get_device():
    """Get the device for training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print('Memory Usage:')
        print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
        print(f'Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')
    # Ensure the record directory exists
    os.makedirs('../record/pics', exist_ok=True)
    return device

def get_transforms():
    """Get the data transforms for training, validation, and testing."""
    mean = [0.48, 0.45, 0.4]
    std = [0.23, 0.235, 0.228]

    train_transforms = {
        'image': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'mask': transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    }

    common_transforms = {
        'image': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'mask': transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
    }

    return train_transforms, common_transforms

def save_metrics(metrics, filename):
    """Save the training/validation metrics."""
    np.save(filename, np.array(metrics))

def load_model(args):
    """Load the appropriate model based on the arguments."""
    if args.model == 'unet':
        model = UNet()
    else:
        model = ResNet34UNet()
    return model

def visualize_predictions(epoch, model, dataset, device, save_dir):
    """Visualize and save model predictions."""
    model.eval()
    with torch.no_grad():
        data = dataset[0]
        image = data['image'].to(device)
        gt = data['mask'].to(device)
        pred = model(image.unsqueeze(0))
        pred = torch.round(pred)
        mean = torch.tensor([0.48, 0.45, 0.4], device=device).view(3, 1, 1)
        std = torch.tensor([0.23, 0.235, 0.228], device=device).view(3, 1, 1)

        unnormalized_image = (image * std + mean).cpu().numpy().transpose(1, 2, 0)

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(unnormalized_image)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(pred.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')  # Ensure correct shape
        plt.title('Prediction')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(gt.squeeze(0).cpu().numpy(), cmap='gray')  # Ensure correct shape
        plt.title('Ground Truth')
        plt.axis('off')

        # 确保保存路径正确
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{epoch}.png')
        print(f'Saving visualization at {save_path}')
        plt.savefig(save_path)
        plt.close()

def train(args):
    device = get_device()
    train_transforms, common_transforms = get_transforms()

    # Create dataset and dataloader
    train_dataset = load_dataset(args.data_path, mode='train', transform=train_transforms)
    valid_dataset = load_dataset(args.data_path, mode='valid', transform=common_transforms)
    test_dataset = load_dataset(args.data_path, mode='test', transform=common_transforms)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size)

    model = load_model(args).to(device)
    criterion = nn.BCELoss()
    optimizer = RAdam(model.parameters(), lr=args.learning_rate)

    # Metrics storage
    train_loss_rec, train_score_rec, valid_score_rec, valid_loss_rec = [], [], [], []
    best_score = 0  # This should start low enough to allow the first improvement

    # 保存模型的路径
    save_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../saved_models')
    os.makedirs(save_model_dir, exist_ok=True)

    print('Training')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_dice, total = 0, 0, 0

        for data in tqdm(train_loader, desc=f'Epoch {epoch}'):
            x = data['image'].to(device)
            y = data['mask'].to(device)

            optimizer.zero_grad()
            pred_mask = model(x)
            loss = criterion(pred_mask, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            total += x.size(0)
            train_dice += dice_score(pred_mask, y) * x.size(0)

        val_loss, val_dice = evaluate(model, valid_loader, criterion)

        if val_dice > best_score:
            best_score = val_dice
            save_path = os.path.join(save_model_dir, f"{args.model}.pth")
            print(f'Best valid score: {best_score}, model saved at {save_path}')
            torch.save(model.state_dict(), save_path)

        print(f'Training loss: {train_loss / total:.4f}    Training dice score: {train_dice / total:.4f}     Valid loss: {val_loss:.4f}     Valid dice score: {val_dice:.4f}')

        train_loss_rec.append(train_loss / total)
        train_score_rec.append(train_dice / total)
        valid_score_rec.append(val_dice)
        valid_loss_rec.append(val_loss)

        # 保存度量数据
        record_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../record')
        os.makedirs(record_dir, exist_ok=True)
        save_metrics(train_loss_rec, os.path.join(record_dir, f'{args.model}_train_loss.npy'))
        save_metrics(train_score_rec, os.path.join(record_dir, f'{args.model}_train_score.npy'))
        save_metrics(valid_score_rec, os.path.join(record_dir, f'{args.model}_valid_score.npy'))
        save_metrics(valid_loss_rec, os.path.join(record_dir, f'{args.model}_valid_loss.npy'))

        # 可视化预测结果
        visualize_predictions(epoch, model, test_dataset, device, os.path.join(record_dir, 'pics'))

        print('-' * 80)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model', '-m', type=str, choices=['unet', 'resnet34_unet'], required=True, help='Choose the model')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)
