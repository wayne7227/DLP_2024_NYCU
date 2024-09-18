import torch
from utils import *
from tqdm import tqdm

def evaluate(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss, val_dice, total_samples = 0, 0, 0
    
    with torch.no_grad():  # Disable gradient computation
        for data in tqdm(dataloader, desc='Valid'):
            inputs = data['image'].cuda()
            targets = data['mask'].cuda()
            outputs = model(inputs)  # Predict masks
            loss = criterion(outputs, targets)  # Compute loss
            val_loss += loss.item() * inputs.size(0)  # Accumulate weighted loss
            total_samples += inputs.size(0)  # Accumulate total number of samples
            val_dice += dice_score(outputs, targets) * inputs.size(0)  # Accumulate weighted Dice score

    avg_val_loss = val_loss / total_samples
    avg_val_dice = val_dice / total_samples
    return avg_val_loss, avg_val_dice  # Compute average loss and Dice score

def test(model, dataloader):

    model.eval()  # Set the model to evaluation mode
    test_dice, total_samples = 0, 0
    
    with torch.no_grad():  # Disable gradient computation
        for data in tqdm(dataloader, desc='Test'):
            inputs = data['image'].cuda()
            targets = data['mask'].cuda()
            outputs = model(inputs)  # Predict masks
            total_samples += inputs.size(0)  # Accumulate total number of samples
            test_dice += dice_score(outputs, targets) * inputs.size(0)  # Accumulate weighted Dice score

    avg_test_dice = test_dice / total_samples  # Compute average Dice score
    return avg_test_dice
