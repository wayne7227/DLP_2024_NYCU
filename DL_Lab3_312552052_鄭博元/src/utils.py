import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def dice_score(pred_mask, gt_mask):
    assert pred_mask.size() == gt_mask.size()
    batch_size = pred_mask.size(0)
    pred_flat = pred_mask.view(batch_size, -1)
    gt_flat = gt_mask.view(batch_size, -1)
    pred_flat = torch.round(pred_flat)
    intersection = torch.sum((pred_flat == gt_flat), dim=1)
    dice = torch.mean(2 * intersection / (pred_flat.size(-1) + gt_flat.size(-1))).item()
    return dice

def plot_metric(metric_train_unet, metric_valid_unet, metric_train_resnet, metric_valid_resnet, graph_title, ylabel):
    epochs = range(1, len(metric_train_unet) + 1)
    plt.figure()
    plt.title(graph_title)
    plt.plot(epochs, metric_train_unet, label='UNet train')
    plt.plot(epochs, metric_valid_unet, label='UNet valid')
    plt.plot(epochs, metric_train_resnet, label='ResNet34-UNet train')
    plt.plot(epochs, metric_valid_resnet, label='ResNet34-UNet valid')
    
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.show()


def plot_loss(train_loss_unet, valid_loss_unet, train_loss_resnet, valid_loss_resnet, graph_title):

    epochs = range(1, len(train_loss_unet) + 1)
    plt.figure()
    plt.title(graph_title)
    plt.plot(epochs, train_loss_unet, label='UNet train loss')
    plt.plot(epochs, valid_loss_unet, label='UNet valid loss')
    plt.plot(epochs, train_loss_resnet, label='ResNet34-UNet train loss')
    plt.plot(epochs, valid_loss_resnet, label='ResNet34-UNet valid loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def visualize_predictions(model, dataset, display_plot=True):
    model.eval()
    with torch.no_grad():
        sample_idx = torch.randint(len(dataset), (1,)).item()
        sample = dataset[sample_idx]
        image = sample['image'].cuda()
        ground_truth = sample['mask']

        # Unnormalize the image
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        unnormalized_image = (image * std + mean).cpu().numpy().transpose(1, 2, 0)

        # Predict the mask
        predicted_mask = model(image.unsqueeze(0))
        predicted_mask = torch.round(predicted_mask)
        dice_value = dice_score(predicted_mask, ground_truth.unsqueeze(0).cuda())
        
        # Plot the results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(unnormalized_image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(predicted_mask.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        plt.text(0.5, -0.1, f'Dice Score: {dice_value:.4f}', transform=plt.gca().transAxes, ha='center')
        
        plt.subplot(1, 3, 3)
        plt.imshow(ground_truth.numpy().transpose(1, 2, 0), cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')
        
        if display_plot:
            plt.show()

if __name__ == '__main__':

    unet_train_scores = np.load('../record/unet_train_score.npy')
    unet_valid_scores = np.load('../record/unet_valid_score.npy')
    unet_train_loss = np.load('../record/unet_train_loss.npy')
    unet_valid_loss = np.load('../record/unet_valid_loss.npy')

    
    resnet34_unet_train_scores = np.load('../record/resnet34_unet_train_score.npy')
    resnet34_unet_valid_scores = np.load('../record/resnet34_unet_valid_score.npy')
    resnet34_unet_train_loss = np.load('../record/resnet34_unet_train_loss.npy')
    resnet34_unet_valid_loss = np.load('../record/resnet34_unet_valid_loss.npy')
    
    plot_metric(unet_train_scores, unet_valid_scores, resnet34_unet_train_scores, resnet34_unet_valid_scores, 'Training and Validation Score', 'Score')
    plot_loss(unet_train_loss ,unet_valid_loss,resnet34_unet_train_loss,resnet34_unet_valid_loss,'Training and Validation Loss')


