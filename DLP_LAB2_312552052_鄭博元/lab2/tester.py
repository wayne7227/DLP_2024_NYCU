import torch
from model.SCCNet import SCCNet
from Dataloader import get_data_loader

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corrects = (preds == labels).sum().item()
    return corrects / labels.size(0)

# 加載模型並設置為評估模式
model = SCCNet()
model.load_state_dict(torch.load("checkpoints/2000_finetune.pt"))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_data_loader = get_data_loader("test")
total_correct = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in test_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        B, Nc, time_samples = inputs.shape
        inputs = inputs.reshape(B, 1, Nc, time_samples)
        outputs = model(inputs)
        
        total_correct += accuracy(outputs, labels) * inputs.size(0)
        total_samples += inputs.size(0)

test_accuracy = total_correct / total_samples
print(f'Test Accuracy: {test_accuracy:.4f}')


