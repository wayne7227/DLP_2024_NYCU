import torch
import torch.optim as optim
from Dataloader import get_data_loader
from model.SCCNet import SCCNet

# Accuracy function
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corrects = (preds == labels).sum().item()
    return corrects / labels.size(0)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):  # epoch:0->num_epochs   [0, 1, ... , num_epochs-1]
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            B, Nc, time_samples = inputs.shape
            inputs = inputs.reshape(B, 1, Nc, time_samples)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += accuracy(outputs, labels) * inputs.size(0)
            total_samples += inputs.size(0)
    
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / total_samples
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')
        
        # Every 50 epochs, evaluate on validation set
        if (epoch + 1) % 50 == 0:
            model.eval()
            val_running_corrects = 0
            val_total_samples = 0

            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    B, Nc, time_samples = val_inputs.shape
                    val_inputs = val_inputs.reshape(B, 1, Nc, time_samples)

                    val_outputs = model(val_inputs)
                    val_running_corrects += accuracy(val_outputs, val_labels) * val_inputs.size(0)
                    val_total_samples += val_inputs.size(0)

            val_acc = val_running_corrects / val_total_samples
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_acc:.4f}')
    
    torch.save(model.state_dict(), "./checkpoints/2000_LOSO.pt")

# Implement your training script here
learning_rate = 0.001
epochs = 2000
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = SCCNet(numClasses=4, Nu=22, Nc=22, Nt=1, dropoutRate=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_loader = get_data_loader('train', batch_size=batch_size)
val_loader = get_data_loader('test', batch_size=batch_size)
criterion = torch.nn.CrossEntropyLoss()

train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
