# implement SCCNet model

import torch
import torch.nn as nn

# reference paper: https://ieeexplore.ieee.org/document/8716937
class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x ** 2

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, Nu=22, Nc=22, Nt=1, dropoutRate=0.5):
        super(SCCNet, self).__init__()
        
        # First convolutional block: spatial filtering
        self.conv1 = nn.Conv2d(1, Nu, (Nc, Nt))
        self.batch_norm1 = nn.BatchNorm2d(Nu)
        
        # Second convolutional block: spatio-temporal filtering
        self.conv2 = nn.Conv2d(Nu, 20, (1, 12))
        self.batch_norm2 = nn.BatchNorm2d(20)
        
        # Average pooling layer
        self.classifier = nn.Linear(620, numClasses, bias=True)
        self.pool = nn.AvgPool2d((1, 62), stride=(1, 12))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropoutRate)
        
        # Square activation layer
        self.square_layer = SquareLayer()
    
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.batch_norm1(x)
        # Second convolutional block
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.square_layer(x)
        # Dropout
        x = self.dropout(x)
        # Average pooling
        x = self.pool(x)
        x = torch.log(x)
        
        # Flatten
        x = x.view(-1, 620)
        # print(x.shape)
        # Fully connected layer
        x = self.classifier(x)
        
        return x
    
    def get_size(self, C, N):
        return C * ((N - 12 + 2 * 5) // 12 + 1)