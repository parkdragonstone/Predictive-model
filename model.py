import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, in_dim, d_model, nhead, num_layers, dim_feedforward, dropout, out_dim):
        super(TransformerModel, self).__init__()
        self.in_dim = in_dim
        self.d_model_dim = d_model
        self.encoder = nn.Linear(in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout = dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model * 36, 100)
        self.decoder = nn.Linear(100, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.view(-1, 36, self.in_dim)
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.view(-1, 36 * self.d_model_dim)
        x = self.dropout(self.fc(x))
        x = self.decoder(x)
        
        return x
    
    

# ResNet 클래스 정의
class CNNNet(nn.Module):
    def __init__(self, block, num_blocks, start, feature1, dropout):
        super(CNNNet, self).__init__()
        self.in_planes = start
        self.relu = nn.ReLU()
        self.pooling = nn.AvgPool2d((9,9))
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(3, start, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(start)
        self.layer1 = self._make_layer(block, start, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, feature1, num_blocks[1], stride=2)
        self.linear = nn.Linear(feature1, int(feature1/2))
        self.fc = nn.Linear(int(feature1/2), 3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = [] 
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(self.relu(self.linear(out)))
        out = self.fc(out)
        return out

class Resnet_Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Resnet_Block, self).__init__()
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) 

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential() 
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) 
        out = self.relu(out)
        return out

# ResNet18 함수 정의
def ResNet10(start, feature1, dropout):
    return CNNNet(Resnet_Block, [1, 1, 1, 1], start, feature1, dropout)




class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(108, 128)
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, 3) 

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x) 
        return x
