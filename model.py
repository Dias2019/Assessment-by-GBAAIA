import torch
import torch.nn as nn


class First_Model(nn.Module):

    def __init__(self):
        super(First_Model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels =1, out_channels =6, kernel_size=3, stride=1, padding='same', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=6),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels =6, out_channels =16, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels =16, out_channels =128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

        self.dropout = nn.Dropout()    # by default PyTorch's dropout ratio is 0.5
        
        self.fc0 = nn.Linear(in_features=2048, out_features=256)
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=7)

        self.softmax = nn.Softmax(dim=0)
    
    
    def forward(self, input):
        x = input
        
        # feature extraction layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = torch.flatten(x)

        # classification layers
        x = self.dropout(x)
        x = self.fc0(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.softmax(x)

        return output


class Second_Model(nn.Module):

    def __init__(self):
        super(Second_Model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels =1, out_channels =32, kernel_size=3, stride=1, padding='same', bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels =32, out_channels =64, kernel_size=3, stride=1, padding='same', bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=64))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels =64, out_channels =96, kernel_size=3, stride=1, padding='same', bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels =96, out_channels =128, kernel_size=3, stride=1, padding='same', bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=128))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels =128, out_channels =192, kernel_size=3, stride=1, padding='same', bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels =192, out_channels =256, kernel_size=3, stride=1, padding='same', bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=256))

        self.dropout = nn.Dropout()    # by default PyTorch's dropout ratio is 0.5
        
        self.fc0 = nn.Linear(in_features=9216, out_features=256)
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=7)

        self.softmax = nn.Softmax(dim=0)

    def forward(self, input):
        
        x = input
        
        # feature extraction layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = torch.flatten(x)

        # classification layers
        x = self.dropout(x)
        x = self.fc0(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.softmax(x)

        return output




model = Second_Model()

# we defined weight_decay parameter to include L2 regularization into optimizer
# because paper mentioned addition of regularization into the Second Model
optimizer_second_model = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)