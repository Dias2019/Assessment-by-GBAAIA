import torch
import torch.nn as nn



def conv3x3(in_planes, out_planes, padding, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=padding, bias=False)


class First_Model(nn.Module):

    def __init__(self):
        super(First_Model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_planes=1, out_planes=6, kernel_size=3, stride=1, padding='same', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=6),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_planes=6, out_planes=16, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_planes=16, out_planes=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

        self.dropout = nn.Dropout()    # by default PyTorch's dropout ratio is 0.5
        
        self.fc1 = nn.Linear(in_features=4 * 4 * 128, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)

        self.softmax = nn.Softmax(dim=0)
    
    
    def forward(self, input):
        x = input
        
        # feature extraction layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = torch.flatten(x)

        # classification layer
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.softmax(x)

        return output


class Second_Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = conv3x3() 

    def forward():

        return 0