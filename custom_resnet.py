import torch.nn as nn
import torch.nn.functional as F


NO_GROUPS = 4

'''

'''
class ResnetBlock(nn.Module):
    def __init__(self, input_channel, output_channel, padding=1, norm='bn', drop=0.01):

        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=padding)

        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(output_channel)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(NO_GROUPS, output_channel)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, output_channel)

        self.drop1 = nn.Dropout2d(drop)

        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=padding)

        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(output_channel)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(NO_GROUPS, output_channel)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, output_channel)

        self.drop2 = nn.Dropout2d(drop)


    '''
    Depending on the model requirement, Convolution block with number of layers is applied to the input image
    '''
    def __call__(self, x):

        x = self.conv1(x)
        x = self.n1(x)
        x = F.relu(x)

        x = self.drop1(x)


        #if layers >= 2:

        x = self.conv2(x)

        x = self.n2(x)
        x = F.relu(x)
        x = self.drop2(x)

        return x


class S10Model(nn.Module):
    def __init__(self, base_channels, norm='bn', drop=0.01):
        super(S10Model, self).__init__()

        self.base_channels = base_channels

        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Dropout2d(drop)
        )

        #layer1
        self.x1 = nn.Sequential(
            nn.Conv2d(base_channels, 2*base_channels, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(2*base_channels),
            nn.ReLU(),
            nn.Dropout2d(drop)
        )

        self.R1 = ResnetBlock(2*base_channels, 2*base_channels, padding=1, drop=drop)

        #layer2
        self.layer2 = nn.Sequential(
            nn.Conv2d(2*base_channels, 4*base_channels, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(4*base_channels),
            nn.ReLU(),
            nn.Dropout2d(drop)
        )

        #layer3
        self.x2 = nn.Sequential(
            nn.Conv2d(4*base_channels, 8*base_channels, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8*base_channels),
            nn.ReLU(),
            nn.Dropout2d(drop)
        )

        self.R2 = ResnetBlock(8*base_channels, 8*base_channels, padding=1, drop=drop)

        self.pool = nn.MaxPool2d(4)

        self.fc = nn.Linear(8*base_channels, 10)

    def forward(self, x):

        #print(x.size())

        x = self.prep_layer(x)
        #print(x.size())

        x = self.x1(x)
        #print('x1', x.size())

        x = self.R1(x) + x
        #print('x', x.size())

        x = self.layer2(x)
        #print(x.size())

        x = self.x2(x)
        #print('x2', x.size())

        x = self.R2(x) + x
        #print('x', x.size())

        x = self.pool(x)
        #print(x.size())

        x = x.view(x.size(0), 8*self.base_channels)
        #print(x.size())

        x = self.fc(x)
        #print(x.size())

        return F.log_softmax(x, dim=1)
