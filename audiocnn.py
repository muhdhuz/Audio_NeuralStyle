import torch 
import torch.nn as nn


# CNN Model (3 conv layer)
    
###################################

class CNN(nn.Module):
    def __init__(self,l1channels,l2channels,l3channels,orientation,in_height,in_width,in_channels,n_classes,kernelsize):
        super(CNN, self).__init__()
        
        self.l1channels = l1channels
        self.l2channels = l2channels
        self.l3channels = l3channels
        self.orientation = orientation
        self.in_height = in_height
        self.in_width = in_width
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        if self.orientation == '2D':
            self.maxpool_kernel = 2
            self.conv_kernel = kernelsize
            self.padding_size = (kernelsize-1)//2 #for same padding
            self.downsampledheight = self.in_height//8
            self.downsampledwidth = self.in_width//8
        elif self.orientation == 'freq' or self.orientation == 'time':
            self.maxpool_kernel = (1,2)
            self.conv_kernel = (1,kernelsize)
            self.padding_size = (0,(kernelsize-1)//2)
            self.downsampledheight = 1
            self.downsampledwidth = self.in_width//8
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, self.l1channels, kernel_size=self.conv_kernel, padding=self.padding_size),
            nn.BatchNorm2d(self.l1channels),
            nn.ReLU(),
            nn.MaxPool2d(self.maxpool_kernel))
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.l1channels, self.l2channels, kernel_size=self.conv_kernel, padding=self.padding_size),
            nn.BatchNorm2d(self.l2channels),
            nn.ReLU(),
            nn.MaxPool2d(self.maxpool_kernel))
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.l2channels, self.l3channels, kernel_size=self.conv_kernel, padding=self.padding_size),
            nn.BatchNorm2d(self.l3channels),
            nn.ReLU(),
            nn.MaxPool2d(self.maxpool_kernel))
        self.fc = nn.Linear(self.downsampledheight*self.downsampledwidth*self.l3channels, self.n_classes)
        
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out3 = out3.view(out3.size(0), -1)
        output = self.fc(out3)
        return output