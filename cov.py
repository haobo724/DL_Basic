import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
class NN (nn.Module):
    def __init__(self,input_size,class_NUM,input_wh=None):
        super(NN, self).__init__()
        self.cov1=nn.Conv2d(1,10,3,1,1,bias=False)
        self.cov2=nn.Conv2d(10,22,3,1,1,bias=False)
        self.cov3=nn.Conv2d(22, class_NUM, 1, bias=False)
        a = 123

        self.cov_test=nn.Conv2d(1, 10, 5,stride=1,padding=2,bias=False)

        # self.fc2=nn.Linear(10,class_NUM)
        # self.block = nn.Sequential(
        #     nn.Linear(input_size, 20),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(20,class_NUM),
        # )
        self.block = nn.Sequential(
            nn.Conv2d(1,10,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, 3,1,1, bias=False),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20,40,5,bias=False),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(40,class_NUM,3,bias=False),
            nn.Conv2d(class_NUM, class_NUM, 1, bias=False),

        )


    def forward(self, x):
        x = self.cov_test(x)
        print(x.size())
        # x=F.relu(self.fc1(x))
        # x=self.fc2(x)
        # print('input:',x.size())
        # x=self.cov1(x)
        # print(x.size())
        # x=self.cov2(x)
        # print(x.size())
        # x = self.cov3(x)
        # x = self.block(x)
        # print(x.size())
        # x=x.view(x.size(0), -1)
        # print(x.size())
        return x
model=NN(1000,5)
x=torch.randn(9,1,28,28)#1000 是一维的
model.forward(x)
