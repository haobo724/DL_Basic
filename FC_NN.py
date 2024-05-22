# MINST数据集是0-9的手写数字，所以有十个类，图片大小是28*28也就是784
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from codecarbon import EmissionsTracker


class NN(nn.Module):
    def __init__(self, input_size, class_NUM):
        super(NN, self).__init__()
        # self.fc1=nn.Linear(input_size,20)
        # self.fc2=nn.Linear(20,class_NUM)
        self.block = nn.Sequential(
            nn.Linear(input_size, 20,bias=False),
            nn.Linear(20, class_NUM,bias=False),
        )

    # 注意最后要把我们的输出reshape一下，可以自己打印出来shape看看
    def forward(self, x):
        x = self.block(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        return x


class FC(nn.Module):
    def __init__(self, input_size, class_NUM):
        super(FC, self).__init__()
        # self.fc1=nn.Linear(input_size,20)
        # self.fc2=nn.Linear(20,class_NUM)

        # 一上来就是一个和输入图像等大的卷积核，把图像卷成【N,1,1,20】的大小，然后在用一个1*1的卷积核替代全连接，到输出层，大小为【B，CLASS】
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(input_size, input_size), bias=False),
            # nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, class_NUM, 1, bias=False)
        )

    # 注意最后要把我们的输出reshape一下，可以自己打印出来shape看看
    def forward(self, x):
        x = self.block(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 1e-3
input_size = 784
class_NUM = 10
epoch = 2
batch_size = 64
train_set = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_set = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)
model = NN(input_size, class_NUM).to(device)
model_FC = FC(28, class_NUM).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()
# 代码注释部分并非错误而是比基础还基础的写法，我改用稍复杂、更实用的结构来替换
with EmissionsTracker() as tracker:
    used_model = model
    print("parameter:", sum(p.numel() for p in used_model.parameters() if p.requires_grad))
    for singleepoch in range(epoch):
        for batch_idx, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)
            if used_model._get_name() == 'NN':
                img = img.reshape(img.shape[0], -1)

            with torch.cuda.amp.autocast():
                predictions = used_model(img)
                loss = loss_function(predictions, label)
            # print("img",img.shape)
            # print("label",label.shape)
            # x=model(img)
            # loss=loss_function(x,label)
            # optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# 记得加上eval（）和train（），因为网络在这两种模式下可以决定有一些层要不要使用，比如dropout等，正确开关可以提高正确率
def check_acc(loader, model):
    num_correct = 0
    num_sample = 0
    model.eval()
    with torch.no_grad():
        for (img, label) in loader:
            img = img.to(device)
            label = label.to(device)
            # 同理如果使用全卷积层，下面这行就要注释掉

            img = img.reshape(img.shape[0], -1)
            # x就是数据过model的结果，根据我们的网络架构我们知道一共有十个class（列）【0...9】,数据（行）有多少个呢，根据loader决定
            x = model(img)
            # 很有启发的写法，直接一步得出每一行最大值所属坐标，preds将会是batchsize长的list，里面记录着每一行最大值坐标
            _, preds = x.max(1)
            # print("x:", x.shape)
            # print("preds:", preds.shape)
            # 分类对为1，然后累加
            num_correct += (preds == label).sum()
            num_sample += preds.size(0)
        print(f"acc{num_correct / num_sample}")
    model.train()
    return num_correct / num_sample


check_acc(test_loader, model)
check_acc(train_loader, model)
