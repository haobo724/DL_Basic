import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
class NN (nn.Module):
    def __init__(self,input_size,class_NUM):
        super(NN, self).__init__()
        # self.fc1=nn.Linear(input_size,10)
        # self.fc2=nn.Linear(10,class_NUM)
        self.block = nn.Sequential(
            nn.Linear(input_size, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200,class_NUM),
        )

    def forward(self, x):
        # x=F.relu(self.fc1(x))
        # x=self.fc2(x)
        return self.block(x)

# model=NN(1000,5)
# x=torch.randn(22,1000)#1000 是一维的
# print(x.shape)
# print(model(x).shape)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate=1e-3
input_size=784
class_NUM=10
epoch=2
batch_size=64
train_set=datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(train_set,batch_size,shuffle=True)
test_set=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(test_set,batch_size,shuffle=True)
model=NN(input_size,class_NUM).to(device)
loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

for singleepoch in range(epoch):
    for batch_idx, (img,label) in enumerate(train_loader):
        img=img.to(device)
        label=label.to(device)
        img=img.reshape(img.shape[0],-1)


        with torch.cuda.amp.autocast():
            predictions = model(img)
            loss = loss_function(predictions, label)
        # print("img",img.shape)
        # print("label",label.shape)
        # x=model(img)
        # loss=loss_function(x,label)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

def check_acc(loader,model):
    num_correct=0
    num_sample=0
    model.eval()
    with torch.no_grad():
        for (img, label) in loader:
            img = img.to(device)
            label = label.to(device)
            img = img.reshape(img.shape[0], -1)

            x = model(img)
            _,preds=x.max(1)

            num_correct+=(preds==label).sum()
            num_sample+=preds.size(0)
        print(f"acc{num_correct/num_sample}")
    model.train()
    return num_correct/num_sample

check_acc(test_loader,model)
check_acc(train_loader,model)