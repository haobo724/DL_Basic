import torch
import torch.nn as nn
import time,glob,cv2
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
class NN (nn.Module):
    def __init__(self,input_size,class_NUM,input_wh=None):
        super(NN, self).__init__()
        # self.fc1=nn.Linear(input_size,10)
        # self.fc2=nn.Linear(10,class_NUM)
        # self.block = nn.Sequential(
        #     nn.Linear(input_size, 20),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(20,class_NUM),
        # )
        self.block = nn.Sequential(
            nn.Conv2d(1,10,3,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(20,40,3,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d( (1, 1)),


            nn.Conv2d(40,class_NUM,1,bias=False),
            # nn.Conv2d(class_NUM, class_NUM, 1, bias=False),

        )


    def forward(self, x):
        # x=F.relu(self.fc1(x))
        # x=self.fc2(x)
        # print('input:',x.size())
        x=self.block(x)
        x=x.view(x.size(0), -1)
        # print(x.size())
        return x


# model=NN(1000,5)
# x=torch.randn(22,1000)#1000 是一维的
# print(x.shape)
# print(model(x).shape)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate=1e-3
input_size=784
class_NUM=10
epoch=3
batch_size=64
train_set=datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(train_set,batch_size,shuffle=True)
test_set=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(test_set,batch_size,shuffle=True)
model=NN(input_size,class_NUM,28).to(device)
total = sum([param.nelement() for param in model.parameters()])
print(total)
loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()
curtime=time.time()
for singleepoch in range(epoch):
    for batch_idx, (img,label) in enumerate(train_loader):
        img=img.to(device)
        label=label.to(device)
        # img=img.reshape(img.shape[0],-1)


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
print('time',time.time()-curtime)
def check_acc(loader,model):
    num_correct=0
    num_sample=0
    model.eval()
    with torch.no_grad():
        for (img, label) in loader:
            img = img.to(device)
            label = label.to(device)
            # img = img.reshape(img.shape[0], -1)
            x = model(img)
            _,preds=x.max(1)
            # print(label)
            # print(preds)
            num_correct+=(preds==label).sum()
            num_sample+=preds.size(0)
        print(f"acc{num_correct/num_sample}")
    model.train()
    return num_correct/num_sample
def test_img(model):
    img = glob.glob('./*.jpg')
    print(img)
    for i in img:
        i = cv2.imread(i,0)
        print(i.shape)
        i = cv2.resize(i,(28,28))
        i = torch.FloatTensor(i)
        i = torch.unsqueeze(i,dim=0)
        i = torch.unsqueeze(i,dim=0)
        i = i.to(device)
        print(i.size())
        x = model(i)
        print(x.size())
        _, preds = x.max(1)
        print(preds)

check_acc(test_loader,model)
test_img(model)
# check_acc(train_loader,model)