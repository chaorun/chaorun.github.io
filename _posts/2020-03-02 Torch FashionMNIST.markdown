---
layout: post
title:  "Torch: Fashion MNIST 多种方案"
date:   2020-03-02 13:34:47 +0800
categories: jekyll update
---


# Fashion MNIST 总结
## 先说一下，我只到了0.94，还是基于Baseline2做的
## 首先 我尝试了魔改了一下LeNet


```python
import os,d2l,sys,torch,math
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import time
from torch.optim import lr_scheduler
import torch.optim as optim

import os
import sys
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
#net
class Flatten(torch.nn.Module):  #展平操作
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape(torch.nn.Module): #将图像大小重定型
    def forward(self, x):
        return x.view(-1,1,28,28)      #(B x C x H x W)
# This function has been saved in the d2l package for future use
#use GPU
def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
    return device

device = try_gpu()
print(device)
```

    cpu


# 计算准确率

```
(1). net.train()
  启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为True
(2). net.eval()
不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False
```


```python
def evaluate_accuracy(data_iter, net,device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    acc_sum,n = torch.tensor([0],dtype=torch.float32,device=device),0
    for X,y in data_iter:
        # If device is the GPU, copy the data to the GPU.
        X,y = X.to(device),y.to(device)
        net.eval()
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))  #[[0.2 ,0.4 ,0.5 ,0.6 ,0.8] ,[ 0.1,0.2 ,0.4 ,0.3 ,0.1]] => [ 4 , 2 ]
            n += y.shape[0]
    return acc_sum.item()/n
#训练函数
def train_ch5(net, train_iter, test_iter,criterion, num_epochs, batch_size, device,lr=None):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_l_sum = torch.tensor([0.0],dtype=torch.float32,device=device)
        train_acc_sum = torch.tensor([0.0],dtype=torch.float32,device=device)
        n, start = 0, time.time()
        for X, y in train_iter:
            net.train()
            
            optimizer.zero_grad()
            X,y = X.to(device),y.to(device) 
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                y = y.long()
                train_l_sum += loss.float()
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == y))).float()
                n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net,device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc,
                 time.time() - start))
```

### 魔改1


```python
net = torch.nn.Sequential(     #Lelet                                                  
    Reshape(), 
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), #b*1*28*28  =>b*6*28*28
    nn.ReLU(),      
    nn.BatchNorm2d(num_features=6),
    nn.AvgPool2d(kernel_size=2, stride=2),                              #b*6*28*28  =>b*6*14*14
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),           #b*6*14*14  =>b*16*10*10
    nn.ReLU(),
    nn.BatchNorm2d(num_features=16),
    nn.AvgPool2d(kernel_size=2, stride=2),                              #b*16*10*10  => b*16*5*5
    Flatten(),                                                          #b*16*5*5   => b*400
    nn.Linear(in_features=16*5*5, out_features=120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
```

### 魔改2


```python
net = torch.nn.Sequential(     #Lelet                                                  
    Reshape(), 
    
    
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),  
    nn.ReLU(),      
    nn.BatchNorm2d(num_features=6),
    nn.AvgPool2d(kernel_size=2, stride=2),                               
    
    
    nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5),            
    nn.ReLU(),
    nn.BatchNorm2d(num_features=8),
    nn.AvgPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5,padding=2),            
    nn.Sigmoid(),
    nn.BatchNorm2d(num_features=16),
    nn.AvgPool2d(kernel_size=2, stride=1),
    
    Flatten(),                                                          
    nn.Linear(in_features=256, out_features=120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
```

### 魔改3


```python
net = torch.nn.Sequential(     #Lelet                                                  
    Reshape(), 
    
    
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),  
    nn.ReLU(),      
    nn.BatchNorm2d(num_features=16),
    nn.AvgPool2d(kernel_size=2, stride=2),                               
    
    
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),            
    nn.ReLU(),
    nn.BatchNorm2d(num_features=32),
    nn.AvgPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5,padding=2),            
    nn.ReLU(),
    nn.BatchNorm2d(num_features=16),
    nn.AvgPool2d(kernel_size=2, stride=1),
    
    Flatten(),                                                          
    nn.Linear(in_features=256, out_features=120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
```

### 魔改4


```python
net = torch.nn.Sequential(     #Lelet                                                  
    Reshape(), 
    
    
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),  
    nn.ReLU(),      
    nn.BatchNorm2d(num_features=32),
    nn.AvgPool2d(kernel_size=2, stride=2),                               
    
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),            
    nn.ReLU(),
    nn.BatchNorm2d(num_features=64),
    nn.AvgPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),            
    nn.ReLU(),
    nn.BatchNorm2d(num_features=32),
    nn.AvgPool2d(kernel_size=2, stride=2),
    
    Flatten(),                                                          
    nn.Linear(in_features=32*2*2, out_features=120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
print(net)
net = net.to(device)
```

    Sequential(
      (0): Reshape()
      (1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (2): ReLU()
      (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (4): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (5): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (6): ReLU()
      (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (9): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
      (10): ReLU()
      (11): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (12): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (13): Flatten()
      (14): Linear(in_features=128, out_features=120, bias=True)
      (15): Sigmoid()
      (16): Linear(in_features=120, out_features=84, bias=True)
      (17): Sigmoid()
      (18): Linear(in_features=84, out_features=10, bias=True)
    )


### 后来还是用了现成的baseline


```python

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # TODO:
class GlobalAvgPool2d(nn.Module):
    """
    全局平均池化层
    可通过将普通的平均池化的窗口形状设置成输入的高和宽实现
    """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
class Residual(nn.Module): 
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        """
            use_1×1conv: 是否使用额外的1x1卷积层来修改通道数
            stride: 卷积层的步幅, resnet使用步长为2的卷积来替代pooling的作用，是个很赞的idea
        """
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    '''
    resnet block

    num_residuals: 当前block包含多少个残差块
    first_block: 是否为第一个block

    一个resnet block由num_residuals个残差块组成
    其中第一个残差块起到了通道数的转换和pooling的作用
    后面的若干残差块就是完成正常的特征提取
    '''
    if first_block:
        assert in_channels == out_channels # 第一个模块的输出通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)
```


```python
# 定义resnet模型结构
net = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),   # TODO: 缩小感受野, 缩channel
        nn.BatchNorm2d(32),
        nn.ReLU())
        #nn.ReLU(),
        #nn.MaxPool2d(kernel_size=2, stride=2))   # TODO：去掉maxpool缩小感受野

# 然后是连续4个block
net.add_module("resnet_block1", resnet_block(32, 32, 2, first_block=True))   # TODO: channel统一减半
net.add_module("resnet_block2", resnet_block(32, 64, 2))
net.add_module("resnet_block3", resnet_block(64, 128, 2))
net.add_module("resnet_block4", resnet_block(128, 256, 2))
# global average pooling
net.add_module("global_avg_pool", GlobalAvgPool2d()) 
# fc layer
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(256, 10)))
print('Print the CNN Structure')
print(net)
```

    Print the CNN Structure
    Sequential(
      (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (resnet_block1): Sequential(
        (0): Residual(
          (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): Residual(
          (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (resnet_block2): Sequential(
        (0): Residual(
          (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv3): Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2))
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): Residual(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (resnet_block3): Sequential(
        (0): Residual(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv3): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): Residual(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (resnet_block4): Sequential(
        (0): Residual(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2))
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): Residual(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (global_avg_pool): GlobalAvgPool2d()
      (fc): Sequential(
        (0): FlattenLayer()
        (1): Linear(in_features=256, out_features=10, bias=True)
      )
    )


### 尝试把Sequential里面的ReLU换成Tanh(),Sigmoid()
#### 也尝试修改过后面的全连接层


```python
net = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),   
        nn.BatchNorm2d(32),
        #nn.ReLU()
        nn.Tanh()
        #nn.LeakyReLU(0.2)
        #nn.Sigmoid()
        )
   
net.add_module("resnet_block1", resnet_block(32, 32, 2, first_block=True))    
net.add_module("resnet_block2", resnet_block(32, 64, 2))
net.add_module("resnet_block3", resnet_block(64, 128, 2))
net.add_module("resnet_block4", resnet_block(128, 256, 2))
# global average pooling
net.add_module("global_avg_pool", GlobalAvgPool2d()) 
# fc layer
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(256, 10)))
#net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(256, 128),nn.Sigmoid(),nn.Linear(128,10)))

print(net)
```

    Sequential(
      (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): Tanh()
      (resnet_block1): Sequential(
        (0): Residual(
          (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): Residual(
          (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (resnet_block2): Sequential(
        (0): Residual(
          (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv3): Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2))
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): Residual(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (resnet_block3): Sequential(
        (0): Residual(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv3): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): Residual(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (resnet_block4): Sequential(
        (0): Residual(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2))
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): Residual(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (global_avg_pool): GlobalAvgPool2d()
      (fc): Sequential(
        (0): FlattenLayer()
        (1): Linear(in_features=256, out_features=10, bias=True)
      )
    )



```python
### 加载数据这里还是用的Baseline2的方法
```


```python
def load_data_fashion_mnist(batch_size, root='../../dataset', use_normalize=False, mean=None, std=None):
    """Download the fashion mnist dataset and then load into memory."""

    if use_normalize:
        normalize = transforms.Normalize(mean=[mean], std=[std])
        train_augs = transforms.Compose([transforms.RandomCrop(28, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), 
                    normalize])
        test_augs = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        train_augs = transforms.Compose([transforms.ToTensor()])
        test_augs = transforms.Compose([transforms.ToTensor()])
    
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=train_augs)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=test_augs)
    if sys.platform.startswith('win'):
        num_workers = 0   
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


batch_size = 128  
train_iter, test_iter = load_data_fashion_mnist(batch_size, root='~/FashionMNIST', use_normalize=False)

temp_sum = 0
cnt = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
        break   
    channel_mean = torch.mean(X, dim=(0,2,3))  
    cnt += 1   
    temp_sum += channel_mean[0].item()
dataset_global_mean = temp_sum / cnt

cnt = 0
temp_sum = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
        break   
    residual = (X - dataset_global_mean) ** 2
    channel_var_mean = torch.mean(residual, dim=(0,2,3))  
    cnt += 1    
    temp_sum += math.sqrt(channel_var_mean[0].item())
dataset_global_std = temp_sum / cnt

batch_size =128  
train_iter, test_iter = load_data_fashion_mnist(batch_size, root='~/FashionMINST', use_normalize=True,
                        mean = dataset_global_mean, std = dataset_global_std)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        
        device = list(net.parameters())[0].device
    net.eval() 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    net.train() 
    return acc_sum / n


def train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_test_acc = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.4f, test acc %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        if test_acc > best_test_acc:
            print('find best! save at model/best.pth')
            best_test_acc = test_acc
            torch.save(net.state_dict(), 'model/best.pth')
```

### 尝试降低学习率，使用RMSProp
#### 但是RMSProp下降的太慢了


```python
lr, num_epochs = 0.01, 50
lr=0.001
optimizer = optim.Adam(net.parameters(), lr=lr)
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)   # TODO:
#optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.9)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

#net.load_state_dict(torch.load('model/best.pth'))
net = net.to(device)

net.eval() 
id = 0
preds_list = []
with torch.no_grad():
    for X, y in test_iter:
        batch_pred = list(net(X.to(device)).argmax(dim=1).cpu().numpy())
        for y_pred in batch_pred:
            preds_list.append((id, y_pred))
            id += 1

with open('submission.csv', 'w') as f:
    f.write('ID,Prediction\n')
    for id, pred in preds_list:
        f.write('{},{}\n'.format(id, pred))
```

### 想要试试其他的方案，比如说DCGAN, 当然也在网上看到了DenseNet的模型


```python

```


```python

```
