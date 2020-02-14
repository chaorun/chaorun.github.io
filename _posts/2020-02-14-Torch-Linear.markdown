---
layout: post
title:  "Torch Linear"
date:   2020-02-14 19:34:47 +0800
categories: jekyll update
---

```python
#线性回归
```


```python
%matplotlib nbagg
from __future__ import print_function, division
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
```


```python
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
```


```python

```


```python
print(features[0], labels[0])
```

    tensor([-0.2368,  1.4790]) tensor(-1.3003)



```python
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

# # 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# import sys
# sys.path.append("..")
# from d2lzh_pytorch import * 

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);
plt.show()
```

```python
# 本函数已保存在d2lzh包中方便以后使用
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)
def linreg(X, w, b):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    return torch.mm(X, w) + b
#Performs a matrix multiplication of the matrices


def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data
```


```python
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break
```

    tensor([[-0.3513,  0.7561],
            [ 0.3961, -0.0920],
            [-1.2184,  0.1623],
            [ 1.4052,  1.6383],
            [ 0.8815, -1.6394],
            [-0.3908,  0.8612],
            [ 2.0732, -0.4766],
            [ 0.6000, -0.3341],
            [ 0.6602, -0.7155],
            [-0.4657, -0.1234]]) tensor([ 0.9471,  5.3013,  1.2147,  1.4539, 11.5369,  0.4846,  9.9688,  6.5471,
             7.9466,  3.6844])



```python
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
```


```python
w
```




    tensor([[-0.0074],
            [ 0.0129]])




```python
w
```




    tensor([[-0.0074],
            [ 0.0129]])



#之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此我们要让它们的requires_grad=True。


```python
w.requires_grad_(requires_grad=True)
```




    tensor([[-0.0074],
            [ 0.0129]], requires_grad=True)




```python
w
```




    tensor([[-0.0074],
            [ 0.0129]], requires_grad=True)




```python
b.requires_grad_(requires_grad=True) 
```




    tensor([0.], requires_grad=True)




```python
b
```




    tensor([0.], requires_grad=True)




```python

```


```python

```


```python

```


```python
A=(y_hat.view(-1) - y) ** 2 / 2
```


```python
y_hat.shape,y.shape
```




    (torch.Size([1000, 1]), torch.Size([1000]))




```python
(y_hat.view(-1) - y).shape,(y_hat - y.view(-1)).shape,((y_hat - y.view(y_hat.shape)) ** 2 / 2).shape
```




    (torch.Size([1000]), torch.Size([1000, 1000]), torch.Size([1000, 1]))




```python
((y_hat - y.view(-1, 1)) ** 2 / 2).shape
```




    torch.Size([1000, 1])




```python
((y_hat - y.view(-1)) ** 2 / 2 ).shape
```




    torch.Size([1000, 1000])




```python
(y_hat-y.view(-1)).shape
```




    torch.Size([1000, 1000])




```python
def squared_loss(y_hat, y): 
    return (y_hat - y.view(y_hat.size())) ** 2 / 2  #torch.Size([1000])
def squared_lossA(y_hat, y): 
    return (y_hat.view(-1) - y) ** 2 / 2    #torch.Size([1000])
def squared_lossB(y_hat, y): 
    return (y_hat - y.view(-1)) ** 2 / 2    #torch.Size([1000,1000])
def squared_lossC(y_hat, y): 
    return (y_hat - y.view(y_hat.shape)) ** 2 / 2   #torch.Size([1000])
def squared_lossD(y_hat, y): 
    return (y_hat - y.view(-1, 1)) ** 2 / 2    #torch.Size([1000, 1])
```


```python
y_hat, y=net(features, w, b), labels
#y_hat.view(-1).size() , y.size()
```


```python
y_hat.size(), y.view(-1).size()
```




    (torch.Size([1000, 1]), torch.Size([1000]))




```python
(y_hat-y.view(-1,-1)).shape
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-264-c6e0233eadd2> in <module>()
    ----> 1 (y_hat-y.view(-1,-1)).shape
    

    RuntimeError: only one dimension can be inferred



```python
y.view(-1,1).shape,y_hat.shape
```




    (torch.Size([1000, 1]), torch.Size([1000, 1]))




```python
2.33	3.14
1.07	0.98
1.23	1.32
```


```python
yhat=torch.tensor([2.33,1.07,1.23])
y=torch.tensor([3.14,0.98,1.32])
def squared_loss(y_hat, y): 
    return (y_hat - y.view(y_hat.size())) ** 2 / 2
print(squared_loss(yhat, y))
print(torch.mean(squared_loss(yhat, y)))
```

    tensor([0.3281, 0.0041, 0.0041])
    tensor(0.1121)



```python



```




    tensor([0.3281, 0.0041, 0.0041])




```python

```




    tensor(0.1121)




```python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True) 
for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
        
        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    #print('w',w[0].item(),w[1].item(),'b',b.item())
    train_l = squared_lossA(net(features, w, b), labels)
    print('w',w[0].item(),w[1].item(),'b',b.item())
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
```

    w 1.9516918659210205 -3.261199712753296 b 3.9988179206848145
    epoch 1, loss 0.031503
    w 1.9997085332870483 -3.3933534622192383 b 4.189835548400879
    epoch 2, loss 0.000116
    w 2.000166654586792 -3.398942232131958 b 4.199468612670898
    epoch 3, loss 0.000048



```python

```


```python
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
```


```python
import torch.utils.data as Data

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
```


```python
for X, y in data_iter:
    print(X, y)
    break
```

    tensor([[ 0.4979, -0.1159],
            [ 0.3042, -0.9498],
            [ 0.3829, -0.4768],
            [ 0.4592,  0.3429],
            [ 0.4872, -1.2047],
            [-0.7133,  0.3192],
            [ 1.5930, -0.0048],
            [ 0.2526, -0.5475],
            [-1.7900, -0.8426],
            [-1.8417, -1.8305]]) tensor([5.5748, 8.0158, 6.5597, 3.9433, 9.2687, 1.7031, 7.4085, 6.5609, 3.4895,
            6.7489])


#首先，引入torch.nn模块。实际上，“ nn”是神经网络（神经网络）的缩写。顾名思义，该模块定义了串联神经网络的层。之前我们已经用过了autograd，而nn就是利用autograd来定义模型。nn的核心数据结构是Module，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。在实际使用中，最常见的做法是继承nn.Module，编写自己的网络/层。一个nn.Module实例应该包含一些层以及返回输出的前向传播（forward）方法。下面先来看看如何用nn.Module实现一个线性回归模型。


```python
from torch import nn
from torch.nn import init
```


```python

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
        #nn.Linear Applies a linear transformation to the incoming data
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net) # 使用print可以打印出网络的结构
```

    LinearNet(
      (linear): Linear(in_features=2, out_features=1, bias=True)
    )



```python
#写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1),
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])
```

    Sequential(
      (linear): Linear(in_features=2, out_features=1, bias=True)
    )
    Linear(in_features=2, out_features=1, bias=True)



```python
net = nn.Sequential(
    nn.Linear(num_inputs, 1),
   # nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )
```


```python
for param in net.parameters():
    print(param)
```

    Parameter containing:
    tensor([[-0.6952,  0.3753]], requires_grad=True)
    Parameter containing:
    tensor([-0.0610], requires_grad=True)



```python
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)
```




    Parameter containing:
    tensor([0.], requires_grad=True)




```python
loss = nn.MSELoss()
```


```python
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)
```

    SGD (
    Parameter Group 0
        dampening: 0
        lr: 0.03
        momentum: 0
        nesterov: False
        weight_decay: 0
    )



```python
# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
```


```python
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
```

    epoch 1, loss: 6.676648
    epoch 2, loss: 3.320895
    epoch 3, loss: 0.623434



```python
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
```

    [2, -3.4] Parameter containing:
    tensor([[ 1.7713, -2.8827]], requires_grad=True)
    4.2 Parameter containing:
    tensor([3.5038], requires_grad=True)



```python

```
