---
layout: post
title:  "Torch MultiLayerSensor"
date:   2020-02-14 13:34:47 +0800
categories: jekyll update
---

```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
sys.path.append("../Dive-into-DL-PyTorch/code/") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l
sys.path.append('/Users/chaorun/')
from SoTinShuiLib import *
```


```python
print(torch.__version__)
```

    1.4.0


### ReLU


```python
def xyplot(x_vals, y_vals, name):
    # d2l.set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
```


```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')
plt.show()
```


![png](output_4_0.png)



```python
print(y.sum().backward())
xyplot(x, x.grad, 'grad of relu')
plt.show()
```

    None






### sigmoid


```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.sigmoid()
xyplot(x, y, 'sigmoid')
plt.show()
```



```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.sigmoid()
y.sum().backward()
#x.grad.zero_()
xyplot(x, x.grad, 'grad of sigmoid')
plt.show()
```






```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.sigmoid()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')
plt.show()
```






```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.tanh()
xyplot(x, y, 'tanh')
plt.show()
```



```python

```


```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,root='~/Datasets/FashionMNIST')
```


```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)
```


```python
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))
```


```python
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    O = torch.matmul(H, W2) + b2
    return O
```


```python
loss = torch.nn.CrossEntropyLoss()
```


```python
num_epochs, lr = 5, 100.0
# def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
#               params=None, lr=None, optimizer=None):
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
#         for X, y in train_iter:
#             y_hat = net(X)
#             l = loss(y_hat, y).sum()
#             
#             # 梯度清零
#             if optimizer is not None:
#                 optimizer.zero_grad()
#             elif params is not None and params[0].grad is not None:
#                 for param in params:
#                     param.grad.data.zero_()
#            
#             l.backward()
#             if optimizer is None:
#                 d2l.sgd(params, lr, batch_size)
#             else:
#                 optimizer.step()  # “softmax回归的简洁实现”一节将用到
#             
#             
#             train_l_sum += l.item()
#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
#             n += y.shape[0]
#         test_acc = evaluate_accuracy(test_iter, net)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
#               % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
```

    epoch 1, loss 0.0030, train acc 0.713, test acc 0.785
    epoch 2, loss 0.0019, train acc 0.825, test acc 0.817
    epoch 3, loss 0.0017, train acc 0.846, test acc 0.820
    epoch 4, loss 0.0015, train acc 0.856, test acc 0.831
    epoch 5, loss 0.0014, train acc 0.865, test acc 0.838



```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)
loss = torch.nn.CrossEntropyLoss() 
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))
def tanh(X):
    return X.tanh()
def net(X):
    X = X.view((-1, num_inputs))
    H = tanh(torch.matmul(X, W1) + b1)
    O = torch.matmul(H, W2) + b2
    return O
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
```

    epoch 1, loss 0.0029, train acc 0.730, test acc 0.672
    epoch 2, loss 0.0020, train acc 0.815, test acc 0.811
    epoch 3, loss 0.0018, train acc 0.828, test acc 0.807
    epoch 4, loss 0.0017, train acc 0.845, test acc 0.842
    epoch 5, loss 0.0016, train acc 0.851, test acc 0.824



```python

```


```python

```


```python
from torch import nn
from torch.nn import init
num_inputs, num_outputs, num_hiddens,num_hiddens2 = 784, 10, 256, 128
    
net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        #nn.Sigmoid(),
        nn.Linear(num_hiddens, num_hiddens2),
        nn.Sigmoid(),
        nn.Linear(num_hiddens2, num_outputs), 
        )
    
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,root='~/Datasets/FashionMNIST')
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
```

    epoch 1, loss 0.0059, train acc 0.405, test acc 0.646
    epoch 2, loss 0.0028, train acc 0.718, test acc 0.770
    epoch 3, loss 0.0023, train acc 0.780, test acc 0.794
    epoch 4, loss 0.0020, train acc 0.809, test acc 0.813
    epoch 5, loss 0.0019, train acc 0.827, test acc 0.818



```python

```


```python
d=256*256
h=1000
q=10
```


```python
d*h+h*q
```




    65546000




```python

```
