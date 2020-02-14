---
layout: post
title:  "Torch Softmax"
date:   2020-02-14 12:34:47 +0800
categories: jekyll update
---

```

```

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
```


```python
#我觉得还是放在nas里比较好
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', 
                                                train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', 
                                               train=False, download=False, transform=transforms.ToTensor())
```


```python
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))
```

    <class 'torchvision.datasets.mnist.FashionMNIST'>
    60000 10000



```python
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width
```

    torch.Size([1, 28, 28]) 9



```python
# 本函数已保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy(),cmap='gray')
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
```


```python
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))
```


![svg](output_5_0.svg)



```python
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
```


```python
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
```

    1.03 sec



```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```


```python
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
```


```python
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True) 
```




    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)




```python
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=1, keepdim=True))
```

    tensor([[5, 7, 9]])
    tensor([[ 6],
            [15]])



```python
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
```


```python
X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1))
```

    tensor([[0.1998, 0.2368, 0.1559, 0.1310, 0.2765],
            [0.3365, 0.1376, 0.1263, 0.2694, 0.1303]]) tensor([1.0000, 1.0000])



```python
a=torch.rand((2, 5)).exp()
a/a.sum(dim=1,keepdim=True)
```




    tensor([[0.2334, 0.1994, 0.2697, 0.1835, 0.1141],
            [0.1431, 0.3011, 0.1531, 0.1824, 0.2203]])




```python
def net(X):
    mmresult=torch.mm(X.view((-1, num_inputs)), W) + b
    return softmax(mmresult)
```


```python
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))
```




    tensor([[0.1000],
            [0.5000]])




```python
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
```


```python
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()
```


```python
print(accuracy(y_hat, y))
```

    0.5



```python
# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
```


```python
print(evaluate_accuracy(test_iter, net))
```

    0.1435



```python
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        train_acc = evaluate_accuracy(train_iter, net)
        test_acc = evaluate_accuracy(test_iter, net)
        #print(train_acc)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


```


```python
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

```

    epoch 1, loss 0.7852, train acc 0.749, test acc 0.785
    epoch 2, loss 0.5709, train acc 0.813, test acc 0.809
    epoch 3, loss 0.5251, train acc 0.826, test acc 0.821
    epoch 4, loss 0.5017, train acc 0.832, test acc 0.821
    epoch 5, loss 0.4860, train acc 0.837, test acc 0.828



```python
softmax(torch.tensor([[100,101,102]],dtype=torch.double)),softmax(torch.tensor([[-2,-1,0]],dtype=torch.double))
```




    (tensor([[0.0900, 0.2447, 0.6652]], dtype=torch.float64),
     tensor([[0.0900, 0.2447, 0.6652]], dtype=torch.float64))




```python
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])
```


![svg](output_25_0.svg)



```python
import torch
from torch import nn
from torch.nn import init
```


```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```


```python
num_inputs = 784
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y

net = LinearNet(num_inputs, num_outputs)
```


```python
# 本函数已保存在d2lzh_pytorch包中方便以后使用
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
```


```python
from collections import OrderedDict

net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)
```


```python
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0) 
```




    Parameter containing:
    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)




```python
loss = nn.CrossEntropyLoss()
```


```python
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
```


```python
num_epochs = 5
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
```

    epoch 1, loss 0.0031, train acc 0.750, test acc 0.789
    epoch 2, loss 0.0022, train acc 0.812, test acc 0.808
    epoch 3, loss 0.0021, train acc 0.826, test acc 0.820
    epoch 4, loss 0.0020, train acc 0.833, test acc 0.825
    epoch 5, loss 0.0019, train acc 0.837, test acc 0.825


