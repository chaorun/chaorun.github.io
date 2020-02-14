---
layout: post
title:  "Torch Text MachineLearning"
date:   2020-02-14 10:34:47 +0800
categories: jekyll update
---

# 这个Notebook 主要讲的是文本相关的东西
包含如下内容：
## 文本预处理
## 语言模型
## RNN基础

## 文本预处理
```python
from __future__ import print_function, division
from IPython import display
from matplotlib import pyplot as plt
from collections import OrderedDict
import sys,re,collections,random,math
#为了方便，引入自己的包
sys.path.append('/Users/chaorun/')
from SoTinShuiLib import *

import torch
from torch import nn
from torch.nn import init
import torch.utils.data as Data

import torchvision
import torchvision.transforms as transforms

sys.path.append("../Dive-into-DL-PyTorch/code/") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l
```

## Text Configuration

### Remember download the timemachine.txt

```python

def read_time_machine():
    with open('data/timemachine.txt', 'r') as f:
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines


lines = read_time_machine()
print('# sentences %d' % len(lines))
```

    # sentences 3583


### 我们对每个句子进行分词，也就是将一个句子划分成若干个词（token），转换为一个词的序列。


```python
def tokenize(sentences, token='word'):
    """Split sentences into word or char tokens"""
    if token == 'word':
        return [sentence.split(' ') for sentence in sentences]
    elif token == 'char':
        return [list(sentence) for sentence in sentences]
    else:
        print('ERROR: unkown token type '+token)

tokens = tokenize(lines)
tokens[0:2]
```




    [[''],
     ['the',
      'project',
      'gutenberg',
      'ebook',
      'of',
      'the',
      'time',
      'machine',
      'by',
      'h',
      'g',
      'wells']]



#建立字典
为了方便模型处理，我们需要将字符串转换为数字。因此我们需要先构建一个字典（vocabulary），将每个词映射到一个唯一的索引编号。


```python
class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = count_corpus(tokens)  # : 
        self.token_freqs = list(counter.items())
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(sentences):
    tokens = [tk for st in sentences for tk in st]
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数
#我觉得Vocab可以用numpy更快的实现
```


```python
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[0:10])
```

    [('', 0), ('the', 1), ('project', 2), ('gutenberg', 3), ('ebook', 4), ('of', 5), ('time', 6), ('machine', 7), ('by', 8), ('h', 9)]



```python
for i in range(8, 10):
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

    words: ['']
    indices: [0]
    words: ['title', 'the', 'time', 'machine']
    indices: [41, 1, 6, 7]


用现有工具进行分词
我们前面介绍的分词方式非常简单，它至少有以下几个缺点:

标点符号通常可以提供语义信息，但是我们的方法直接将其丢弃了
类似“shouldn't", "doesn't"这样的词会被错误地处理
类似"Mr.", "Dr."这样的词会被错误地处理
我们可以通过引入更复杂的规则来解决这些问题，但是事实上，有一些现有的工具可以很好地进行分词，我们在这里简单介绍其中的两个：spaCy和NLTK。

下面是一个简单的例子：


```python
text = "Apart from individual data packages, you can download the entire collection (using “all”), or just the data required for the examples and exercises in the book (using “book”),or just the corpora and no grammars or trained models (using “all-corpora”)."#"Mr. Chen doesn't agree with my suggestion."
```


```python
import spacy
#ref https://spacy.io/usage/models
#sudo /opt/intel/intelpython3/bin/python3 -m  spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
print([token.text for token in doc])
```

    ['Apart', 'from', 'individual', 'data', 'packages', ',', 'you', 'can', 'download', 'the', 'entire', 'collection', '(', 'using', '“', 'all', '”', ')', ',', 'or', 'just', 'the', 'data', 'required', 'for', 'the', 'examples', 'and', 'exercises', 'in', 'the', 'book', '(', 'using', '“', 'book”),or', 'just', 'the', 'corpora', 'and', 'no', 'grammars', 'or', 'trained', 'models', '(', 'using', '“', 'all', '-', 'corpora', '”', ')', '.']



```python

```

# 语言模型与数据集


```python
with open('data/jaychou_lyrics.txt') as f:
    corpus_chars = f.read()
print(len(corpus_chars))
print(corpus_chars[: 40])
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[: 10000]
```

    63282
    想要有直升机
    想要和你飞到宇宙去
    想要和你融化在一起
    融化在宇宙里
    我每天每天每



```python
idx_to_char = list(set(corpus_chars)) # 去重，得到索引到字符的映射
char_to_idx = {char: i for i, char in enumerate(idx_to_char)} # 字符到索引的映射
vocab_size = len(char_to_idx)
print(vocab_size)

corpus_indices = [char_to_idx[char] for char in corpus_chars]  # 将每个字符转化为索引，得到一个索引的序列
sample = corpus_indices[: 20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)
```

    1027
    chars: 想要有直升机 想要和你飞到宇宙去 想要和
    indices: [98, 764, 638, 803, 670, 958, 587, 98, 764, 1016, 295, 1007, 848, 637, 364, 114, 587, 98, 764, 1016]



```python
def load_data_jay_lyrics():
    with open('data/jaychou_lyrics.txt') as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size
```


```python
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为对于长度为n的序列，X最多只有包含其中的前n - 1个字符
    num_examples = (len(corpus_indices) - 1) // num_steps  # 下取整，得到不重叠情况下的样本个数
    example_indices = [i * num_steps for i in range(num_examples)]  # 每个样本的第一个字符在corpus_indices中的下标
    random.shuffle(example_indices)

    def _data(i):
        # 返回从i开始的长为num_steps的序列
        return corpus_indices[i: i + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(0, num_examples, batch_size):
        # 每次选出batch_size个随机样本
        batch_indices = example_indices[i: i + batch_size]  # 当前batch的各个样本的首字符的下标
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield torch.tensor(X, device=device), torch.tensor(Y, device=device)
```


```python
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
```

    X:  tensor([[18, 19, 20, 21, 22, 23],
            [ 6,  7,  8,  9, 10, 11]]) 
    Y: tensor([[19, 20, 21, 22, 23, 24],
            [ 7,  8,  9, 10, 11, 12]]) 
    
    X:  tensor([[ 0,  1,  2,  3,  4,  5],
            [12, 13, 14, 15, 16, 17]]) 
    Y: tensor([[ 1,  2,  3,  4,  5,  6],
            [13, 14, 15, 16, 17, 18]]) 
    



```python
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_len = len(corpus_indices) // batch_size * batch_size  # 保留下来的序列的长度
    corpus_indices = corpus_indices[: corpus_len]  # 仅保留前corpus_len个字符
    indices = torch.tensor(corpus_indices, device=device)
    indices = indices.view(batch_size, -1)  # resize成(batch_size, )
    batch_num = (indices.shape[1] - 1) // num_steps
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
```


```python
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
```

    X:  tensor([[ 0,  1,  2,  3,  4,  5],
            [15, 16, 17, 18, 19, 20]]) 
    Y: tensor([[ 1,  2,  3,  4,  5,  6],
            [16, 17, 18, 19, 20, 21]]) 
    
    X:  tensor([[ 6,  7,  8,  9, 10, 11],
            [21, 22, 23, 24, 25, 26]]) 
    Y: tensor([[ 7,  8,  9, 10, 11, 12],
            [22, 23, 24, 25, 26, 27]]) 
    



```python
#给定训练数据[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]，
#批量大小为batch_size=2，时间步数为2，使用本节课的实现方法进行相邻采样，第二个批量为：
for X, Y in data_iter_consecutive([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], batch_size=2, num_steps=2):
    print('X: ', X, '\nY:', Y, '\n')
```

    X:  tensor([[0, 1],
            [5, 6]]) 
    Y: tensor([[1, 2],
            [6, 7]]) 
    
    X:  tensor([[2, 3],
            [7, 8]]) 
    Y: tensor([[3, 4],
            [8, 9]]) 
    


# Recurrence Neural Network


```python

```


```python
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```


```python
def one_hot(x, n_class, dtype=torch.float32):
    result = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)  # shape: (n, n_class)
    result.scatter_(1, x.long().view(-1, 1), 1)  # result[i, x[i, 0]] = 1
    return result
    
x = torch.tensor([0, 2,7 ])
x_one_hot = one_hot(x, 10)#vocab_size)
print(x_one_hot)
print(x_one_hot.shape)
print(x_one_hot.sum(axis=1))
```

    tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])
    torch.Size([3, 10])
    tensor([1., 1., 1.])



```python
def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

X = torch.arange(10).view(2, 5)
inputs = to_onehot(X, 10)#vocab_size)
print(len(inputs), inputs[0].shape,X)
print(inputs)
```

    5 torch.Size([2, 10]) tensor([[0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]])
    [tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]]), tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]), tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]]), tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]]), tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])]



```python
print(inputs[1])
```

    tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])



```python
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
# num_inputs: d
# num_hiddens: h, 隐藏单元的个数是超参数
# num_outputs: q

def get_params():
    def _one(shape):
        param = torch.zeros(shape, device=device, dtype=torch.float32)
        nn.init.normal_(param, 0, 0.01)
        return torch.nn.Parameter(param)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device))
    return (W_xh, W_hh, b_h, W_hq, b_q)
```


```python
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)
```


```python
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```


```python
print(X.shape)
print(num_hiddens)
print(vocab_size)
state = init_rnn_state(X.shape[0], num_hiddens, device)
inputs = to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(inputs), inputs[0].shape)
print(len(outputs), outputs[0].shape)
print(len(state), state[0].shape)
print(len(state_new), state_new[0].shape)
```

    torch.Size([2, 5])
    256
    1027
    5 torch.Size([2, 1027])
    5 torch.Size([2, 1027])
    1 torch.Size([2, 256])
    1 torch.Size([2, 256])


# 裁剪梯度


```python
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)
```

# 定义预测函数


```python
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]   # output记录prefix加上预测的num_chars个字符
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y[0].argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])
```


```python
predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
            device, idx_to_char, char_to_idx)
```




    '分开映盘坏屉屋威步宁鸠碌'




```python
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用随机采样，在epoch开始时初始化隐藏状态  WHY?
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach_()
            # inputs是num_steps个形状为(batch_size, vocab_size)的矩阵
            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            
            # Y的形状是(batch_size, num_steps)，转置后再变成形状为
            # (num_steps * batch_size,)的向量，这样跟输出的行一一对应
            y = torch.flatten(Y.T)
            
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())
            
            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
```


```python
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['思念', '想吃']
```


```python
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, device, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
```

    epoch 50, perplexity 70.346115, time 0.31 sec
     - 思念 我想要 说爱我 我想就 我想就 我想就 我想就 我想就 我想就 我想就 我想就 我想就 我想就 我
     - 想吃 我想要 我爱就 一颗四颗三颗四 我想要你 你着我有多 我不能 说爱我 一子四颗三颗 我有要 一子两
    epoch 100, perplexity 9.545188, time 0.32 sec
     - 思念 娘子她人去棍 一哼哈依 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 快
     - 想吃你不着我 想要球 快沉默 娘子我有手汉 有话去人医药箱说 别怪我 别怪我 说你的人不著我 说乡的爹笑
    epoch 150, perplexity 2.914106, time 0.32 sec
     - 思念 娘子的美等人我 家乡的爹娘早已苍老了轮廓 娘子我欠你太多 一壶好酒 再来一碗热记 配者几敌的牛肉 
     - 想吃你 陪我去吃汉堡  说穿了其实我的愿望就怎么小 就在再每天 祷我的始乡相忧 牧少林没有 我马儿有些瘦
    epoch 200, perplexity 1.600396, time 0.31 sec
     - 思念 有一心美哭 到后它什么 懂窝在角落 不爽就反驳 到底拽什么 懂不懂 球烧知 装属了那信片老 干什么
     - 想吃有直升要 想要和你飞到宇宙去 想要和你融化在一起 融化在宇宙里 我每天每 又谁说 分数怎么停 一只灰
    epoch 250, perplexity 1.323169, time 0.31 sec
     - 思念撑 像可植物铜走 唱狠年以后 还爱让人难过 心伤透 娘子她人在江南等我 泪不休 语沉默 一壶她武在江
     - 想吃你 陪我去吃汉堡  说穿了其实我的愿望就怎么小 就怎么每天祈祷我的心跳你知道  杵在伊斯坦堡 却只想



```python

```

#  循环神经网络的简介实现


```python
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
num_steps, batch_size = 35, 2
X = torch.rand(num_steps, batch_size, vocab_size)
state = None
Y, state_new = rnn_layer(X, state)
print(Y.shape, state_new.shape)
```

    torch.Size([35, 2, 256]) torch.Size([1, 2, 256])



```python
class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1) 
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, inputs, state):
        # inputs.shape: (batch_size, num_steps)
        X = to_onehot(inputs, vocab_size)
        X = torch.stack(X)  # X.shape: (num_steps, batch_size, vocab_size)
        hiddens, state = self.rnn(X, state)
        hiddens = hiddens.view(-1, hiddens.shape[-1])  # hiddens.shape: (num_steps * batch_size, hidden_size)
        output = self.dense(hiddens)
        return output, state
def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                      char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output记录prefix加上预测的num_chars个字符
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y.argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])
```


```python
model = RNNModel(rnn_layer, vocab_size).to(device)
predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)
```




    '分开因性嘟笔柔壁性嘟笔柔'




```python
def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
        state = None
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态
                if isinstance (state, tuple): # LSTM, state:(h, c)  
                    state[0].detach_()
                    state[1].detach_()
                else: 
                    state.detach_()
            (output, state) = model(X, state) # output.shape: (num_steps * batch_size, vocab_size)
            y = torch.flatten(Y.T)
            l = loss(output, y.long())
            
            optimizer.zero_grad()
            l.backward()
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))
```


```python
num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)
```

    epoch 50, perplexity 12.296707, time 0.23 sec
     - 分开 你在那里 我不要再想  不知再觉 我不了空着 我不要再想  不要再这样打我妈  想要你的爱女人 坏
     - 不分开 我你在你里 我想要你的微笑每天都说不多 一颗两颗三颗四颗 连成线背著背默默许下心愿 我不要再一个 
    epoch 100, perplexity 1.281940, time 0.24 sec
     - 分开 你在那里 在小村外的溪边河口默默等著我 娘子依旧每日折一枝杨柳 你在那里 在小村外的溪边 默默等待
     - 不分开 想你 没有  我不懂 你的没有 在对着我进攻 我   说 有你看着我 多难熬  没有了在我 我有要
    epoch 150, perplexity 1.063682, time 0.25 sec
     - 分开 你在橱窗  凝视碑文的字眼 我却在旁静静欣赏你那张我深爱的脸 祭司 神殿 征战 弓箭 是谁的从前 
     - 不分开 了你不手 你 无处失的 我每天 想你我 你的着 有人 不发  有上为回忆的在西元前 深埋在美索不达
    epoch 200, perplexity 1.031007, time 0.24 sec
     - 分开 你在橱窗  凝视碑文的字眼 我却在旁静静欣赏你那张我深爱的脸 祭司 神殿 征战 弓箭 是谁的从前 
     - 不分开 了它不定 只 上一脚踢开 快使用双截棍 哼哼哈兮 快使用双截棍 哼哼哈兮 习武之人切记 仁者无敌 
    epoch 250, perplexity 1.019415, time 0.25 sec
     - 分开 你不知不  你已经离 我 不想着你 没有对不我不多 你  我说妈怎 我对不的生活 说 情你的太有 
     - 不分开 了它不定看球 有话去直你 我有你的生活 不知不觉 你已经离开我 不知不觉 我跟了这节奏 后知后觉 



```python

```


```python

```


```python

```
