# Model.h文档

在这里我会写一点关于Model.h的相关信息，比如接口，比如调用规则等。

## 约定

以下做一些描述上的约定

### Tensor\<T> A

1. 我们称A**形如**$\{a_1,a_2,\dots,a_n\}$，表示其有 $n$ 维，在Tensor.h中 $shape = \{a_1,a_2,\dots,a_n\}$，而某一位可以表示为 $A(x_1,x_2,\dots,x_n),(\forall i\in[1,n], 0\le x_i\le a_i)$

## Layer类

抽象基类，只提供了

```cpp
Tensor<float> forward(const Tensor<float>& x) = 0;
```

接口

## Linear类

### 对于构造方法

1. ~~可以传入in\_feature和out\_feature，然后会构造出最基础的线性层类~~。已停用，因为助教说会给参数。
2. 传入Tensor\<float> \_weight,Tensor\<float> \_bias，然后直接调用这个参数。

我们保证\_weight形如 $\{hidden\_dim,Output\}$，\_bias形如 $\{1,Output\}$

### 对于forward方法

格式：

```cpp
Tensor<float> forward(const Tensor<float>& x);
```

需要传入一个Tensor\<float> x，形如 $\{batch=1,tokens=1+16,hidden\_dim=32\}$，然后会对其进行仿射变换（可以近似看作 $y=kx+b$），然后返回这个 $y$。

具体而言，这个方法会返回 $x\times\_weight+\_bias$。

其中 $\times$ 是 $matmul$，表示将 $\{B,tokens,hidden\_dim\}\times \{hidden\_dim,Output\}$，对最后两维做矩阵乘法，最后输出$\{B,tokens,Output\}$。

$+$ 是 $bias\_add$，表示将 $\forall (x_1,x_2,\dots,x_n),A(x_1,x_2,\dots,x_n)+= bias(x_n)$

## PatchEmbedding类

### 对于构造方法

1. ~~什么都不传，然后会构造出最基础的类~~。已停用，因为助教说会给参数。
2. 传入Tensor\<float> cls\_token,Tensor\<float> pos\_embedding,Linear linear，然后直接调用这些参数。需要在loader中写。

我们约定，cls\_token和pos\_embedding都形如 $\{hidden\_dim\}$，linear的构造见Linear类。

### 对于forward方法

```cpp
Tensor<float> forward(const Tensor<float>& x);
```

传入一个张量x，形如 $\{batch=1,image_h=28,image_w=28\}$。

输出一个张量，形如 $\{batch,tokens=1+16,hidden\_dim=32\}$

首先，我们会将image分割成$4\times 4$ 个patches，然后对每个patch，将其flatten()展平成形如 $\{49\}$ 的向量（张量）。

接下来我们通过$Linear(\{49,32\},\{1,32\}).forward(\{49\})$ 将其转换成 $\{32\}$的token。

其中的Linear(\_weight,\_bias)参数由助教给出，文件名待定。

将所有的token组合起来，同时在前面加上一个特殊的CLS token，便成为最终的输出。

## LayerNorm类

归一化，通过softmax函数，将最后一维的总和保持在1，可能需要通过调用Tensor.h中的softmax函数。

## MultiHeadAttention类


