# Model.h文档

在这里我会写一点关于Model.h的相关信息，比如接口，比如调用规则等。

以下对Tensor\<T> x做一些描述上的约定：

1. 我们称A**形如**$\{a_1,a_2,\dots,a_n\}$，表示其有 $n$ 维，在Tensor.h中 $shape = \{a_1,a_2,\dots,a_n\}$，而某一位可以表示为 $A(x_1,x_2,\dots,x_n),(\forall i\in[1,n], 0\le x_i\le a_i)$

## Layer类

抽象基类，只提供了

```cpp
Tensor<float> forward(const Tensor<float>& x) = 0;
```

接口

## Linear类

对于构造方法：

1. ~~可以传入in\_feature和out\_feature，然后会构造出最基础的线性层类~~。已停用，因为助教说会给文件。
2. 传入Tensor\<float> \_weight,Tensor\<float> \_bias，然后直接调用这个参数。

我们保证\_weight形如 $\{\}$

对于forward方法：
格式：

```cpp
Tensor<float> forward(const Tensor<float>& x);
```

需要传入一个Tensor\<float> x，然后会对其进行仿射变换（可以近似看作 $y=kx+b$），然后返回这个 $y$。

具体而言，这个方法会返回 $x\times\_weight+\_bias$，其中 $\times$ 是 $matmul$，而 $+$ 是 $bias_add$。
