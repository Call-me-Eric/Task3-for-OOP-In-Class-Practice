# readme

## info

根据投票，我们选择了[作业2](/大作业/实验课+B+Tensor+and+ViT.pdf)作为大作业内容。

根据pdf，此大作业需要完成三个板块：

```text
Tensor.h,Model.h,main.cpp
```

目前分工：

Tensor.h 由 @? 完成
Model.h 由 @Call\_me\_Eric 完成
main.cpp 由 @? 完成

## 每个部分需要完成的接口

### tensor.h

需要完成所有pdf中出现的接口。

有待补充$\dots$

### model.h

需要完成：

1. Layer 抽象基类 done
2. Linear 线性层类（用于完成“仿射变换”）done
3. PatchEmbedding 类（用于将$28\times 28$的灰度图转换成32维向量）
4. LayerNorm 类（用于做矩阵乘法和加法的线性变换）
5. MultiHeadAttention 类（还没看）
6. 前馈网络类(MLP)（还没看）
7. Transformer 编码器块(TransformerBlock)（还没看）
8. VisionTransformer 类（还没看）

### main.cpp

需要完成 WeightLoader类，与助教确认前端的交互界面。

需要统筹计划模型如何训练，以及套一层壳之类的。
