# Model.h文档

本文件说明 `大作业/Model.h` 中已实现的类、输入输出形状、参数注入接口以及使用方法。

## 约定

- `Tensor<T>::shape()` 返回张量形状，例如 `{B, T, hidden_dim}`。
- `Tensor<T>` 的元素访问使用 `operator()` 多维索引。
- `Tensor.h` 已实现 `matmul`, `bias_add`, `softmax`, `slice`, `concat`, `reshape`, `permute` 等操作。
- 本代码只保留网络结构和超参数接口，所有权重、偏置、CLS token、位置编码等参数应由外部载入后注入。

## Layer 类

`Layer` 是抽象基类，只提供接口：

```cpp
virtual Tensor<float> forward(const Tensor<float>& x) = 0;
```

所有子类都应实现 `forward`，接收输入张量并返回输出张量。

## Linear 类

`Linear` 表示一个线性仿射变换层。

### 构造方法

- `Linear(size_t in_f = 0, size_t out_f = 0)`
  - 仅使用超参数 `in_f` 和 `out_f` 构造张量结构。
  - 默认权重和偏置不会被赋予实际训练值，调用方应在加载参数后使用 `set_weights` 注入。
- `Linear(const Tensor<float>& weight, const Tensor<float>& bias)`
  - 直接使用给定权重和偏置。
  - `weight` 形状应为 `{in_f, out_f}`。
  - `bias` 形状应为 `{1, out_f}`。

### 参数注入

- `set_weights(const Tensor<float>& weight, const Tensor<float>& bias)`
- `weight()` / `bias()` 返回当前参数引用。

### forward 方法

```cpp
Tensor<float> forward(const Tensor<float>& x);
```

- 输入 `x` 形状为 `{B, tokens, in_f}`。
- 计算过程：
  - `x.matmul(_weight)`：对最后两维执行矩阵乘法。
  - `bias_add(_bias)`：对输出的最后一维加偏置。
- 输出形状为 `{B, tokens, out_f}`。

`Linear` 等价于 `y = xW + b`。

## PatchEmbedding 类

`PatchEmbedding` 负责将输入图像切成 patch，并把每个 patch 映射到 `hidden_dim` 维 token，同时添加 `CLS` token 和位置编码。

### 构造方法

- `PatchEmbedding(size_t input_h = 28, size_t input_w = 28, size_t patch_h = 7, size_t patch_w = 7, size_t hidden_dim = 32)`
  - 仅保留超参数：输入图像尺寸、patch 大小、隐藏维度。
  - 计算 `num_patches = (input_h / patch_h) * (input_w / patch_w)`。
- `PatchEmbedding(const Tensor<float>& cls_token, const Tensor<float>& pos_embedding, const Linear& linear, size_t input_h = 28, size_t input_w = 28, size_t patch_h = 7, size_t patch_w = 7)`
  - 用于将外部已加载的 CLS token、位置编码和 patch 投影一起构造对象。

### 参数注入

- `set_cls_token(const Tensor<float>& token)`
- `set_pos_embedding(const Tensor<float>& embedding)`
- `set_projection(const Linear& proj)`

### forward 方法

```cpp
Tensor<float> forward(const Tensor<float>& x);
```

- 输入 `x` 形状为 `{B, input_h, input_w}` 的灰度图像。
- 计算过程：
  1. 按 `patch_h x patch_w` 切分图像。
  2. 将每个 patch reshape 为 `{B, patch_h * patch_w}`。
  3. 调用 `linear.forward(...)` 映射为 `{B, hidden_dim}`。
  4. reshape 为 `{B, 1, hidden_dim}`，加上对应位置编码。
  5. 拼接到输出序列中。
- 输出形状为 `{B, 1 + num_patches, hidden_dim}`，其中第一个 token 是 CLS token。

> 这里的 `PatchEmbedding` 只由超参数定义结构，具体 `cls_token`、`pos_embedding` 和 `linear` 参数需要外部加载后注入。

## LayerNorm 类

`LayerNorm` 当前实现通过 softmax 对最后一维归一化。

### forward 方法

```cpp
Tensor<float> forward(const Tensor<float>& x);
```

- 返回 `x.softmax(x.rank() - 1)`。
- 输出形状与输入一致，最后一维元素和为 1。

## MultiHeadAttention 类

`MultiHeadAttention` 实现了多头自注意力计算。

### 构造方法

- `MultiHeadAttention(size_t hidden_dim, size_t num_heads)`
  - 仅保留超参数 `hidden_dim` 和 `num_heads`。
  - 要求 `hidden_dim % num_heads == 0`。

### 参数注入

- `set_q_proj(const Tensor<float>& weight, const Tensor<float>& bias)`
- `set_k_proj(const Tensor<float>& weight, const Tensor<float>& bias)`
- `set_v_proj(const Tensor<float>& weight, const Tensor<float>& bias)`
- `set_out_proj(const Tensor<float>& weight, const Tensor<float>& bias)`

### forward 方法

```cpp
Tensor<float> forward(const Tensor<float>& x);
```

- 输入 `x` 形状为 `{B, T, hidden_dim}`。
- 计算过程：
  1. 生成 `Q`, `K`, `V`。
  2. reshape 为 `{B, T, num_heads, head_dim}`。
  3. permute 为 `{B, num_heads, T, head_dim}`。
  4. 计算 `scores = Q * K^T`。
  5. 缩放并对最后一维 `softmax`。
  6. 计算 `attention_weights.matmul(V)`。
  7. reshape 回 `{B, T, hidden_dim}`。
  8. 通过 `_out_proj.forward(...)` 得到最终输出。
- 输出形状为 `{B, T, hidden_dim}`。

### get_attention_map 方法

```cpp
Tensor<float> get_attention_map() const;
```

- 返回最后一次前向计算得到的注意力权重。
- 输出形状为 `{B, num_heads, T, T}`。

## MLP 类

`MLP` 表示一个前馈网络层，用于 Transformer 编码器中的非线性变换。

### 构造方法

- `MLP(size_t hidden_dim = 32, size_t mlp_dim = 64)`
  - 仅保留超参数 `hidden_dim` 和 `mlp_dim`。

### 参数注入

- `set_fc1(const Tensor<float>& weight, const Tensor<float>& bias)`
- `set_fc2(const Tensor<float>& weight, const Tensor<float>& bias)`

### forward 方法

```cpp
Tensor<float> forward(const Tensor<float>& x);
```

- 过程：
  1. `x` 经过 `_fc1`。
  2. 经过 GELU 激活。
  3. 经过 `_fc2`。
- 输出形状为 `{B, T, hidden_dim}`。

## TransformerBlock 类

`TransformerBlock` 组合了标准的 Transformer 编码器模块。

### 构造方法

- `TransformerBlock(size_t hidden_dim = 32, size_t num_heads = 4, size_t mlp_dim = 64)`
  - 仅保留超参数 `hidden_dim`, `num_heads`, `mlp_dim`。

### 参数注入

- `set_attention_weights(...)`：注入 Q/K/V/out 投影权重和偏置。
- `set_mlp_weights(...)`：注入 MLP 的 `fc1` 和 `fc2` 权重与偏置。

### forward 方法

- 预归一化：先执行 `_norm1`，再进行自注意力。
- 残差相加后再执行 `_norm2` 和 MLP。
- 输出形状为 `{B, T, hidden_dim}`。

## VisionTransformer 类

`VisionTransformer` 实现了完整的 ViT 推理前向过程。

### 构造方法

- `VisionTransformer(size_t input_h = 28, size_t input_w = 28, size_t patch_h = 7, size_t patch_w = 7, size_t hidden_dim = 32, size_t num_heads = 4, size_t mlp_dim = 64, size_t num_layers = 1, size_t num_classes = 10)`
  - 仅保留超参数结构。

### 参数注入

- `set_patch_embedding(const PatchEmbedding& patch_embed)`
- `set_head(const Tensor<float>& weight, const Tensor<float>& bias)`
- `set_block(size_t index, const TransformerBlock& block)`

### forward 方法

- 先通过 `PatchEmbedding` 获取 token 序列。
- 再依次通过多个 `TransformerBlock`。
- 最后对 token 做 `LayerNorm` 并提取 `CLS` token。
- 输出 `Linear` 分类头结果，形状为 `{B, num_classes}`。

## 参数加载与使用示例

以下示例展示如何在外部加载参数后使用 `Model.h` 类。

```cpp
// 1. 创建网络结构，超参数由代码给出
PatchEmbedding patch_embed(28, 28, 7, 7, 32);
MultiHeadAttention mha(32, 4);
MLP mlp(32, 64);
TransformerBlock block(32, 4, 64);
VisionTransformer vit(28, 28, 7, 7, 32, 4, 64, 1, 10);

// 2. 从文件加载权重和偏置，假设已经得到以下 Tensor
Tensor<float> q_weight({32, 32});
Tensor<float> q_bias({1, 32});
Tensor<float> k_weight({32, 32});
Tensor<float> k_bias({1, 32});
Tensor<float> v_weight({32, 32});
Tensor<float> v_bias({1, 32});
Tensor<float> out_weight({32, 32});
Tensor<float> out_bias({1, 32});

// 3. 注入模型参数
mha.set_q_proj(q_weight, q_bias);
mha.set_k_proj(k_weight, k_bias);
mha.set_v_proj(v_weight, v_bias);
mha.set_out_proj(out_weight, out_bias);

Tensor<float> fc1_weight({32, 64});
Tensor<float> fc1_bias({1, 64});
Tensor<float> fc2_weight({64, 32});
Tensor<float> fc2_bias({1, 32});
block.set_mlp_weights(fc1_weight, fc1_bias, fc2_weight, fc2_bias);

Tensor<float> cls_token({1, 1, 32});
Tensor<float> pos_embed({1, 17, 32});
Linear project(49, 32);
patch_embed.set_cls_token(cls_token);
patch_embed.set_pos_embedding(pos_embed);
patch_embed.set_projection(project);

Tensor<float> head_weight({32, 10});
Tensor<float> head_bias({1, 10});
vit.set_head(head_weight, head_bias);

// 4. 前向推理
Tensor<float> image({1, 28, 28});
Tensor<float> logits = vit.forward(image);
```

> 说明：本示例中权重加载过程不在 `Model.h` 内实现，调用方需要根据文件格式自行读取 Tensor 数据后调用 setter 注入。

## 已完成内容总结

- `Layer`：定义 `forward` 抽象接口。
- `Linear`：实现 `matmul` + `bias_add` 的仿射变换，并支持外部参数注入。
- `PatchEmbedding`：实现结构超参数接口，并支持外部注入 `cls_token`、`pos_embedding`、`projection`。
- `LayerNorm`：实现最后一维 softmax 归一化。
- `MultiHeadAttention`：实现 Q/K/V 投影、注意力分数计算、缩放 softmax、注意力加权求和与输出投影，并支持外部投影参数注入。
- `MLP`：实现前馈网络并支持外部参数注入。
- `TransformerBlock`：实现编码器块并支持外部注入注意力和 MLP 参数。
- `VisionTransformer`：实现 ViT 结构并支持外部注入分类头与子模块参数。
