# main.cpp 文档

## 概述

`main.cpp` 是 Tiny Vision Transformer 推理系统的入口文件，负责：

1. 从二进制文件中读取 MNIST 灰度图像
2. 从权重目录中加载所有模型参数
3. 构建并初始化完整的 VisionTransformer 模型
4. 执行前向推理，输出手写数字的预测结果

---

## 命令行接口

```bash
./程序名 <image_path> <weight_dir>
```

| 参数 | 说明 |
|------|------|
| `image_path` | MNIST 图像的二进制文件路径（28×28 灰度图，每像素 1 字节） |
| `weight_dir` | 存放模型权重 `.wts` 文件的目录路径 |

---

## 模型超参数

以下参数在 `main` 函数中以常量形式硬编码，与实验规格严格对应：

| 参数 | 值 | 说明 |
|------|----|------|
| `INPUT_H / INPUT_W` | 28 | 输入图像尺寸 |
| `PATCH_H / PATCH_W` | 7 | Patch 大小，产生 16 个 patch |
| `HIDDEN_DIM` | 32 | Token 隐藏维度 |
| `NUM_HEADS` | 2 | 多头注意力头数 |
| `MLP_DIM` | 64 | 前馈网络隐藏维度 |
| `NUM_LAYERS` | 2 | Transformer 编码器块数量 |
| `NUM_CLASSES` | 10 | 分类类别数（0~9） |

---

## 函数说明

### `load_tensor`

```cpp
Tensor<float> load_tensor(const string& filepath);
```

从 `.wts` 文本文件中读取张量数据。

**文件格式：**

- 第一行为形状注释，格式为 `# shape: R C`（2D）或 `# shape: N`（1D bias）
- 后续行为空格分隔的浮点数

**处理逻辑：**

- 若解析到两个正整数 `R C`，则构造形状 `{R, C}` 的张量
- 若只解析到一个正整数 `N`（1D bias 情形），则自动转为形状 `{1, N}`，与 `Linear` 层期望的 bias 形状一致
- 若实际读取数据量与张量容量不符，输出警告但不中止，取二者最小值进行填充

**异常：** 文件无法打开或 shape 行格式非法时抛出 `runtime_error`。

---

### `load_image`

```cpp
Tensor<float> load_image(const string& path);
```

从二进制文件中读取 MNIST 28×28 灰度图像。

- 输出张量形状：`{1, 28, 28}`
- 每个像素值除以 255.0f，归一化到 `[0, 1]`
- 按行优先顺序逐像素读取

**异常：** 文件无法打开或像素读取失败时抛出 `runtime_error`。

---

### `load_block`

```cpp
TransformerBlock load_block(const string& wdir, int idx,
                            size_t hidden_dim, size_t num_heads, size_t mlp_dim);
```

从权重目录中加载第 `idx` 个 Transformer 编码器块的全部参数。

**加载的权重文件（以 `blocks.<idx>.` 为前缀）：**

| 文件后缀 | 对应参数 |
|----------|----------|
| `attn.q.weight.wts` / `attn.q.bias.wts` | Q 投影权重与偏置 |
| `attn.k.weight.wts` / `attn.k.bias.wts` | K 投影权重与偏置 |
| `attn.v.weight.wts` / `attn.v.bias.wts` | V 投影权重与偏置 |
| `attn.o.weight.wts` / `attn.o.bias.wts` | 输出投影权重与偏置 |
| `mlp.fc1.weight.wts` / `mlp.fc1.bias.wts` | MLP 第一层权重与偏置 |
| `mlp.fc2.weight.wts` / `mlp.fc2.bias.wts` | MLP 第二层权重与偏置 |

---

### `main`

程序主入口，按以下顺序执行：

1. **加载图像**：调用 `load_image` 读取输入图片
2. **加载 PatchEmbedding 参数**：
   - `patch.weight.wts`：Patch 线性投影权重，形状 `{49, 32}`
   - `patch.bias.wts`：投影偏置，形状 `{1, 32}`
   - `cls_token.wts`：CLS token，reshape 为 `{1, 1, 32}`
   - `pos_embed.wts`：位置编码，reshape 为 `{1, 17, 32}`
3. **构建模型**：依次构造 `Linear`、`PatchEmbedding`、`VisionTransformer`
4. **注入 Transformer 块参数**：循环调用 `load_block` 加载 2 个编码器块
5. **加载分类头参数**：
   - `head.weight.wts`：形状 `{32, 10}`
   - `head.bias.wts`：形状 `{1, 10}`
6. **前向推理**：调用 `vit.forward(image)`，返回 logits 张量，形状 `{1, 10}`
7. **输出结果**：取 `argmax() % 10` 作为预测数字，并打印所有 logits 值

---

## 权重目录结构

运行时 `weight_dir` 下应包含以下文件：

```
weight_dir/
├── patch.weight.wts
├── patch.bias.wts
├── cls_token.wts
├── pos_embed.wts
├── blocks.0.attn.q.weight.wts
├── blocks.0.attn.q.bias.wts
├── blocks.0.attn.k.weight.wts
├── blocks.0.attn.k.bias.wts
├── blocks.0.attn.v.weight.wts
├── blocks.0.attn.v.bias.wts
├── blocks.0.attn.o.weight.wts
├── blocks.0.attn.o.bias.wts
├── blocks.0.mlp.fc1.weight.wts
├── blocks.0.mlp.fc1.bias.wts
├── blocks.0.mlp.fc2.weight.wts
├── blocks.0.mlp.fc2.bias.wts
├── blocks.1.attn.q.weight.wts
├── ... （blocks.1 同上）
├── head.weight.wts
└── head.bias.wts
```

---

## 输出示例

```
[INFO] Loading image...
[INFO] Loading PatchEmbedding weights...
[INFO] patch.weight.wts load successful
[INFO] patch.bias.wts load successful
[INFO] cls_token.wts load successful
[INFO] cls_token reshaped to {1, 1, 32}successful
[INFO] pos_embed.wts load successful
[INFO] pos_embed begin to reshape to {1, 17, 32}successful
[INFO] Building VisionTransformer...
[INFO] Loading 2 Transformer blocks...
[INFO] Loading classification head...
[INFO] Running forward inference...

=============================================
           MNIST Digit Recognition Result     
=============================================
Predicted digit: 7
Raw logits:     -1.2  0.3  -0.8  ...  2.1  
=============================================
[INFO] Inference completed!
```

---

## 图像输入约定

本项目所用输入图像均为**黑底白字**的手写数字图片（与 MNIST 数据集风格一致）。

在传入 `vit_infer.exe` 之前，需先使用 `convert_image.py` 将 PNG 图像转换为程序可读的裸二进制格式：

```bash
python convert_image.py
# 按提示输入：
# 请输入 PNG 文件路径: 大作业\1.png
# 请输入输出 RAW 文件路径: 大作业\1.raw
```

转换完成后再执行推理：

```bash
./vit_infer.exe 1.raw weights
```

`convert_image.py` 的转换逻辑如下：

```python
from PIL import Image
import numpy as np

def png_to_raw(input_png, output_raw):
    img = Image.open(input_png)
    img = img.convert("L")      # 转为灰度图
    img = img.resize((28, 28))  # 缩放至 28×28
    img_array = np.array(img, dtype=np.uint8)
    # 黑底白字无需反色，若为白底黑字请取消下一行注释
    # img_array = 255 - img_array
    img_array.tofile(output_raw)
    print(f"转换完成")

if __name__ == "__main__":
    input_png = input("请输入 PNG 文件路径: ")
    output_raw = input("请输入输出 RAW 文件路径: ")
    png_to_raw(input_png, output_raw)
```

**注意：**

- 输入图片必须为**黑底白字**；若为白底黑字，需在 `convert_image.py` 中启用 `img_array = 255 - img_array` 反色处理
- `convert_image.py` 会自动将图片缩放至 28×28 并转为灰度图，无需手动预处理
- 输出的 `.raw` 文件为 784 字节的裸二进制文件，不含任何文件头，直接按行优先顺序存储像素值

---

## 注意事项

- 所有推理日志输出到 `stderr`，便于与标准输出分离
- `load_tensor` 对数据量不匹配的情况只发出警告，不终止程序；调用方应确保权重文件与模型结构严格对应
- 本文件未实现 PDF 要求的 `export_attention` 功能（导出指定层指定头的 attention 矩阵），如需此功能可通过 `TransformerBlock` 内部的 `MultiHeadAttention::get_attention_map()` 接口扩展实现