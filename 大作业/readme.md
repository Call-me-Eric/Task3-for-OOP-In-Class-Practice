# readme

## info

根据投票，我们选择了[作业2](/大作业/实验课+B+Tensor+and+ViT.pdf)作为大作业内容。

根据pdf，此大作业需要完成三个板块：

```text
Tensor.h,Model.h,main.cpp
```

目前分工：

Tensor.h 由 @linnnn 完成
Model.h 由 @Call\_me\_Eric 完成
main.cpp 由 @bianbiandaren 完成

## 使用方法：

通过g++ main.cpp -o main -O2 -std=c++20来编译可执行文件main

使用规则：

```cpp
./main <test_file_path> <weight_file_pait> <output_DEBUG>
```

example：

```cpp
./main 1.raw weights 0
```

将会尝试打开1.raw并预测，由于DEBUG参数是0，不会输出任何DEBUG信息，仅一行输出答案。

特别的，如果不输入DEBUG参数，默认参数为0。

如果不想手动手写文件，那么可以用/mnist_raw/tests中的数据。

不过，由于weights并没有非常nb的泛化，亲自手写可能会导致问题。
