#include "Tensor.h"
#include<vector>
#include<iostream>
#include<random>
using std::cerr;

class Layer//layer抽象基类
{
public:
    virtual Tensor<float> forward(const Tensor<float>& x) = 0;
    virtual ~Layer() = default;
};

class Linear : public Layer//linear线性层类
{
private:
    Tensor<float> _weight;
    Tensor<float> _bias;
    size_t in_features, out_features;
public:
    //做了两种接口
    Linear(size_t in_f = 0,size_t out_f = 0):in_features(in_f),out_features(out_f)//一种是直接传参数然后构造一套默认的仿射变换
    {
        _weight.he_normal_init(in_f,out_f);
        _bias.zero_init(in_f,out_f);
    }
    Linear(Tensor<float> _weight, Tensor<float> _bias):_weight(_weight),_bias(_bias){}//另一种是指定权重和偏移
    Tensor<float> forward(const Tensor<float>& x)
    {
        Tensor<float> output = x.matmul(_weight);
        return output.bias_add(_bias);
    }
};

/*
PatchEmbedding类通过将1\times 28\times 28的图像转换成16个7\times 7的Patch，映射到hidden_dim=32维向量上，以供下一步Transformer学习。
同时需要加一个CLS token用于学习。
输入[B, 28, 28],输出[1, patch_num=16 + CLS_num=1, hidden_dim=32]
*/
class PatchEmbedding : public Layer
{
private:
    static const int _patch_h = 7, _patch_w = 7;
    static const int _in_channels = 1;
    static const int _hidden_dim = 32;
    static const int NUM_PATCHES = 16;
    Linear projection;
    Tensor<float> cls_token;
    Tensor<float> pos_embedding;
public:
    PatchEmbedding()
        : projection(_in_channels * _patch_h * _patch_w, _hidden_dim),
          cls_token({1, 1, _hidden_dim}),
          pos_embedding({1, NUM_PATCHES + 1, _hidden_dim})
    {
        for (size_t h = 0; h < _hidden_dim; ++h)
        {
            cls_token(0, 0, h) = 0.0f;
        }
        for (size_t p = 0; p < NUM_PATCHES + 1; ++p)
        {
            for (size_t h = 0; h < _hidden_dim; ++h)
            {
                pos_embedding(0, p, h) = 0.0f;
            }
        }
    }
    Tensor<float> forward(const Tensor<float>& x)
    {
        
    }
};
