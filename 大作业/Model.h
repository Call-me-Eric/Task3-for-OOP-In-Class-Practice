#include "Tensor.h"
#include<vector>
#include<iostream>
#include<random>
#include<cmath>
using std::cerr;

// 所有权重和偏置参数由外部文件加载并注入。
// 这里的类实现只保留模型结构和超参数接口。
class Layer//layer抽象基类
{
public:
    virtual Tensor<float> forward(const Tensor<float>& x) = 0;
    virtual ~Layer() = default;
};
// template <class T>
// Tensor<T> Tensor<T>::bias_add(const Tensor<T> &other)const
// {
//     Tensor<T> ans = *this;
//     for(int i = 0;i < ans._data.size();i++)ans._data[i] += other[index_to_coordinates(i).back()];
//     return ans;
// }

class Linear : public Layer//linear线性层类
{
private:
    Tensor<float> _weight;
    Tensor<float> _bias;
    size_t in_features;
    size_t out_features;
public:
    Linear(size_t in_f = 0, size_t out_f = 0)
        : in_features(in_f), out_features(out_f), _weight({in_f, out_f}), _bias({1, out_f}) {}

    Linear(const Tensor<float>& weight, const Tensor<float>& bias)
        : _weight(weight), _bias(bias), in_features(weight.shape()[0]), out_features(weight.shape()[1]) {}

    void set_weights(const Tensor<float>& weight, const Tensor<float>& bias) {
        _weight = weight;
        _bias = bias;
        in_features = weight.shape()[0];
        out_features = weight.shape()[1];
    }

    const Tensor<float>& weight() const { return _weight; }
    const Tensor<float>& bias() const { return _bias; }

    Tensor<float> forward(const Tensor<float>& x) override
    {
        Tensor<float> output = x.matmul(_weight);
        return output.bias_add(_bias);
    }
};

/*
PatchEmbedding类通过将输入图像切成Patch并将每个Patch映射到hidden_dim维向量上，以供后续Transformer使用。
CLS token、位置编码和线性投影参数均通过外部配置提供。
输入形状为 {B, input_h, input_w}，输出形状为 {B, 1 + num_patches, hidden_dim}。
*/

class PatchEmbedding : public Layer
{
private:
    size_t _input_h;
    size_t _input_w;
    size_t _patch_h;
    size_t _patch_w;
    size_t _hidden_dim;
    size_t _num_patches;
    Tensor<float> cls_token;
    Tensor<float> pos_embedding;
    Linear linear;
public:
    PatchEmbedding(size_t input_h = 28, size_t input_w = 28,
                   size_t patch_h = 7, size_t patch_w = 7,
                   size_t hidden_dim = 32)
        : _input_h(input_h), _input_w(input_w), _patch_h(patch_h), _patch_w(patch_w),
          _hidden_dim(hidden_dim),
          _num_patches((input_h / patch_h) * (input_w / patch_w)),
          cls_token({1, 1, hidden_dim}),
          pos_embedding({1, _num_patches + 1, hidden_dim}),
          linear(patch_h * patch_w, hidden_dim)
    {
        if (input_h % patch_h != 0 || input_w % patch_w != 0) {
            throw std::invalid_argument("PatchEmbedding input size must be divisible by patch size");
        }
    }

    PatchEmbedding(const Tensor<float>& cls_token,
                   const Tensor<float>& pos_embedding,
                   const Linear& linear,
                   size_t input_h = 28,
                   size_t input_w = 28,
                   size_t patch_h = 7,
                   size_t patch_w = 7)
        : _input_h(input_h), _input_w(input_w), _patch_h(patch_h), _patch_w(patch_w),
          _hidden_dim(cls_token.shape()[2]),
          _num_patches(pos_embedding.shape()[1] - 1),
          cls_token(cls_token), pos_embedding(pos_embedding), linear(linear)
    {}

    void set_cls_token(const Tensor<float>& token) {
        cls_token = token;
    }

    void set_pos_embedding(const Tensor<float>& embedding) {
        pos_embedding = embedding;
        if (embedding.rank() >= 2) {
            _num_patches = embedding.shape()[1] - 1;
        }
    }

    void set_projection(const Linear& proj) {
        linear = proj;
    }

    Tensor<float> forward(const Tensor<float>& x) override
    {
        auto x_shape = x.shape();
        if (x_shape.size() != 3 || x_shape[1] != _input_h || x_shape[2] != _input_w) {
            throw std::invalid_argument("PatchEmbedding input shape must be {B, input_h, input_w}");
        }

        size_t B = x_shape[0];
        Tensor<float> output({B, 1, _hidden_dim});
        for (size_t b = 0; b < B; ++b) {
            for (size_t i = 0; i < _hidden_dim; ++i) {
                output(b, 0, i) = cls_token(0, 0, i);
            }
        }

        size_t patch_index = 1;
        for (size_t i = 0; i < _input_h; i += _patch_h) {
            for (size_t j = 0; j < _input_w; j += _patch_w) {
                auto patch = x.slice(1, i, i + _patch_h).slice(2, j, j + _patch_w);
                Tensor<float> patch_flat = patch.reshaped({B, _patch_h * _patch_w});
                Tensor<float> token = linear.forward(patch_flat);
                token.reshape({B, 1, _hidden_dim});
                Tensor<float> pos = pos_embedding.slice(1, patch_index, patch_index + 1);
                token = token + pos;
                output = Tensor<float>::concat({output, token}, 1);
                ++patch_index;
            }
        }
        return output;
    }
};

class LayerNorm : public Layer {
public:
    Tensor<float> forward(const Tensor<float>& x) override {
        Tensor<float> out = x;
        size_t last_dim = x.shape().back();
        size_t outer = x.size() / last_dim;
        
        for (size_t o = 0; o < outer; ++o) {
            // 计算均值
            float mean = 0.0f;
            for (size_t i = 0; i < last_dim; ++i)
                mean += out[o * last_dim + i];
            mean /= last_dim;
            
            // 计算方差
            float var = 0.0f;
            for (size_t i = 0; i < last_dim; ++i) {
                float diff = out[o * last_dim + i] - mean;
                var += diff * diff;
            }
            var /= last_dim;
            
            // 归一化
            float std = std::sqrt(var + 1e-5f);
            for (size_t i = 0; i < last_dim; ++i)
                out[o * last_dim + i] = (out[o * last_dim + i] - mean) / std;
        }
        return out;
    }
};
class MultiHeadAttention {
private:
    size_t _hidden_dim;
    size_t _num_heads;
    size_t _head_dim;
    
    Linear _q_proj;
    Linear _k_proj;
    Linear _v_proj;
    Linear _out_proj;
    
    Tensor<float> _last_attention;

public:
    MultiHeadAttention(size_t hidden_dim, size_t num_heads)
        : _hidden_dim(hidden_dim), _num_heads(num_heads), _head_dim(hidden_dim / num_heads),
          _q_proj(hidden_dim, hidden_dim), _k_proj(hidden_dim, hidden_dim),
          _v_proj(hidden_dim, hidden_dim), _out_proj(hidden_dim, hidden_dim)
    {
        if (hidden_dim % num_heads != 0) {
            throw std::invalid_argument("hidden_dim must be divisible by num_heads");
        }
    }

    void set_q_proj(const Tensor<float>& weight, const Tensor<float>& bias) {
        _q_proj.set_weights(weight, bias);
    }
    void set_k_proj(const Tensor<float>& weight, const Tensor<float>& bias) {
        _k_proj.set_weights(weight, bias);
    }
    void set_v_proj(const Tensor<float>& weight, const Tensor<float>& bias) {
        _v_proj.set_weights(weight, bias);
    }
    void set_out_proj(const Tensor<float>& weight, const Tensor<float>& bias) {
        _out_proj.set_weights(weight, bias);
    }

    Tensor<float> forward(const Tensor<float>& x) {
        auto input_shape = x.shape();
        size_t batch_size = input_shape[0];
        size_t seq_len = input_shape[1];
        
        Tensor<float> Q = _q_proj.forward(x);
        Tensor<float> K = _k_proj.forward(x);
        Tensor<float> V = _v_proj.forward(x);
        
        auto q_reshaped = Q.reshaped({batch_size, seq_len, _num_heads, _head_dim});
        auto k_reshaped = K.reshaped({batch_size, seq_len, _num_heads, _head_dim});
        auto v_reshaped = V.reshaped({batch_size, seq_len, _num_heads, _head_dim});
        
        auto q_permuted = q_reshaped.permute({0, 2, 1, 3});
        auto k_permuted = k_reshaped.permute({0, 2, 1, 3});
        auto v_permuted = v_reshaped.permute({0, 2, 1, 3});
        
        auto k_transposed = k_permuted.permute({0, 1, 3, 2});
        
        Tensor<float> scores = q_permuted.matmul(k_transposed);
        float scale_factor = std::sqrt(static_cast<float>(_head_dim));
        scores = scores / scale_factor;
        
        Tensor<float> attention_weights = scores.softmax(scores.rank() - 1);
        Tensor<float> context = attention_weights.matmul(v_permuted);
        _last_attention = attention_weights;
        
        auto context_transposed = context.permute({0, 2, 1, 3});
        auto output = context_transposed.reshaped({batch_size, seq_len, _hidden_dim});
        output = _out_proj.forward(output);
        return output;
    }
    
    Tensor<float> get_attention_map() const {
        return _last_attention;
    }
};

class MLP : public Layer {
private:
    Linear _fc1;
    Linear _fc2;

    Tensor<float> gelu(const Tensor<float>& x) const {
        Tensor<float> out = x;
        for (size_t i = 0; i < out.size(); ++i) {
            float v = out[i];
            float c = std::sqrt(2.0f / 3.1415926f);
            float x3 = v * v * v;
            out[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * x3)));
        }
        return out;
    }

public:
    MLP(size_t hidden_dim = 32, size_t mlp_dim = 64)
        : _fc1(hidden_dim, mlp_dim), _fc2(mlp_dim, hidden_dim) {}

    void set_fc1(const Tensor<float>& weight, const Tensor<float>& bias) {
        _fc1.set_weights(weight, bias);
    }
    void set_fc2(const Tensor<float>& weight, const Tensor<float>& bias) {
        _fc2.set_weights(weight, bias);
    }

    Tensor<float> forward(const Tensor<float>& x) override {
        Tensor<float> hidden = _fc1.forward(x);
        hidden = gelu(hidden);
        return _fc2.forward(hidden);
    }
};

class TransformerBlock : public Layer {
private:
    LayerNorm _norm1;
    LayerNorm _norm2;
    MultiHeadAttention _mha;
    MLP _mlp;

public:
    TransformerBlock(size_t hidden_dim = 32, size_t num_heads = 4, size_t mlp_dim = 64)
        : _norm1(), _norm2(), _mha(hidden_dim, num_heads), _mlp(hidden_dim, mlp_dim) {}

    void set_attention_weights(const Tensor<float>& q_weight, const Tensor<float>& q_bias,
                               const Tensor<float>& k_weight, const Tensor<float>& k_bias,
                               const Tensor<float>& v_weight, const Tensor<float>& v_bias,
                               const Tensor<float>& out_weight, const Tensor<float>& out_bias) {
        _mha.set_q_proj(q_weight, q_bias);
        _mha.set_k_proj(k_weight, k_bias);
        _mha.set_v_proj(v_weight, v_bias);
        _mha.set_out_proj(out_weight, out_bias);
    }

    void set_mlp_weights(const Tensor<float>& fc1_weight, const Tensor<float>& fc1_bias,
                         const Tensor<float>& fc2_weight, const Tensor<float>& fc2_bias) {
        _mlp.set_fc1(fc1_weight, fc1_bias);
        _mlp.set_fc2(fc2_weight, fc2_bias);
    }

    Tensor<float> forward(const Tensor<float>& x) override {
        Tensor<float> attn_input = _norm1.forward(x);
        Tensor<float> attn_output = _mha.forward(attn_input);
        Tensor<float> residual1 = x + attn_output;

        Tensor<float> mlp_input = _norm2.forward(residual1);
        Tensor<float> mlp_output = _mlp.forward(mlp_input);
        return residual1 + mlp_output;
    }
};

class VisionTransformer : public Layer {
private:
    size_t _input_h;
    size_t _input_w;
    size_t _patch_h;
    size_t _patch_w;
    size_t _hidden_dim;
    size_t _num_heads;
    size_t _mlp_dim;
    size_t _num_layers;
    size_t _num_classes;

    PatchEmbedding _patch_embed;
    std::vector<TransformerBlock> _blocks;
    LayerNorm _norm;
    Linear _head;

public:
    VisionTransformer(size_t input_h = 28,
                      size_t input_w = 28,
                      size_t patch_h = 7,
                      size_t patch_w = 7,
                      size_t hidden_dim = 32,
                      size_t num_heads = 4,
                      size_t mlp_dim = 64,
                      size_t num_layers = 1,
                      size_t num_classes = 10)
        : _input_h(input_h), _input_w(input_w), _patch_h(patch_h), _patch_w(patch_w),
          _hidden_dim(hidden_dim), _num_heads(num_heads), _mlp_dim(mlp_dim),
          _num_layers(num_layers), _num_classes(num_classes),
          _patch_embed(input_h, input_w, patch_h, patch_w, hidden_dim),
          _head(hidden_dim, num_classes)
    {
        if (hidden_dim % num_heads != 0) {
            throw std::invalid_argument("hidden_dim must be divisible by num_heads");
        }
        for (size_t i = 0; i < num_layers; ++i) {
            _blocks.emplace_back(hidden_dim, num_heads, mlp_dim);
        }
    }

    void set_patch_embedding(const PatchEmbedding& patch_embed) {
        _patch_embed = patch_embed;
    }

    void set_head(const Tensor<float>& weight, const Tensor<float>& bias) {
        _head.set_weights(weight, bias);
    }

    void set_block(size_t index, const TransformerBlock& block) {
        if (index >= _blocks.size()) {
            throw std::out_of_range("Transformer block index out of range");
        }
        _blocks[index] = block;
    }

    Tensor<float> forward(const Tensor<float>& x) override {
        Tensor<float> tokens = _patch_embed.forward(x);
        for (auto& block : _blocks) {
            tokens = block.forward(tokens);
        }
        tokens = _norm.forward(tokens);

        size_t batch_size = tokens.shape()[0];
        Tensor<float> cls_token = tokens.slice(1, 0, 1);
        cls_token = cls_token.reshaped({batch_size, _hidden_dim});
        return _head.forward(cls_token);
    }
};