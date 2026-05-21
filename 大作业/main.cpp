#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

// #include "Tensor.h"
#include "Model.h"

using namespace std;

// ==============================
// 工具函数：加载二进制张量文件
// ==============================
Tensor<float> load_tensor(const string& filepath)
{
    ifstream file(filepath, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("无法打开权重文件: " + filepath);
    }

    // 读取维度数
    size_t rank;
    if (!file.read(reinterpret_cast<char*>(&rank), sizeof(rank))) {
        throw runtime_error("读取张量维度失败: " + filepath);
    }

    // 读取形状
    vector<size_t> shape(rank);
    if (!file.read(reinterpret_cast<char*>(shape.data()), rank * sizeof(size_t))) {
        throw runtime_error("读取张量形状失败: " + filepath);
    }

    // 读取数据
    Tensor<float> tensor(shape);
    if (tensor.size() > 0) {
        if (!file.read(reinterpret_cast<char*>(&tensor[0]), tensor.size() * sizeof(float))) {
            throw runtime_error("读取张量数据失败: " + filepath);
        }
    }

    file.close();
    return tensor;
}

// ==============================
// 加载 MNIST 28x28 灰度图像
// ==============================
Tensor<float> load_image(const string& path)
{
    ifstream file(path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("无法打开图像文件: " + path);
    }

    Tensor<float> img({1, 28, 28});
    unsigned char pix;

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (!file.read(reinterpret_cast<char*>(&pix), 1)) {
                throw runtime_error("图像像素读取异常");
            }
            img(0, i, j) = static_cast<float>(pix) / 255.0f;
        }
    }

    file.close();
    return img;
}

// ==============================
// 加载单个 Transformer Block
// ==============================
TransformerBlock load_block(const string& wdir, int idx,
                            size_t hidden_dim, size_t num_heads, size_t mlp_dim)
{
    TransformerBlock block(hidden_dim, num_heads, mlp_dim);
    string prefix = wdir + "/block" + to_string(idx);

    // 注意力层权重
    auto q_w = load_tensor(prefix + "_q_weight.bin");
    auto q_b = load_tensor(prefix + "_q_bias.bin");
    auto k_w = load_tensor(prefix + "_k_weight.bin");
    auto k_b = load_tensor(prefix + "_k_bias.bin");
    auto v_w = load_tensor(prefix + "_v_weight.bin");
    auto v_b = load_tensor(prefix + "_v_bias.bin");
    auto out_w = load_tensor(prefix + "_out_weight.bin");
    auto out_b = load_tensor(prefix + "_out_bias.bin");

    block.set_attention_weights(q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b);

    // MLP 层权重
    auto fc1_w = load_tensor(prefix + "_fc1_weight.bin");
    auto fc1_b = load_tensor(prefix + "_fc1_bias.bin");
    auto fc2_w = load_tensor(prefix + "_fc2_weight.bin");
    auto fc2_b = load_tensor(prefix + "_fc2_bias.bin");

    block.set_mlp_weights(fc1_w, fc1_b, fc2_w, fc2_b);

    return block;
}

// ==============================
// 主函数：ViT 推理入口
// ==============================
int main(int argc, char* argv[])
{
    if (argc < 3) {
        cerr << "Usage: " << endl;
        cerr << "  " << argv[0] << " <image_path> <weight_dir>" << endl;
        return 1;
    }

    try {
        string img_path  = argv[1];
        string weight_dir = argv[2];

        // ==============================
        // 实验指定超参数（完全按要求）
        // ==============================
        const size_t INPUT_H     = 28;
        const size_t INPUT_W     = 28;
        const size_t PATCH_H     = 7;
        const size_t PATCH_W     = 7;
        const size_t HIDDEN_DIM  = 32;
        const size_t NUM_HEADS   = 2;
        const size_t MLP_DIM     = 64;
        const size_t NUM_LAYERS  = 2;
        const size_t NUM_CLASSES = 10;

        cout << "[INFO] 开始加载图像..." << endl;
        Tensor<float> image = load_image(img_path);

        cout << "[INFO] 加载 PatchEmbedding 权重..." << endl;
        auto patch_w    = load_tensor(weight_dir + "/patch_weight.bin");
        auto patch_b    = load_tensor(weight_dir + "/patch_bias.bin");
        auto cls_token  = load_tensor(weight_dir + "/cls_token.bin");
        auto pos_embed  = load_tensor(weight_dir + "/pos_embed.bin");

        Linear patch_proj(patch_w, patch_b);
        PatchEmbedding patch_embed(cls_token, pos_embed, patch_proj,
                                   INPUT_H, INPUT_W, PATCH_H, PATCH_W);

        cout << "[INFO] 构建 VisionTransformer..." << endl;
        VisionTransformer vit(INPUT_H, INPUT_W, PATCH_H, PATCH_W,
                              HIDDEN_DIM, NUM_HEADS, MLP_DIM,
                              NUM_LAYERS, NUM_CLASSES);
        vit.set_patch_embedding(patch_embed);

        cout << "[INFO] 加载 " << NUM_LAYERS <<  " 层 Transformer Block..." << endl;
        for (size_t i = 0; i < NUM_LAYERS; ++i) {
            TransformerBlock block = load_block(weight_dir, static_cast<int>(i),
                                                HIDDEN_DIM, NUM_HEADS, MLP_DIM);
            vit.set_block(i, block);
        }

        cout << "[INFO] 加载分类头权重..." << endl;
        auto head_w = load_tensor(weight_dir + "/head_weight.bin");
        auto head_b = load_tensor(weight_dir + "/head_bias.bin");
        vit.set_head(head_w, head_b);

        // ==============================
        // 前向推理
        // ==============================
        cout << "[INFO] 模型前向推理中..." << endl;
        Tensor<float> logits = vit.forward(image);
        int predict = static_cast<int>(logits.argmax() % 10);

        // ==============================
        // 输出结果
        // ==============================
        cout << "\n=============================================" << endl;
        cout << "           MNIST 手写数字识别结果            " << endl;
        cout << "=============================================" << endl;
        cout << "预测数字： " << predict << endl;
        cout << "原始 logits：";
        for (size_t i = 0; i < 10; ++i) {
            cout << logits[i] << "  ";
        }
        cout << "\n=============================================" << endl;
        cout << "[INFO] 推理完成！" << endl;

    } catch (const exception& e) {
        cerr << "\n[ERROR] 程序异常：" << endl;
        cerr << "  " << e.what() << endl;
        return 1;
    }

    return 0;
}