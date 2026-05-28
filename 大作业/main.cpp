#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <algorithm>

#include "Model.h"

using namespace std;

Tensor<float> load_tensor(const string& filepath)
{
    ifstream file(filepath);
    if (!file.is_open()) {
        throw runtime_error("Failed to open weight file: " + filepath);
    }

    vector<float> data;
    string line;
    int rows = 0, cols = 0;

    // 读取 shape
    getline(file, line);
    int parsed = sscanf(line.c_str(), "# shape: %d %d", &rows, &cols);

    if (parsed == 1 && rows > 0) {
        cols = rows;  // 1D bias，当作 (1, N)
        rows = 1;
    } else if (parsed != 2 || rows <= 0 || cols <= 0) {
        throw runtime_error("Invalid shape in file: " + filepath +
                            " (line=[" + line + "])");
    }

    // 读取所有数据
    while (getline(file, line)) {
        istringstream iss(line);
        float val;
        while (iss >> val) {
            data.push_back(val);
        }
    }

    file.close();

    vector<size_t> shape = {(size_t)rows, (size_t)cols};
    Tensor<float> tensor(shape);

    if (data.size() != tensor.size()) {
        cerr << "[WARN] " << filepath << ": expected " << tensor.size()
             << " values but got " << data.size() << endl;
    }
    size_t n = min(data.size(), tensor.size());
    for (size_t i = 0; i < n; ++i) {
        tensor[i] = data[i];
    }

    return tensor;
}

// ==============================
// 读取 MNIST 28x28 图像
// ==============================
Tensor<float> load_image(const string& path)
{
    ifstream file(path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Failed to open image file: " + path);
    }

    Tensor<float> img({1, 28, 28});
    unsigned char pix;

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (!file.read(reinterpret_cast<char*>(&pix), 1)) {
                throw runtime_error("Failed to read image pixels");
            }
            img(0, i, j) = static_cast<float>(pix) / 255.0f;
        }
    }

    file.close();
    return img;
}

// ==============================
// 加载 Transformer Block
// ==============================
TransformerBlock load_block(const string& wdir, int idx,
                            size_t hidden_dim, size_t num_heads, size_t mlp_dim)
{
    TransformerBlock block(hidden_dim, num_heads, mlp_dim);
    string prefix = wdir + "/blocks." + to_string(idx);

    auto q_w = load_tensor(prefix + ".attn.q.weight.wts");
    auto q_b = load_tensor(prefix + ".attn.q.bias.wts");
    auto k_w = load_tensor(prefix + ".attn.k.weight.wts");
    auto k_b = load_tensor(prefix + ".attn.k.bias.wts");
    auto v_w = load_tensor(prefix + ".attn.v.weight.wts");
    auto v_b = load_tensor(prefix + ".attn.v.bias.wts");
    auto out_w = load_tensor(prefix + ".attn.o.weight.wts");
    auto out_b = load_tensor(prefix + ".attn.o.bias.wts");

    block.set_attention_weights(q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b);

    auto fc1_w = load_tensor(prefix + ".mlp.fc1.weight.wts");
    auto fc1_b = load_tensor(prefix + ".mlp.fc1.bias.wts");
    auto fc2_w = load_tensor(prefix + ".mlp.fc2.weight.wts");
    auto fc2_b = load_tensor(prefix + ".mlp.fc2.bias.wts");

    block.set_mlp_weights(fc1_w, fc1_b, fc2_w, fc2_b);

    return block;
}

// ==============================
// 主函数
// ==============================
int main(int argc, char* argv[])
{
    if (argc < 3) {
        cerr << "Usage: " << endl;
        cerr << "  " << argv[0] << " <image_path> <weight_dir>" << endl;
        return 1;
    }

    try {
        string img_path = argv[1];
        string weight_dir = argv[2];

        const size_t INPUT_H     = 28;
        const size_t INPUT_W     = 28;
        const size_t PATCH_H     = 7;
        const size_t PATCH_W     = 7;
        const size_t HIDDEN_DIM  = 32;
        const size_t NUM_HEADS   = 2;
        const size_t MLP_DIM     = 64;
        const size_t NUM_LAYERS  = 2;
        const size_t NUM_CLASSES = 10;

        cerr << "[INFO] Loading image..." << endl;
        Tensor<float> image = load_image(img_path);

        cerr << "[INFO] Loading PatchEmbedding weights..." << endl;
        
        auto patch_w = load_tensor(weight_dir + "/patch.weight.wts");
        cerr << "[INFO] patch.weight.wts load successful" << endl;
        auto patch_b = load_tensor(weight_dir + "/patch.bias.wts");
        cerr << "[INFO] patch.bias.wts load successful" << endl;

        auto cls_token = load_tensor(weight_dir + "/cls_token.wts");
        cerr << "[INFO] cls_token.wts load successful" << endl;
        cerr << "[INFO] cls_token reshaped to {1, 1, " << HIDDEN_DIM << "}";
        cls_token.reshape({1, 1, HIDDEN_DIM});
        cerr << "successful" << endl;

        auto pos_embed = load_tensor(weight_dir + "/pos_embed.wts");
        cerr << "[INFO] pos_embed.wts load successful" << endl;
        cerr << "[INFO] pos_embed begin to reshape to {1, " << (INPUT_H / PATCH_H) * (INPUT_W / PATCH_W) + 1 << ", " << HIDDEN_DIM << "}";
        pos_embed.reshape({1, (INPUT_H / PATCH_H) * (INPUT_W / PATCH_W) + 1, HIDDEN_DIM});
        cerr << "successful" << endl;

        Linear patch_proj(patch_w, patch_b);
        PatchEmbedding patch_embed(cls_token, pos_embed, patch_proj,
                                   INPUT_H, INPUT_W, PATCH_H, PATCH_W);

        cerr << "[INFO] Building VisionTransformer..." << endl;
        VisionTransformer vit(INPUT_H, INPUT_W, PATCH_H, PATCH_W,
                              HIDDEN_DIM, NUM_HEADS, MLP_DIM,
                              NUM_LAYERS, NUM_CLASSES);
        vit.set_patch_embedding(patch_embed);

        cerr << "[INFO] Loading " << NUM_LAYERS << " Transformer blocks..." << endl;
        for (size_t i = 0; i < NUM_LAYERS; ++i) {
            TransformerBlock block = load_block(weight_dir, static_cast<int>(i),
                                                HIDDEN_DIM, NUM_HEADS, MLP_DIM);
            vit.set_block(i, block);
        }

        cerr << "[INFO] Loading classification head..." << endl;
        auto head_w = load_tensor(weight_dir + "/head.weight.wts");
        auto head_b = load_tensor(weight_dir + "/head.bias.wts");
        vit.set_head(head_w, head_b);

        cerr << "[INFO] Running forward inference..." << endl;
        Tensor<float> logits = vit.forward(image);
        int predict = static_cast<int>(logits.argmax() % 10);

        cerr << "\n=============================================" << endl;
        cerr << "           MNIST Digit Recognition Result     " << endl;
        cerr << "=============================================" << endl;
        cerr << "Predicted digit: " << predict << endl;
        cerr << "Raw logits:     ";
        for (size_t i = 0; i < 10; ++i) {
            cerr << logits[i] << "  ";
        }
        cerr << "\n=============================================" << endl;
        cerr << "[INFO] Inference completed!" << endl;

    } catch (const exception& e) {
        cerr << "\n[ERROR] Exception occurred:" << endl;
        cerr << "  " << e.what() << endl;
        return 1;
    }

    return 0;
}