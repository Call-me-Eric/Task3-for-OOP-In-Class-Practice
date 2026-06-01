#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <cstdlib>

#include "Model.h"
#include <filesystem>

using namespace std;

int debug;
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
        if(debug)cerr << "[WARN] " << filepath << ": expected " << tensor.size()
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
    // If input is a PNG, use an external Python helper to convert it to
    // a 28x28 grayscale raw byte file, then read the bytes.
    auto ends_with = [](const string &s, const string &suffix) {
        if (s.size() < suffix.size()) return false;
        return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    string lower = path;
    for (auto &c : lower) c = static_cast<char>(tolower((unsigned char)c));

    string raw_path = path + ".raw";
    if (ends_with(lower, ".png")) {
        // Call the Python converter in the same folder: convert_image.py
        string cmd = "python convert_image.py \"" + path + "\" \"" + raw_path + "\"";
        int ret = system(cmd.c_str());
        if (ret != 0) {
            throw runtime_error("Failed to convert PNG to raw bytes. Ensure Python and Pillow are installed.");
        }
    }

    ifstream file(raw_path, ios::binary);
    bool remove_raw_after = false;
    if (!file.is_open()) {
        // Fall back: try to open original path as raw bytes
        file.open(path, ios::binary);
        if (!file.is_open()) {
            throw runtime_error("Failed to open image file: " + path);
        }
    } else {
        remove_raw_after = true;
    }

    Tensor<float> img({1, 28, 28});
    unsigned char pix;
    std::vector<float> pixels;
    pixels.reserve(28*28);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (!file.read(reinterpret_cast<char *>(&pix), 1)) {
                file.close();
                if (remove_raw_after) std::remove(raw_path.c_str());
                throw runtime_error("Failed to read image pixels (expected 28x28 raw bytes)");
            }
            pixels.push_back(static_cast<float>(pix) / 255.0f);
        }
    }
    float mean = 0.0f;
    for (float v : pixels) mean += v;
    mean /= pixels.size();
    float var = 0.0f;
    for (float v : pixels) {
        float d = v - mean;
        var += d * d;
    }
    float std = std::sqrt(var / pixels.size() + 1e-9f);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            img(0, i, j) = (pixels[i * 28 + j] - mean) / std;
        }
    }
    if(debug)
    {
        cerr << "[DEBUG] Loaded image pixel values (normalized to [0,1]):" << endl;
        for(int i = 0;i < 28;i++)
        {
            for(int j = 0;j < 28;j++)
            {
                cerr << img(0, i, j) << " ";
            }
            cerr << endl;
        }
        cerr << "[DEBUG] Loaded image to {0,1}" << endl;
        for(int i = 0;i < 28;i++)
        {
            for(int j = 0;j < 28;j++)
            {
                cerr << (img(0, i, j) > 0) << " ";
            }
            cerr << endl;
        }
    }
    file.close();
    if (remove_raw_after) std::remove(raw_path.c_str());
    return img;
}

// 读取原始像素并返回 [0,1] 浮点张量（不做均值/标准化）
Tensor<float> load_raw_image(const string& path)
{
    string raw_path = path + ".raw";
    ifstream file(raw_path, ios::binary);
    if (!file.is_open()) {
        file.open(path, ios::binary);
        if (!file.is_open()) {
            throw runtime_error("Failed to open image file: " + path);
        }
    }
    Tensor<float> img({1, 28, 28});
    unsigned char pix;
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (!file.read(reinterpret_cast<char *>(&pix), 1)) {
                file.close();
                throw runtime_error("Failed to read image pixels (expected 28x28 raw bytes)");
            }
            img(0, i, j) = static_cast<float>(pix) / 255.0f;
        }
    }
    if(debug){
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                cerr << (img(0, i, j) > 0.5f) << " ";
            }
            cerr << endl;
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

    auto norm1_gamma = load_tensor(prefix + ".norm1.gamma.wts");
    auto norm1_beta  = load_tensor(prefix + ".norm1.beta.wts");
    auto norm2_gamma = load_tensor(prefix + ".norm2.gamma.wts");
    auto norm2_beta  = load_tensor(prefix + ".norm2.beta.wts");
    block.set_norm_weights(norm1_gamma, norm1_beta, norm2_gamma, norm2_beta);

    return block;
}

// ==============================
// 主函数
// ==============================
int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Usage: " << endl;
        cerr << "  " << argv[0] << " <image_path> [weight_dir] [0/1 debug]" << endl;
        cerr << "Examples:" << endl;
        cerr << "  " << argv[0] << " tests/test_0000_label_1.raw weights 0" << endl;
        return 1;
    }

    try {
        string img_path = argv[1];
        // 默认：使用微调后的权重目录
        string weight_dir = (argc >= 3 ? string(argv[2]) : string("weights_finetuned_norm"));
        int _debug = (argc >= 4 ? atoi(argv[3]) : 0); debug = _debug;

        const size_t INPUT_H     = 28;
        const size_t INPUT_W     = 28;
        const size_t PATCH_H     = 7;
        const size_t PATCH_W     = 7;
        const size_t HIDDEN_DIM  = 32;
        const size_t NUM_HEADS   = 4;
        const size_t MLP_DIM     = 64;
        const size_t NUM_LAYERS  = 2;
        const size_t NUM_CLASSES = 10;

        if(debug)cerr << "[INFO] Loading image..." << endl;
        Tensor<float> raw = load_raw_image(img_path);
        Tensor<float> img_std = raw;
        for (size_t i = 0; i < img_std.size(); ++i) {
            img_std[i] = (img_std[i] - 0.1307f) / 0.3081f;
        }

        // 现在加载权重并构建模型
        if(debug)cerr << "[INFO] Loading PatchEmbedding weights..." << endl;
        auto patch_w = load_tensor(weight_dir + "/patch.weight.wts");
        auto patch_b = load_tensor(weight_dir + "/patch.bias.wts");

        auto cls_token = load_tensor(weight_dir + "/cls_token.wts");
        cls_token.reshape({1, 1, HIDDEN_DIM});

        auto pos_embed = load_tensor(weight_dir + "/pos_embed.wts");
        pos_embed.reshape({1, (INPUT_H / PATCH_H) * (INPUT_W / PATCH_W) + 1, HIDDEN_DIM});

        Linear patch_proj(patch_w, patch_b);
        PatchEmbedding patch_embed(cls_token, pos_embed, patch_proj,
                                   INPUT_H, INPUT_W, PATCH_H, PATCH_W);

        VisionTransformer vit(INPUT_H, INPUT_W, PATCH_H, PATCH_W,
                              HIDDEN_DIM, NUM_HEADS, MLP_DIM,
                              NUM_LAYERS, NUM_CLASSES);
        vit.set_patch_embedding(patch_embed);

        for (size_t i = 0; i < NUM_LAYERS; ++i) {
            TransformerBlock block = load_block(weight_dir, static_cast<int>(i),
                                                HIDDEN_DIM, NUM_HEADS, MLP_DIM);
            vit.set_block(i, block);
        }

        auto head_w = load_tensor(weight_dir + "/head.weight.wts");
        auto head_b = load_tensor(weight_dir + "/head.bias.wts");
        vit.set_head(head_w, head_b);

        // also compute center preprocess and use confidence-based selection
        Tensor<float> img_center = raw;
        for (size_t i = 0; i < img_center.size(); ++i) {
            img_center[i] = img_center[i] - 0.5f;
        }

        Tensor<float> logits_std = vit.forward(img_std.reshaped({1, 28, 28}));
        Tensor<float> logits_ctr = vit.forward(img_center.reshaped({1, 28, 28}));

        int pred_std = static_cast<int>(logits_std.argmax() % 10);
        int pred_ctr = static_cast<int>(logits_ctr.argmax() % 10);
        int predict = pred_std;
        if (pred_std == pred_ctr) {
            predict = pred_std;
        } else {
            // compute softmax max confidence for each
            Tensor<float> prob_std = logits_std.softmax(logits_std.rank() - 1);
            Tensor<float> prob_ctr = logits_ctr.softmax(logits_ctr.rank() - 1);
            float max_std = 0.0f, max_ctr = 0.0f;
            // logits are shape {10} (flattened)
            for (size_t i = 0; i < 10; ++i) {
                if (prob_std[i] > max_std) max_std = prob_std[i];
                if (prob_ctr[i] > max_ctr) max_ctr = prob_ctr[i];
            }
            if (max_ctr > max_std) predict = pred_ctr;
            else predict = pred_std;
        }

        if(debug)cerr << "\n=============================================" << endl;
        if(debug)cerr << "           MNIST Digit Recognition Result     " << endl;
        if(debug)cerr << "=============================================" << endl;
        if(debug)cerr << "Predicted digit: " << predict << endl;
        if(debug)cerr << "Raw logits:     ";
        Tensor<float> logits_final = (predict == pred_std) ? logits_std : logits_ctr;
        for (size_t i = 0; i < 10; ++i) {
            if(debug)cerr << logits_final[i] << "  ";
        }
        if(debug)cerr << "\n=============================================" << endl;
        if(debug)cerr << "[INFO] Inference completed!" << endl;
        cout << predict << endl;

    } catch (const exception& e) {
        cerr << "\n[ERROR] Exception occurred:" << endl;
        cerr << "  " << e.what() << endl;
        return 1;
    }

    return 0;
}