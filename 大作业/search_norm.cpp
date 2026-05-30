#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include "Model.h"
using namespace std;
Tensor<float> load_tensor(const string& filepath) {
    ifstream file(filepath);
    if (!file) throw runtime_error("open fail");
    vector<float> data;
    string line;
    int r=0,c=0;
    getline(file,line);
    int parsed = sscanf(line.c_str(),"# shape: %d %d",&r,&c);
    if(parsed==1){ c=r; r=1; }
    vector<size_t> shape={size_t(r), size_t(c)};
    Tensor<float> t(shape);
    while(getline(file,line)){
        istringstream iss(line);
        float v;
        while(iss>>v) data.push_back(v);
    }
    for(size_t i=0;i<min(data.size(),t.size());++i) t[i]=data[i];
    return t;
}
Tensor<float> load_raw_image(const string& path) {
    string raw_path = path + ".raw";
    ifstream file(raw_path, ios::binary);
    if(!file.is_open()) file.open(path, ios::binary);
    if(!file.is_open()) throw runtime_error("open fail");
    Tensor<float> img({1,28,28});
    unsigned char pix;
    for(int i=0;i<28;i++) for(int j=0;j<28;j++) { if(!file.read((char*)&pix,1)) throw runtime_error("read fail"); img(0,i,j)=float(pix)/255.0f; }
    return img;
}
TransformerBlock load_block(const string& wdir, int idx) {
    TransformerBlock block(32, 4, 64);
    string prefix = wdir + "/blocks." + to_string(idx);
    auto q_w = load_tensor(prefix + ".attn.q.weight.wts");
    auto q_b = load_tensor(prefix + ".attn.q.bias.wts");
    auto k_w = load_tensor(prefix + ".attn.k.weight.wts");
    auto k_b = load_tensor(prefix + ".attn.k.bias.wts");
    auto v_w = load_tensor(prefix + ".attn.v.weight.wts");
    auto v_b = load_tensor(prefix + ".attn.v.bias.wts");
    auto out_w = load_tensor(prefix + ".attn.o.weight.wts");
    auto out_b = load_tensor(prefix + ".attn.o.bias.wts");
    block.set_attention_weights(q_w,q_b,k_w,k_b,v_w,v_b,out_w,out_b);
    auto fc1_w = load_tensor(prefix + ".mlp.fc1.weight.wts");
    auto fc1_b = load_tensor(prefix + ".mlp.fc1.bias.wts");
    auto fc2_w = load_tensor(prefix + ".mlp.fc2.weight.wts");
    auto fc2_b = load_tensor(prefix + ".mlp.fc2.bias.wts");
    block.set_mlp_weights(fc1_w, fc1_b, fc2_w, fc2_b);
    auto g1 = load_tensor(prefix + ".norm1.gamma.wts");
    auto b1 = load_tensor(prefix + ".norm1.beta.wts");
    auto g2 = load_tensor(prefix + ".norm2.gamma.wts");
    auto b2 = load_tensor(prefix + ".norm2.beta.wts");
    block.set_norm_weights(g1,b1,g2,b2);
    return block;
}
int main(){ string weight_dir="weights"; string f1="mnist_1.raw", f4="mnist_4.raw";
    auto patch_w = load_tensor(weight_dir+"/patch.weight.wts");
    auto patch_b = load_tensor(weight_dir+"/patch.bias.wts");
    auto cls_token = load_tensor(weight_dir+"/cls_token.wts"); cls_token.reshape({1,1,32});
    auto pos_embed = load_tensor(weight_dir+"/pos_embed.wts"); pos_embed.reshape({1,17,32});
    Linear patch_proj(patch_w, patch_b);
    PatchEmbedding patch_embed(cls_token, pos_embed, patch_proj, 28,28,7,7);
    VisionTransformer vit(28,28,7,7,32,4,64,2,10);
    vit.set_patch_embedding(patch_embed);
    for(int i=0;i<2;i++){ vit.set_block(i, load_block(weight_dir, i)); }
    auto head_w = load_tensor(weight_dir+"/head.weight.wts");
    auto head_b = load_tensor(weight_dir+"/head.bias.wts");
    vit.set_head(head_w, head_b);
    auto raw1 = load_raw_image(f1);
    auto raw4 = load_raw_image(f4);
    for(float a=0.0f; a<=0.5f; a+=0.01f){ for(float b=0.2f; b<=1.0f; b+=0.02f){
        auto t1 = raw1;
        auto t4 = raw4;
        for(size_t i=0;i<t1.size();++i){ t1[i]=(t1[i]-a)/b; t4[i]=(t4[i]-a)/b; }
        int p1 = vit.forward(t1).argmax();
        int p4 = vit.forward(t4).argmax();
        if(p1==1 && p4==4){ cout<<"found a="<<a<<" b="<<b<<"\n"; return 0; }
    }}
    cout<<"none found\n";
}
