#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include "Model.h"
using namespace std;
Tensor<float> load_tensor(const string& filepath) {
    ifstream file(filepath);
    if(!file) throw runtime_error("open fail");
    vector<float> data; string line; int r=0,c=0;
    getline(file,line); int parsed=sscanf(line.c_str(),"# shape: %d %d",&r,&c);
    if(parsed==1){ c=r; r=1; }
    Tensor<float> t({(size_t)r,(size_t)c});
    while(getline(file,line)){ istringstream iss(line); float v; while(iss>>v) data.push_back(v); }
    for(size_t i=0;i<min(data.size(),t.size());++i) t[i]=data[i]; return t;
}
Tensor<float> load_raw(const string& path){ string raw=path+".raw"; ifstream file(raw,ios::binary); if(!file.is_open()) file.open(path,ios::binary); if(!file) throw runtime_error("open fail"); Tensor<float> t({1,28,28}); unsigned char pix; for(int i=0;i<28;i++) for(int j=0;j<28;j++){ if(!file.read((char*)&pix,1)) throw runtime_error("read fail"); t(0,i,j)=float(pix)/255.0f; } return t; }
TransformerBlock load_block(const string& wdir, int idx){ TransformerBlock block(32,4,64); string prefix=wdir+"/blocks."+to_string(idx); auto q_w=load_tensor(prefix+".attn.q.weight.wts"); auto q_b=load_tensor(prefix+".attn.q.bias.wts"); auto k_w=load_tensor(prefix+".attn.k.weight.wts"); auto k_b=load_tensor(prefix+".attn.k.bias.wts"); auto v_w=load_tensor(prefix+".attn.v.weight.wts"); auto v_b=load_tensor(prefix+".attn.v.bias.wts"); auto o_w=load_tensor(prefix+".attn.o.weight.wts"); auto o_b=load_tensor(prefix+".attn.o.bias.wts"); block.set_attention_weights(q_w,q_b,k_w,k_b,v_w,v_b,o_w,o_b); auto f1=load_tensor(prefix+".mlp.fc1.weight.wts"); auto b1=load_tensor(prefix+".mlp.fc1.bias.wts"); auto f2=load_tensor(prefix+".mlp.fc2.weight.wts"); auto b2=load_tensor(prefix+".mlp.fc2.bias.wts"); block.set_mlp_weights(f1,b1,f2,b2); auto g1=load_tensor(prefix+".norm1.gamma.wts"); auto bb1=load_tensor(prefix+".norm1.beta.wts"); auto g2=load_tensor(prefix+".norm2.gamma.wts"); auto bb2=load_tensor(prefix+".norm2.beta.wts"); block.set_norm_weights(g1,bb1,g2,bb2); return block; }
int main(){ vector<string> files={"mnist_0.raw","mnist_1.raw","mnist_2.raw","mnist_4.raw","mnist_5.raw","mnist_7.raw","mnist_9.raw"}; string wd="weights"; auto patch_w=load_tensor(wd+"/patch.weight.wts"); auto patch_b=load_tensor(wd+"/patch.bias.wts"); auto cls_token=load_tensor(wd+"/cls_token.wts"); cls_token.reshape({1,1,32}); auto pos=load_tensor(wd+"/pos_embed.wts"); pos.reshape({1,17,32}); Linear patch_proj(patch_w,patch_b); PatchEmbedding patch_embed(cls_token,pos,patch_proj,28,28,7,7); VisionTransformer vit(28,28,7,7,32,4,64,2,10); vit.set_patch_embedding(patch_embed); for(int i=0;i<2;i++) vit.set_block(i, load_block(wd,i)); auto head_w=load_tensor(wd+"/head.weight.wts"); auto head_b=load_tensor(wd+"/head.bias.wts"); vit.set_head(head_w,head_b);
    for(auto &f:files){ auto raw=load_raw(f); Tensor<float> stdn=raw; for(size_t i=0;i<stdn.size();++i) stdn[i]=(stdn[i]-0.1307f)/0.3081f; Tensor<float> cent=raw; for(size_t i=0;i<cent.size();++i) cent[i]=cent[i]-0.5f; Tensor<float> norm=raw; for(size_t i=0;i<norm.size();++i) norm[i]=(norm[i]-0.35f)/0.6f; cout<<f<<" std="<<vit.forward(stdn).argmax()<<" cent="<<vit.forward(cent).argmax()<<" norm="<<vit.forward(norm).argmax()<<"\n"; }
}
