#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include "Model.h"
using namespace std;
Tensor<float> load_tensor(const string& filepath) { ifstream file(filepath); if(!file) throw runtime_error("open fail"); vector<float> data; string line; int r=0,c=0; getline(file,line); int parsed=sscanf(line.c_str(),"# shape: %d %d",&r,&c); if(parsed==1){ c=r; r=1;} Tensor<float> t({(size_t)r,(size_t)c}); while(getline(file,line)){ istringstream iss(line); float v; while(iss>>v) data.push_back(v);} for(size_t i=0;i<min(data.size(),t.size());++i) t[i]=data[i]; return t; }
Tensor<float> load_raw(const string& path){ string raw=path+".raw"; ifstream file(raw,ios::binary); if(!file.is_open()) file.open(path,ios::binary); if(!file) throw runtime_error("open fail"); Tensor<float> t({1,28,28}); unsigned char pix; for(int i=0;i<28;i++) for(int j=0;j<28;j++){ if(!file.read((char*)&pix,1)) throw runtime_error("read fail"); t(0,i,j)=float(pix)/255.0f;} return t; }
Tensor<float> shift_image(const Tensor<float>& img,int dy,int dx){ Tensor<float> out({1,28,28}); for(int i=0;i<28;i++) for(int j=0;j<28;j++){ int sy=i-dy, sx=j-dx; out(0,i,j) = (sy>=0 && sy<28 && sx>=0 && sx<28) ? img(0,sy,sx) : 0.0f; } return out; }
int main(){ vector<string> files={"mnist_0.raw","mnist_1.raw","mnist_2.raw","mnist_4.raw","mnist_5.raw","mnist_7.raw","mnist_9.raw"}; string wd="weights";
 auto patch_w=load_tensor(wd+"/patch.weight.wts"); auto patch_b=load_tensor(wd+"/patch.bias.wts"); auto cls=load_tensor(wd+"/cls_token.wts"); cls.reshape({1,1,32}); auto pos=load_tensor(wd+"/pos_embed.wts"); pos.reshape({1,17,32}); Linear patch_proj(patch_w,patch_b); PatchEmbedding patch_embed(cls,pos,patch_proj,28,28,7,7); VisionTransformer vit(28,28,7,7,32,4,64,2,10); vit.set_patch_embedding(patch_embed); for(int i=0;i<2;i++) vit.set_block(i, {load_tensor(wd+"/blocks."+to_string(i)+".attn.q.weight.wts"),load_tensor(wd+"/blocks."+to_string(i)+".attn.q.bias.wts")});}
