#include<vector>
#include<cmath>
#include<random>
#include<numeric>
#include<cstring>
#include<stdexcept>
using namespace std;

template <class T>
class Tensor {
private:
    vector<T> _data;
    vector<size_t> _shape;
    vector<size_t> _strides;

    //步长
    void compute_strides(){
        _strides.resize(_shape.size(),1);

        for(int i=_shape.size()-2;i>=0;i--){
            _strides[i]=_strides[i+1]*_shape[i+1];
        }
    }

    //多维转一维
    template <class...Index>
    size_t to_flat_idx(Index... idx) const{
        vector<size_t> indices={static_cast<size_t>(idx)...};
        if (indices.size() != _shape.size()) {
            throw invalid_argument("索引维度与张量维度不匹配");
        }
        size_t flat=0;
        for (size_t i=0;i<indices.size();++i){
            if(indices[i]>=_shape[i]){
                throw out_of_range("索引越界");
            }
            flat+=indices[i]*_strides[i];
        }
        return flat;
    }

     //形状元素一致
    static size_t count_elements(const vector<size_t>& shape){
        return accumulate(shape.begin(),shape.end(),1ULL,multiplies<>());
    }

public:
    //默认构造
    Tensor() = default;

    //形状构造
    Tensor(const vector<size_t>& shape):_shape(shape){
        compute_strides();
        _data.resize(count_elements(_shape));
    }

    //形状+值构造
    Tensor(const vector<size_t>& shape, const T& init_val):_shape(shape){
        compute_strides();
        _data.resize(count_elements(_shape),init_val);
    }

    //总元素数
    size_t size() const{
        return _data.size();
    }

    //维度数
    size_t rank() const{
        return _shape.size();
    }

    //获取形状
    const vector<size_t>& shape() const{
        return _shape;
    }

    //一维索引访问
    T& operator[](size_t idx){
        if(idx>=_data.size()){
            throw out_of_range("一维索引访问越界");
        }
        return _data[idx];
    }


    const T& operator[](size_t idx) const{
                if(idx>=_data.size()){
            throw out_of_range("一维索引访问越界");
        }
        return _data[idx];
    }

    //多维索引访问
    template <class... Index>
    T& operator()(Index... idx){
        return _data[to_flat_idx(idx...)];
    }
    template <class... Index>
    const T& operator()(Index... idx) const{
        return _data[to_flat_idx(idx...)];
    }

    //原地修改形状
    void reshape(const vector<size_t>& new_shape){
        size_t old_size=count_elements(_shape);
        size_t new_size=count_elements(new_shape);
        if (old_size!=new_size){
            throw invalid_argument("reshape前后元素总数必须相等");
        }
        _shape=new_shape;
        compute_strides();
    }

    //返回新形状
    Tensor<T> reshaped(const vector<size_t>& new_shape) const{
        Tensor<T> out=*this;
        out.reshape(new_shape);
        return out; 
    }

    //维度重排
    Tensor<T> permute(const vector<size_t>& dims) const{
        if(dims.size()!=_shape.size()){
            throw invalid_argument("permute维度必须与原维度一致");
        }
        vector<size_t> new_shape,new_strides;
        for(size_t d:dims){
            new_shape.push_back(_shape[d]);
            new_strides.push_back(_strides[d]);
        }
        Tensor<T> out(new_shape);
        memcpy(out._data.data(),_data.data(),_data.size()*sizeof(T));
        out._strides=new_strides;
        return out;
        
    }

    //矩阵转置
    Tensor<T> transpose() const{
        if(_shape.size()!=2){
            throw logic_error("transpose仅支持二维张量");
        }
        return permute({1,0});
    }

    //增加维度
    Tensor<T> unsqueeze(size_t dim) const{
        if (dim>_shape.size()){
            throw out_of_range("unsqueeze维度越界");
        }
        std::vector<size_t> new_shape=_shape;
        new_shape.insert(new_shape.begin()+dim,1);
        Tensor<T> out(new_shape);
        memcpy(out._data.data(),_data.data(),_data.size()*sizeof(T));
        return out;
    }

    //压缩维度
    Tensor<T> squeeze(size_t dim) const{
        if (dim>=_shape.size()||_shape[dim]!=1){
            throw invalid_argument("squeeze仅支持size=1的维度");
        }
        vector<size_t> new_shape=_shape;
        new_shape.erase(new_shape.begin()+dim);
        Tensor<T> out(new_shape);
        memcpy(out._data.data(),_data.data(),_data.size()*sizeof(T));
        return out;
    }

    //切片
    Tensor<T> slice(size_t dim,size_t begin,size_t end) const{
        if (dim>=_shape.size()||begin>=end||end>_shape[dim]){
            throw out_of_range("slice参数非法");
        }
        vector<size_t> new_shape=_shape;
        new_shape[dim] =end-begin;
        Tensor<T> out(new_shape);

        size_t step=_strides[dim];
        size_t block_size=count_elements(_shape)/_shape[dim];

        for (size_t i=0;i<end-begin;++i){
            size_t src_off=(begin+i)*step;
            size_t dst_off=i*step;
            memcpy(out._data.data()+dst_off,
                        _data.data()+src_off,
                        block_size*sizeof(T));
        }
        return out;
    }

    //拼接
    static Tensor<T> concat(const vector<Tensor<T>>& tensors,size_t dim){
        if(tensors.empty()) throw invalid_argument("待拼接张量列表为空");
        size_t rank=tensors[0].rank();
        if (dim>=rank) throw out_of_range("concat维度越界");

        vector<size_t> new_shape=tensors[0].shape();
        size_t total_dim=0;
        for(const auto& t:tensors){
            if(t.shape()!=new_shape){
                throw invalid_argument("concat仅指定维度可不同,其余维度必须一致");
            }
            total_dim+=t.shape()[dim];
        }
        new_shape[dim]=total_dim;
        Tensor<T> out(new_shape);

        size_t offset=0;
        size_t block=count_elements(new_shape)/total_dim;
        for (const auto& t:tensors){
            size_t cnt=t.shape()[dim];
            memcpy(out._data.data()+offset,
                        t._data.data(),
                        cnt*block*sizeof(T));
            offset+=cnt*block;
        }
        return out;
    }

    //矩阵乘法
    Tensor<T> matmul(const Tensor<T>& other) const{
        if(_shape.size()<2||other.rank()<2){
            throw logic_error("matmul至少为二维张量");
        }
        size_t M=_shape[_shape.size()-2];
        size_t K=_shape.back();
        size_t N=other.shape().back();
        if(K!=other.shape()[other.rank()-2]){
            throw std::invalid_argument("matmul维度不匹配");
        }

        vector<size_t> out_shape=_shape;
        out_shape.pop_back();
        out_shape.push_back(N);
        Tensor<T> out(out_shape,T(0));

        size_t batch=size()/(M*K);
        size_t other_batch=other.size()/(K*N);
        if (batch!=other_batch){
            throw invalid_argument("matmul batch维度不匹配");
        }

        for(size_t b=0;b<batch;++b){
            for(size_t i=0;i<M;++i){
                for (size_t k=0;k<K;++k){
                    T a=_data[b*M*K+i*K+k];
                    for(size_t j=0;j<N;++j){
                        out[b*M*N+i*N+j]+=a*other._data[b*K*N+k*N+j];
                    }
                }
            }
        }
        return out;
    }

    //偏置加法
    Tensor<T> bias_add(const Tensor<T>& other) const{
        return *this+other;
    }

    //指定维度归一化
    Tensor<T> softmax(size_t dim) const{
        if(dim>=_shape.size()) throw out_of_range("softmax维度越界");
        Tensor<T> out=*this;
        size_t outer=count_elements(_shape)/_shape[dim];
        size_t inner=_shape[dim];

        for (size_t o=0;o<outer;++o){
            T max_val=_data[o*inner];
            for (size_t i=1;i<inner;++i){
                if (_data[o*inner+i]>max_val) max_val=_data[o*inner+i];
            }
            T sum=0;
            for (size_t i=0;i<inner;++i){
                out._data[o*inner+i]=exp(_data[o*inner+i]-max_val);
                sum+=out._data[o*inner+i];
            }
            for (size_t i=0;i<inner;++i){
                out._data[o*inner+i]/=sum;
            }
        }
        return out;
    }

    size_t argmax() const{
        if(_data.empty())   throw logic_error("空张量无法argmax");
        size_t idx=0;
        for(size_t i=1 ;i<_data.size();++i){
            if(_data[i]>_data[idx]) idx=i;
        }
        return idx;
    }

    //加号运算符
    Tensor<T> operator+(const Tensor<T>& other) const{
        vector<size_t> shape1=_shape;
        vector<size_t> shape2=other._shape;

        while(shape1.size()<shape2.size()) shape1.insert(shape1.begin(),1);
        while(shape2.size()<shape1.size()) shape2.insert(shape2.begin(),1);

        vector<size_t> result_shape;
        for (size_t i=0;i<shape1.size();++i){
            if(shape1[i]!=shape2[i]&&shape1[i]!=1&&shape2[i]!=1){
                throw runtime_error("+: shape not compatible");
            }
            result_shape.push_back(max(shape1[i],shape2[i]));
        }

        Tensor<T> res(result_shape);
        vector<size_t> idx(result_shape.size(),0);

        for(size_t i=0;i<res.size();++i){
            size_t i1=0,i2=0;
            int off1=(int)result_shape.size()-(int)_shape.size();
            int off2=(int)result_shape.size()-(int)other._shape.size();

            for(size_t d=0;d<_shape.size();++d)
                if(_shape[d]!=1)
                    i1+=idx[d+off1]*_strides[d];

            for(size_t d=0;d<other._shape.size();++d)
                if(other._shape[d]!=1)
                    i2+=idx[d+off2]*other._strides[d];

            res._data[i]=_data[i1]+other._data[i2];

            for(int d=(int)idx.size()-1;d>=0;--d){
                if(++idx[d]<result_shape[d]) break;
                idx[d]=0;
            }
        }
        return res;
    }

    Tensor<T> broadcast(const vector<size_t>& new_shape) const{
        if(new_shape.size()!=_shape.size()){
            throw runtime_error("Broadcast: rank mismatch");
        }
        for(size_t i=0;i<_shape.size();++i){
            if(_shape[i]!=1&&_shape[i]!=new_shape[i]){
                throw runtime_error("Broadcast: incompatible shapes");
            }
        }
        Tensor<T> result(new_shape);
        vector<size_t> indices(new_shape.size(),0);
        for(size_t i=0;i<result.size();++i){
            size_t src_i=0;
            for(size_t d=0;d<_shape.size();++d){
                if(_shape[d]!=1){
                    src_i+=indices[d]*_strides[d];
                }
            }
            result._data[i]=_data[src_i];
            for (int d=(int)indices.size()-1;d>=0;--d) {
                if (++indices[d]<new_shape[d]) break;
                indices[d] = 0;
            }
        }
        return result;
    }
    void he_normal_init(size_t in_feature,size_t out_feature)
    {
        *this = Tensor<T>({in_feature, out_feature});
        float std_dev = std::sqrt(2.0f / static_cast<float>(in_feature));

        random_device rd;
        mt19937 rnd(rd());
        normal_distribution<float> normal_dist(0.0f, 1.0f);
        for(auto &i : _data)i = normal_dist(rnd) * std_dev;
    }
    void zero_init(size_t in_feature,size_t out_feature)
    {
        *this = Tensor<T>({in_feature, out_feature});
        for(auto &i : _data)i = 0;
    }
    std::vector<size_t> Tensor<T>::index_to_coordinates(size_t idx) const;
};

