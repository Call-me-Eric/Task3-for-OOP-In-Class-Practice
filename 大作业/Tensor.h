#include<vector>
#include<cmath>
#include<random>

template <class T>
class Tensor {
private:
    std::vector<T> _data;
    std::vector<size_t> _shape;
    std::vector<size_t> _strides;
public:
    Tensor() = default;
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, const T& init_val);
    size_t size() const;
    size_t rank() const;
    const std::vector<size_t>& shape() const;
    T& operator[](size_t idx);
    const T& operator[](size_t idx) const;
    template <class... Index>
    T& operator()(Index... idx);
    template <class... Index>
    const T& operator()(Index... idx) const;
    void reshape(const std::vector<size_t>& new_shape);
    Tensor<T> reshaped(const std::vector<size_t>& new_shape) const;
    Tensor<T> permute(const std::vector<size_t>& dims) const;
    Tensor<T> transpose() const;
    Tensor<T> unsqueeze(size_t dim) const;
    Tensor<T> squeeze(size_t dim) const;
    Tensor<T> slice(size_t dim, size_t begin, size_t end) const;
    static Tensor<T> concat(const std::vector<Tensor<T>>& tensors, size_t dim);
    Tensor<T> bias_add(const Tensor<T>& other) const;
    Tensor<T> matmul(const Tensor<T>& other) const;
    Tensor<T> softmax(size_t dim) const;
    size_t argmax() const;

    Tensor<T> operator+(const Tensor<T>& other) const;

    Tensor<T> broadcast(const std::vector<size_t>& new_shape) const;

    void he_normal_init(size_t in_feature,size_t out_feature)
    {
        *this = Tensor<T>({in_feature, out_feature});
        float std_dev = std::sqrt(2.0f / static_cast<float>(in_feature));

        std::random_device rd;
        std::mt19937 rnd(rd());
        std::normal_distribution<float> normal_dist(0.0f, 1.0f);
        for(auto &i : _data)i = normal_dist(rnd) * std_dev;
    }
    void zero_init(size_t in_feature,size_t out_feature)
    {
        *this = Tensor<T>({in_feature, out_feature});
        for(auto &i : _data)i = 0;
    }
};
template <class T>
Tensor<T> Tensor<T>::broadcast(const std::vector<size_t>& new_shape) const {//提供了broadcast张量广播方法
    if (new_shape.size() != _shape.size()) {
        throw std::runtime_error("Broadcast: rank mismatch");
    }
    for (size_t i = 0; i < _shape.size(); ++i) {
        if (_shape[i] != 1 && _shape[i] != new_shape[i]) {
            throw std::runtime_error("Broadcast: incompatible shapes");
        }
    }
    Tensor<T> result(new_shape);
    // 简单实现：复制数据到新形状
    // 假设广播只在某些维度
    std::vector<size_t> indices(new_shape.size(), 0);
    for (size_t i = 0; i < result.size(); ++i) {
        size_t src_i = 0;
        for (size_t d = 0; d < _shape.size(); ++d) {
            if (_shape[d] != 1) {
                src_i += indices[d] * _strides[d];
            }
        }
        result._data[i] = _data[src_i];
        // 递增 indices
        for (int d = (int)indices.size() - 1; d >= 0; --d) {
            if (++indices[d] < new_shape[d]) break;
            indices[d] = 0;
        }
    }
    return result;
}

template <class T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
    // 支持广播：扩展较小张量的形状
    std::vector<size_t> shape1 = _shape;
    std::vector<size_t> shape2 = other._shape;
    // 使形状长度相同
    while (shape1.size() < shape2.size()) shape1.insert(shape1.begin(), 1);
    while (shape2.size() < shape1.size()) shape2.insert(shape2.begin(), 1);
    std::vector<size_t> result_shape = shape1;
    for (size_t i = 0; i < result_shape.size(); ++i) {
        if (shape1[i] != shape2[i] && shape1[i] != 1 && shape2[i] != 1) {
            throw std::runtime_error("Operator+: incompatible shapes");
        }
        result_shape[i] = std::max(shape1[i], shape2[i]);
    }
    Tensor<T> result(result_shape);
    // 逐元素加法
    std::vector<size_t> indices(result_shape.size(), 0);
    for (size_t i = 0; i < result.size(); ++i) {
        size_t idx1 = 0;
        size_t offset1 = result_shape.size() - _shape.size();
        for (size_t d = 0; d < _shape.size(); ++d) {
            if (_shape[d] != 1) {
                idx1 += indices[d + offset1] * _strides[d];
            }
        }
        size_t idx2 = 0;
        size_t offset2 = result_shape.size() - other._shape.size();
        for (size_t d = 0; d < other._shape.size(); ++d) {
            if (other._shape[d] != 1) {
                idx2 += indices[d + offset2] * other._strides[d];
            }
        }
        result._data[i] = _data[idx1] + other._data[idx2];
        // 递增 indices
        for (int d = (int)indices.size() - 1; d >= 0; --d) {
            if (++indices[d] < result_shape[d]) break;
            indices[d] = 0;
        }
    }
    return result;
}
