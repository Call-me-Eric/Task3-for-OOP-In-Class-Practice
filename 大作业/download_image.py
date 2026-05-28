from torchvision import datasets
from PIL import Image

# 下载 MNIST 测试集
mnist = datasets.MNIST(root='.', train=False, download=True)

# 保存前10张图片，文件名就是标签
for i in range(10):
    img, label = mnist[i]
    img.save(f"mnist_{label}.png")
    print(f"保存: mnist_{label}.png，标签: {label}")