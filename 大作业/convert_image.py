from PIL import Image
import numpy as np

def png_to_raw(input_png, output_raw):
    img = Image.open(input_png)
    img = img.convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.uint8)
    img_array = 255 - img_array     # 关键：反色
    img_array.tofile(output_raw)
    print(f"转换完成")

if __name__ == "__main__":
    input_png = input("请输入 PNG 文件路径: ")
    output_raw = input("请输入输出 RAW 文件路径: ")
    png_to_raw(input_png, output_raw)