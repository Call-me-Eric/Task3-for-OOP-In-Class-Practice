from PIL import Image
import numpy as np

def png_to_raw(input_png, output_raw, mode="RGB"):
    """
    将 PNG 转换为 RAW 原始像素数据

    参数:
        input_png : 输入 PNG 文件路径
        output_raw: 输出 RAW 文件路径
        mode      : 图像模式
                     "RGB"  -> 3通道
                     "RGBA" -> 4通道
                     "L"    -> 灰度
    """

    # 打开图像
    img = Image.open(input_png)

    # 转换格式
    img = img.convert(mode)

    # 转 numpy 数组
    img_array = np.array(img)

    # 写入 raw
    img_array.tofile(output_raw)

    print(f"转换完成:")
    print(f"输入文件: {input_png}")
    print(f"输出文件: {output_raw}")
    print(f"尺寸: {img.width} x {img.height}")
    print(f"模式: {mode}")


if __name__ == "__main__":
    input_png = str(input("请输入 PNG 文件路径: "))
    output_raw = str(input("请输入输出 RAW 文件路径: "))

    png_to_raw(input_png, output_raw, mode="L")