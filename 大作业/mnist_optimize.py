#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mnist_optimize.py

1. 下载 MNIST 数据集并解析为 raw 二进制格式
2. 读取 main.cpp 所使用的 weights 目录中的 .wts 权重
3. 在 Python 中复现 Vision Transformer 前向推理
4. 评估不同预处理策略，帮助你优化 main.cpp 的输入处理逻辑

用法示例：
    python mnist_optimize.py --weights-dir weights --output-dir mnist_raw --max-samples 2000

如果你希望直接对现有编译好的 C++ 二进制做评估，可提供 --binary ./main 或 --binary ./vit_infer.exe
"""

import argparse
import gzip
import os
import ssl
import struct
import sys
import urllib.request
import subprocess
from pathlib import Path
from scipy.special import erf

import numpy as np

MNIST_FILES = {
    "train_images": (
        "train-images-idx3-ubyte.gz",
        [
            "https://raw.githubusercontent.com/geektutu/tensorflow-tutorial-samples/master/mnist/data_set/train-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        ],
    ),
    "train_labels": (
        "train-labels-idx1-ubyte.gz",
        [
            "https://raw.githubusercontent.com/geektutu/tensorflow-tutorial-samples/master/mnist/data_set/train-labels-idx1-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        ],
    ),
    "test_images": (
        "t10k-images-idx3-ubyte.gz",
        [
            "https://raw.githubusercontent.com/geektutu/tensorflow-tutorial-samples/master/mnist/data_set/t10k-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        ],
    ),
    "test_labels": (
        "t10k-labels-idx1-ubyte.gz",
        [
            "https://raw.githubusercontent.com/geektutu/tensorflow-tutorial-samples/master/mnist/data_set/t10k-labels-idx1-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        ],
    ),
}


def download_file(urls: list[str], target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    last_exception = None
    for url in urls:
        print(f"尝试下载 {url} -> {target}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
                with target.open("wb") as out_file:
                    out_file.write(response.read())
            return
        except Exception as exc:
            last_exception = exc
            print(f"  下载失败: {url} => {exc}")
    raise RuntimeError(f"所有 MNIST 下载源都失败: {last_exception}")


def parse_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2051:
            raise ValueError(f"不是有效的 MNIST 图像文件: {path}")
        count = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(count * rows * cols), dtype=np.uint8)
        return data.reshape(count, rows, cols)


def parse_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2049:
            raise ValueError(f"不是有效的 MNIST 标签文件: {path}")
        count = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(count), dtype=np.uint8)
        return data


def download_mnist(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, (filename, urls) in MNIST_FILES.items():
        download_file(urls, data_dir / filename)

    train_images = parse_idx_images(data_dir / MNIST_FILES["train_images"][0])
    train_labels = parse_idx_labels(data_dir / MNIST_FILES["train_labels"][0])
    test_images = parse_idx_images(data_dir / MNIST_FILES["test_images"][0])
    test_labels = parse_idx_labels(data_dir / MNIST_FILES["test_labels"][0])
    return train_images, train_labels, test_images, test_labels


def save_raw_images(images: np.ndarray, labels: np.ndarray, out_dir: Path, prefix: str = "test") -> Path:
    out_dir = out_dir / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = out_dir / "labels.txt"
    with mapping_path.open("w", encoding="utf-8") as f:
        for idx, (img, label) in enumerate(zip(images, labels)):
            filename = f"{prefix}_{idx:05d}_label_{label}.raw"
            raw_path = out_dir / filename
            img.astype(np.uint8).tofile(raw_path)
            f.write(f"{filename} {label}\n")
    return mapping_path


def read_wts(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line.startswith("# shape:"):
            raise ValueError(f"无效的 wts 文件头: {path}")
        shape_values = [int(x) for x in first_line[len("# shape:"):].split() if x.strip()]
        if len(shape_values) == 1:
            shape = (1, shape_values[0])
        elif len(shape_values) == 2:
            shape = tuple(shape_values)
        else:
            raise ValueError(f"不支持的 shape 维度: {shape_values}")
        data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.extend(float(x) for x in line.split())
        arr = np.array(data, dtype=np.float32)
        expected = int(np.prod(shape))
        if arr.size < expected:
            arr = np.pad(arr, (0, expected - arr.size), mode="constant")
        return arr.reshape(shape)


def load_vit_weights(weights_dir: Path) -> dict[str, np.ndarray]:
    weights = {}
    weights["patch.weight"] = read_wts(weights_dir / "patch.weight.wts")
    weights["patch.bias"] = read_wts(weights_dir / "patch.bias.wts")
    weights["cls_token"] = read_wts(weights_dir / "cls_token.wts").reshape(1, 1, 32)
    weights["pos_embed"] = read_wts(weights_dir / "pos_embed.wts").reshape(1, 17, 32)
    for i in range(2):
        prefix = weights_dir / f"blocks.{i}"
        weights[f"blocks.{i}.attn.q.weight"] = read_wts(prefix.with_name(prefix.name + ".attn.q.weight.wts"))
        weights[f"blocks.{i}.attn.q.bias"] = read_wts(prefix.with_name(prefix.name + ".attn.q.bias.wts"))
        weights[f"blocks.{i}.attn.k.weight"] = read_wts(prefix.with_name(prefix.name + ".attn.k.weight.wts"))
        weights[f"blocks.{i}.attn.k.bias"] = read_wts(prefix.with_name(prefix.name + ".attn.k.bias.wts"))
        weights[f"blocks.{i}.attn.v.weight"] = read_wts(prefix.with_name(prefix.name + ".attn.v.weight.wts"))
        weights[f"blocks.{i}.attn.v.bias"] = read_wts(prefix.with_name(prefix.name + ".attn.v.bias.wts"))
        weights[f"blocks.{i}.attn.o.weight"] = read_wts(prefix.with_name(prefix.name + ".attn.o.weight.wts"))
        weights[f"blocks.{i}.attn.o.bias"] = read_wts(prefix.with_name(prefix.name + ".attn.o.bias.wts"))
        weights[f"blocks.{i}.mlp.fc1.weight"] = read_wts(prefix.with_name(prefix.name + ".mlp.fc1.weight.wts"))
        weights[f"blocks.{i}.mlp.fc1.bias"] = read_wts(prefix.with_name(prefix.name + ".mlp.fc1.bias.wts"))
        weights[f"blocks.{i}.mlp.fc2.weight"] = read_wts(prefix.with_name(prefix.name + ".mlp.fc2.weight.wts"))
        weights[f"blocks.{i}.mlp.fc2.bias"] = read_wts(prefix.with_name(prefix.name + ".mlp.fc2.bias.wts"))
        weights[f"blocks.{i}.norm1.gamma"] = read_wts(prefix.with_name(prefix.name + ".norm1.gamma.wts"))
        weights[f"blocks.{i}.norm1.beta"] = read_wts(prefix.with_name(prefix.name + ".norm1.beta.wts"))
        weights[f"blocks.{i}.norm2.gamma"] = read_wts(prefix.with_name(prefix.name + ".norm2.gamma.wts"))
        weights[f"blocks.{i}.norm2.beta"] = read_wts(prefix.with_name(prefix.name + ".norm2.beta.wts"))
    weights["head.weight"] = read_wts(weights_dir / "head.weight.wts")
    weights["head.bias"] = read_wts(weights_dir / "head.bias.wts")
    return weights


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if gamma.size and beta.size:
        out = out * gamma.reshape(1, 1, -1) + beta.reshape(1, 1, -1)
    return out


def mha_forward(x: np.ndarray, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, num_heads: int) -> np.ndarray:
    batch, seq_len, hidden = x.shape
    head_dim = hidden // num_heads
    q = x @ q_w + q_b
    k = x @ k_w + k_b
    v = x @ v_w + v_b
    q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    scores = q @ k.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(float(head_dim))
    attention = softmax(scores, axis=-1)
    context = attention @ v
    context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, hidden)
    output = context @ out_w + out_b
    return output


def transformer_block_forward(x: np.ndarray, weights: dict, idx: int) -> np.ndarray:
    x_norm = layer_norm(x, weights[f"blocks.{idx}.norm1.gamma"], weights[f"blocks.{idx}.norm1.beta"])
    attn = mha_forward(
        x_norm,
        weights[f"blocks.{idx}.attn.q.weight"],
        weights[f"blocks.{idx}.attn.q.bias"],
        weights[f"blocks.{idx}.attn.k.weight"],
        weights[f"blocks.{idx}.attn.k.bias"],
        weights[f"blocks.{idx}.attn.v.weight"],
        weights[f"blocks.{idx}.attn.v.bias"],
        weights[f"blocks.{idx}.attn.o.weight"],
        weights[f"blocks.{idx}.attn.o.bias"],
        num_heads=4,
    )
    x = x + attn
    x_norm2 = layer_norm(x, weights[f"blocks.{idx}.norm2.gamma"], weights[f"blocks.{idx}.norm2.beta"])
    mlp = gelu(x_norm2 @ weights[f"blocks.{idx}.mlp.fc1.weight"] + weights[f"blocks.{idx}.mlp.fc1.bias"])
    mlp = mlp @ weights[f"blocks.{idx}.mlp.fc2.weight"] + weights[f"blocks.{idx}.mlp.fc2.bias"]
    return x + mlp


def vit_forward(image: np.ndarray, weights: dict) -> np.ndarray:
    batch = image.shape[0]
    patch_h = patch_w = 7
    hidden_dim = 32
    cls_token = weights["cls_token"]
    pos_embed = weights["pos_embed"]
    patch_out = []
    for i in range(0, 28, patch_h):
        for j in range(0, 28, patch_w):
            patch = image[:, i:i + patch_h, j:j + patch_w].reshape(batch, patch_h * patch_w)
            token = patch @ weights["patch.weight"] + weights["patch.bias"]
            token = token.reshape(batch, 1, hidden_dim)
            pos = pos_embed[:, len(patch_out) + 1 : len(patch_out) + 2, :]
            patch_out.append(token + pos)
    cls = np.broadcast_to(cls_token + pos_embed[:, :1, :], (batch, 1, hidden_dim))
    tokens = np.concatenate([cls] + patch_out, axis=1)
    for idx in range(2):
        tokens = transformer_block_forward(tokens, weights, idx)
    cls_token_out = tokens[:, 0, :]
    logits = cls_token_out @ weights["head.weight"] + weights["head.bias"]
    return logits


def evaluate_strategies(images: np.ndarray, labels: np.ndarray, weights: dict) -> dict[str, float]:
    predictions = {
        "raw": [],
        "std": [],
        "center": [],
        "norm": [],
        "confidence_choice": [],
        "margin_choice": [],
    }
    count = len(images)
    for idx in range(count):
        x = images[idx : idx + 1].astype(np.float32) / 255.0
        x_raw = x
        x_std = (x_raw - 0.1307) / 0.3081
        x_center = x_raw - 0.5
        x_norm = (x_raw - 0.35) / 0.6
        logits_raw = vit_forward(x_raw, weights)
        logits_std = vit_forward(x_std, weights)
        logits_center = vit_forward(x_center, weights)
        logits_norm = vit_forward(x_norm, weights)
        probs_std = softmax(logits_std, axis=1)
        probs_center = softmax(logits_center, axis=1)
        std_pred = int(np.argmax(logits_std[0]))
        center_pred = int(np.argmax(logits_center[0]))
        raw_pred = int(np.argmax(logits_raw[0]))
        norm_pred = int(np.argmax(logits_norm[0]))
        predictions["raw"].append(raw_pred)
        predictions["std"].append(std_pred)
        predictions["center"].append(center_pred)
        predictions["norm"].append(norm_pred)
        if std_pred == center_pred:
            predictions["confidence_choice"].append(std_pred)
            predictions["margin_choice"].append(std_pred)
        else:
            std_conf = float(np.max(probs_std))
            center_conf = float(np.max(probs_center))
            predictions["confidence_choice"].append(std_pred if std_conf >= center_conf else center_pred)
            sorted_std = np.partition(probs_std[0], -2)
            sorted_center = np.partition(probs_center[0], -2)
            std_margin = float(sorted_std[-1] - sorted_std[-2])
            center_margin = float(sorted_center[-1] - sorted_center[-2])
            predictions["margin_choice"].append(std_pred if std_margin >= center_margin else center_pred)

    results = {}
    for key, preds in predictions.items():
        results[key] = float(np.mean(np.array(preds, dtype=np.int64) == labels[: len(preds)]))
    return results


def run_binary_on_samples(binary_path: Path, weights_dir: Path, raw_dir: Path, labels_path: Path, max_samples: int) -> float:
    if not binary_path.exists():
        raise FileNotFoundError(f"二进制文件不存在: {binary_path}")
    labels = []
    names = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            name, label = line.strip().split()
            names.append(name)
            labels.append(int(label))
            if len(names) >= max_samples:
                break
    correct = 0
    for name, label in zip(names, labels):
        raw_path = raw_dir / name
        proc = subprocess.run(
            [str(binary_path), str(raw_path), str(weights_dir), "0"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            print(f"[WARN] binary 运行失败: {raw_path} -> {proc.stderr.strip()}")
            continue
        try:
            pred = int(proc.stdout.strip().splitlines()[-1])
        except Exception as ex:
            print(f"[WARN] 无法解析输出: {proc.stdout!r}, {ex}")
            continue
        if pred == label:
            correct += 1
    return correct / len(names) if names else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 MNIST、生成 raw 文件，并评估 main.cpp 预处理策略")
    parser.add_argument("--data-dir", default="mnist_data", help="MNIST 数据下载与缓存目录")
    parser.add_argument("--output-dir", default="mnist_raw", help="生成 raw 文件的输出目录")
    parser.add_argument("--weights-dir", default="weights", help="权重目录")
    parser.add_argument("--max-samples", type=int, default=1000, help="最多评估的测试图像数量")
    parser.add_argument("--binary", default=None, help="可选：已编译的 C++ 可执行文件路径，用于直接评估 main.cpp 行为")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    weights_dir = Path(args.weights_dir)

    print("=== MNIST 数据下载与准备 ===")
    train_images, train_labels, test_images, test_labels = download_mnist(data_dir)
    print(f"下载并解析完成: {train_images.shape[0]} 训练图像, {test_images.shape[0]} 测试图像")

    print("=== 生成 raw 文件 ===")
    mapping_path = save_raw_images(test_images[: args.max_samples], test_labels[: args.max_samples], output_dir, prefix="test")
    raw_dir = mapping_path.parent
    print(f"已生成 {len(test_images[:args.max_samples])} 个 raw 文件到 {raw_dir}")

    print("=== 加载权重 ===")
    weights = load_vit_weights(weights_dir)
    print("权重加载完成")

    print("=== 评估不同预处理策略 ===")
    metrics = evaluate_strategies(test_images[: args.max_samples], test_labels[: args.max_samples], weights)
    for key, acc in metrics.items():
        print(f"{key:20s} accuracy = {acc * 100:.2f}%")

    if args.binary is not None:
        binary_path = Path(args.binary)
        print("=== 使用 C++ 二进制直接评估 main.cpp ===")
        try:
            binary_acc = run_binary_on_samples(binary_path, weights_dir, raw_dir, mapping_path, args.max_samples)
            print(f"binary accuracy = {binary_acc * 100:.2f}%")
        except Exception as ex:
            print(f"[WARN] 无法执行二进制评估: {ex}")

    print("\n=== 建议 ===")
    print("如果 std 或 center 之一表现更好，可以将 main.cpp 中的双分支硬编码逻辑替换为: ")
    print("  - 先分别计算 std 和 center 两种预处理后的 logits")
    print("  - 如果两者预测相同，直接使用该预测")
    print("  - 否则比较 softmax 置信度或 margin，选择更有信心的结果")
    print("例如：如果 logits_std 和 logits_center 不一致，就选置信度更高的一组输出。")

    print("如果需要，我也可以继续帮你把 main.cpp 中对应部分改成更稳妥的选择策略。")


if __name__ == "__main__":
    main()
