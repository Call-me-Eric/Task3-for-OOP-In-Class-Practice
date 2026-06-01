#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_test_data.py

对当前 tests/ 目录中的 raw 测试样本运行编译好的 C++ 可执行文件，并输出最终预测准确率。

用法示例：
    python evaluate_test_data.py --binary ./main --tests-dir mnist_raw/test --weights-dir weights_finetuned_norm
"""

import argparse
import re
import subprocess
from pathlib import Path


def parse_label_from_filename(filename: str) -> int | None:
    m = re.search(r'label_(\d+)\.raw$', filename)
    if m:
        return int(m.group(1))
    return None


def collect_labeled_samples(tests_dir: Path) -> list[tuple[Path, int]]:
    samples = []
    labels_file = tests_dir / 'labels.txt'
    if labels_file.exists():
        with labels_file.open('r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                sample_name, label_text = parts
                sample_path = tests_dir / sample_name
                if not sample_path.exists():
                    continue
                try:
                    label = int(label_text)
                except ValueError:
                    continue
                samples.append((sample_path, label))
        return samples

    for path in sorted(tests_dir.glob('*.raw')):
        label = parse_label_from_filename(path.name)
        if label is None:
            continue
        samples.append((path, label))
    return samples


def run_binary(binary_path: Path, sample_path: Path, weights_dir: Path) -> int | None:
    proc = subprocess.run(
        [str(binary_path), str(sample_path), str(weights_dir), '0'],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        raise RuntimeError(f'可执行文件运行失败: {sample_path} -> {proc.stderr.strip()}')
    out = proc.stdout.strip().splitlines()
    if not out:
        raise RuntimeError(f'可执行文件未输出结果: {sample_path}')
    try:
        return int(out[-1])
    except ValueError as ex:
        raise RuntimeError(f'无法解析预测值: {out[-1]!r}') from ex


def main() -> None:
    parser = argparse.ArgumentParser(description='评估 MNIST raw 样本的最终预测准确率')
    parser.add_argument('--binary', default='./main', help='C++ 可执行文件路径')
    parser.add_argument('--tests-dir', default='mnist_raw/test', help='测试样本目录')
    parser.add_argument('--weights-dir', default='weights_finetuned_norm', help='权重目录')
    parser.add_argument('--max-samples', type=int, default=0, help='最多评估样本数量，0 表示全部')
    args = parser.parse_args()

    binary_path = Path(args.binary)
    if not binary_path.is_absolute():
        binary_path = Path.cwd() / binary_path
    tests_dir = Path(args.tests_dir)
    weights_dir = Path(args.weights_dir)

    if not binary_path.exists():
        raise FileNotFoundError(f'找不到可执行文件: {binary_path}')
    if not tests_dir.exists():
        raise FileNotFoundError(f'找不到测试目录: {tests_dir}')
    if not weights_dir.exists():
        raise FileNotFoundError(f'找不到权重目录: {weights_dir}')

    samples = collect_labeled_samples(tests_dir)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    if not samples:
        raise RuntimeError('未找到任何带 label 的 raw 测试样本')

    correct = 0
    total = len(samples)
    for idx, (sample_path, label) in enumerate(samples, 1):
        try:
            pred = run_binary(binary_path, sample_path, weights_dir)
        except Exception as ex:
            print(f'[WARN] 第 {idx} 个样本评估失败: {sample_path.name}: {ex}')
            continue
        if pred == label:
            correct += 1
        if idx % 100 == 0 or idx == total:
            print(f'已评估 {idx}/{total}，当前准确数 {correct}')

    accuracy = correct / total * 100.0
    print('\n=== 评估结果 ===')
    print(f'样本数量: {total}')
    print(f'正确数量: {correct}')
    print(f'最终预测准确率: {accuracy:.2f}%')


if __name__ == '__main__':
    main()
