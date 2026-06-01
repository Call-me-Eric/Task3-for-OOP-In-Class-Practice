#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_local_mnist.py

Use local MNIST files (downloaded by mnist_optimize) to fine-tune the ViT
and export updated .wts files.

Example:
    python3 train_local_mnist.py --weights-dir weights --out-dir weights_finetuned --epochs 3 --max-samples 2000
"""

import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', default='weights')
    parser.add_argument('--out-dir', default='weights_finetuned')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-samples', type=int, default=2000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--freeze-backbone', action='store_true', help='只微调 head，冻结其余层')
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception:
        print('需要安装 torch 才能训练')
        raise

    # import helper functions and model code from fine_tune_weights.py
    import fine_tune_weights as fw_mod
    import mnist_optimize as mn_mod

    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.out_dir)

    print('加载原始权重...')
    weights = fw_mod.load_weights_dict(weights_dir)

    print('构建 PyTorch 模型...')
    import importlib.util, types
    spec = importlib.util.spec_from_loader('vt_model', loader=None)
    vt_mod = types.ModuleType('vt_model')
    exec(fw_mod.MODEL_CODE, vt_mod.__dict__)
    Vit = vt_mod.VisionTransformer

    device = torch.device(args.device)
    model = Vit().to(device)

    def to_t(t):
        return torch.from_numpy(t).to(torch.float32)

    # initialize
    model.patch_proj.weight.data.copy_(to_t(weights['patch.weight']).T)
    model.patch_proj.bias.data.copy_(to_t(weights['patch.bias']).reshape(-1))
    model.cls_token.data.copy_(to_t(weights['cls_token'].reshape(1,1,32)))
    model.pos_embed.data.copy_(to_t(weights['pos_embed']))

    for i in range(2):
        blk = model.blocks[i]
        blk.q.weight.data.copy_(to_t(weights[f'blocks.{i}.attn.q.weight']).T)
        blk.q.bias.data.copy_(to_t(weights[f'blocks.{i}.attn.q.bias']).reshape(-1))
        blk.k.weight.data.copy_(to_t(weights[f'blocks.{i}.attn.k.weight']).T)
        blk.k.bias.data.copy_(to_t(weights[f'blocks.{i}.attn.k.bias']).reshape(-1))
        blk.v.weight.data.copy_(to_t(weights[f'blocks.{i}.attn.v.weight']).T)
        blk.v.bias.data.copy_(to_t(weights[f'blocks.{i}.attn.v.bias']).reshape(-1))
        blk.out.weight.data.copy_(to_t(weights[f'blocks.{i}.attn.o.weight']).T)
        blk.out.bias.data.copy_(to_t(weights[f'blocks.{i}.attn.o.bias']).reshape(-1))
        blk.mlp.fc1.weight.data.copy_(to_t(weights[f'blocks.{i}.mlp.fc1.weight']).T)
        blk.mlp.fc1.bias.data.copy_(to_t(weights[f'blocks.{i}.mlp.fc1.bias']).reshape(-1))
        blk.mlp.fc2.weight.data.copy_(to_t(weights[f'blocks.{i}.mlp.fc2.weight']).T)
        blk.mlp.fc2.bias.data.copy_(to_t(weights[f'blocks.{i}.mlp.fc2.bias']).reshape(-1))
        try:
            blk.norm1.weight.data.copy_(to_t(weights[f'blocks.{i}.norm1.gamma']).reshape(-1))
            blk.norm1.bias.data.copy_(to_t(weights[f'blocks.{i}.norm1.beta']).reshape(-1))
            blk.norm2.weight.data.copy_(to_t(weights[f'blocks.{i}.norm2.gamma']).reshape(-1))
            blk.norm2.bias.data.copy_(to_t(weights[f'blocks.{i}.norm2.beta']).reshape(-1))
        except Exception:
            pass

    model.head.weight.data.copy_(to_t(weights['head.weight']).T)
    model.head.bias.data.copy_(to_t(weights['head.bias']).reshape(-1))

    print('准备 MNIST 数据（使用 mnist_optimize 下载的原始文件）...')
    train_images, train_labels, test_images, test_labels = mn_mod.download_mnist(Path('mnist_data'))
    if args.max_samples > 0:
        train_images = train_images[: args.max_samples]
        train_labels = train_labels[: args.max_samples]

    # to torch tensors and apply dataset-level normalization expected by the model
    X = torch.from_numpy(train_images.astype(np.float32) / 255.0).to(device)
    # apply MNIST global normalization used in earlier analysis
    X = (X - 0.1307) / 0.3081
    y = torch.from_numpy(train_labels.astype(np.int64)).to(device)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    # optionally freeze backbone (only train head)
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith('head'):
                p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable, lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('开始训练...')
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            inputs = inputs.squeeze(1) if inputs.dim()==4 else inputs
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if i % 100 == 0:
                print(f'epoch {epoch+1}/{args.epochs} step {i} loss {running_loss/(i+1):.4f} acc {correct/total:.4f}')

    print('训练完成，导出权重...')
    new_w = {}
    new_w['patch.weight'] = model.patch_proj.weight.data.cpu().numpy().T
    new_w['patch.bias'] = model.patch_proj.bias.data.cpu().numpy().reshape(1, -1)
    new_w['cls_token'] = model.cls_token.data.cpu().numpy().reshape(1,1,-1)
    new_w['pos_embed'] = model.pos_embed.data.cpu().numpy().reshape(1,17,-1)
    for i in range(2):
        blk = model.blocks[i]
        new_w[f'blocks.{i}.attn.q.weight'] = blk.q.weight.data.cpu().numpy().T
        new_w[f'blocks.{i}.attn.q.bias'] = blk.q.bias.data.cpu().numpy().reshape(1,-1)
        new_w[f'blocks.{i}.attn.k.weight'] = blk.k.weight.data.cpu().numpy().T
        new_w[f'blocks.{i}.attn.k.bias'] = blk.k.bias.data.cpu().numpy().reshape(1,-1)
        new_w[f'blocks.{i}.attn.v.weight'] = blk.v.weight.data.cpu().numpy().T
        new_w[f'blocks.{i}.attn.v.bias'] = blk.v.bias.data.cpu().numpy().reshape(1,-1)
        new_w[f'blocks.{i}.attn.o.weight'] = blk.out.weight.data.cpu().numpy().T
        new_w[f'blocks.{i}.attn.o.bias'] = blk.out.bias.data.cpu().numpy().reshape(1,-1)
        new_w[f'blocks.{i}.mlp.fc1.weight'] = blk.mlp.fc1.weight.data.cpu().numpy().T
        new_w[f'blocks.{i}.mlp.fc1.bias'] = blk.mlp.fc1.bias.data.cpu().numpy().reshape(1,-1)
        new_w[f'blocks.{i}.mlp.fc2.weight'] = blk.mlp.fc2.weight.data.cpu().numpy().T
        new_w[f'blocks.{i}.mlp.fc2.bias'] = blk.mlp.fc2.bias.data.cpu().numpy().reshape(1,-1)
        new_w[f'blocks.{i}.norm1.gamma'] = blk.norm1.weight.data.cpu().numpy().reshape(1,-1)
        new_w[f'blocks.{i}.norm1.beta'] = blk.norm1.bias.data.cpu().numpy().reshape(1,-1)
        new_w[f'blocks.{i}.norm2.gamma'] = blk.norm2.weight.data.cpu().numpy().reshape(1,-1)
        new_w[f'blocks.{i}.norm2.beta'] = blk.norm2.bias.data.cpu().numpy().reshape(1,-1)
    new_w['head.weight'] = model.head.weight.data.cpu().numpy().T
    new_w['head.bias'] = model.head.bias.data.cpu().numpy().reshape(1,-1)

    fw_mod.save_weights_dict(out_dir, new_w)
    print('已将微调后的权重保存到', out_dir)


if __name__ == '__main__':
    main()
