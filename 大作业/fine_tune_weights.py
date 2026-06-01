#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_tune_weights.py

使用 PyTorch 在 MNIST 上微调现有 .wts 权重并将更新后的权重导出为相同格式。

用法示例：
    python3 fine_tune_weights.py --weights-dir weights --out-dir weights_finetuned --epochs 3 --max-samples 2000

注意：运行需要安装 `torch` 和 `torchvision`。
"""

import argparse
from pathlib import Path
import numpy as np
import os
import sys


def read_wts(path: Path) -> np.ndarray:
    with path.open('r', encoding='utf-8') as f:
        first = f.readline().strip()
        if not first.startswith('# shape:'):
            raise ValueError('invalid wts header: ' + str(path))
        vals = [int(x) for x in first[len('# shape:'):].split() if x.strip()]
        if len(vals) == 1:
            shape = (1, vals[0])
        elif len(vals) == 2:
            shape = tuple(vals)
        else:
            raise ValueError('unsupported shape in ' + str(path))
        data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.extend(float(x) for x in line.split())
        arr = np.array(data, dtype=np.float32)
        expected = int(np.prod(shape))
        if arr.size < expected:
            arr = np.pad(arr, (0, expected - arr.size), mode='constant')
        return arr.reshape(shape)


def write_wts(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        if arr.ndim == 1:
            f.write(f"# shape: {arr.shape[0]}\n")
            flat = arr.reshape(-1)
            for i in range(0, flat.size, 10):
                f.write(' '.join(f"{x:.8g}" for x in flat[i:i+10]) + '\n')
        elif arr.ndim == 2:
            f.write(f"# shape: {arr.shape[0]} {arr.shape[1]}\n")
            for r in range(arr.shape[0]):
                row = arr[r].reshape(-1)
                for i in range(0, row.size, 10):
                    f.write(' '.join(f"{x:.8g}" for x in row[i:i+10]) + '\n')
        else:
            # flatten leading dims into rows
            s0 = arr.shape[0]
            rest = int(np.prod(arr.shape[1:]))
            f.write(f"# shape: {s0} {rest}\n")
            flat = arr.reshape(s0, rest)
            for r in range(s0):
                row = flat[r]
                for i in range(0, row.size, 10):
                    f.write(' '.join(f"{x:.8g}" for x in row[i:i+10]) + '\n')


def load_weights_dict(weights_dir: Path):
    # mirror mnist_optimize.load_vit_weights but as numpy
    w = {}
    w['patch.weight'] = read_wts(weights_dir / 'patch.weight.wts')
    w['patch.bias'] = read_wts(weights_dir / 'patch.bias.wts')
    w['cls_token'] = read_wts(weights_dir / 'cls_token.wts').reshape(1,1,32)
    w['pos_embed'] = read_wts(weights_dir / 'pos_embed.wts').reshape(1,17,32)
    for i in range(2):
        prefix = weights_dir / f'blocks.{i}'
        def p(name):
            return read_wts(prefix.with_name(prefix.name + '.' + name + '.wts'))
        w[f'blocks.{i}.attn.q.weight'] = p('attn.q.weight')
        w[f'blocks.{i}.attn.q.bias'] = p('attn.q.bias')
        w[f'blocks.{i}.attn.k.weight'] = p('attn.k.weight')
        w[f'blocks.{i}.attn.k.bias'] = p('attn.k.bias')
        w[f'blocks.{i}.attn.v.weight'] = p('attn.v.weight')
        w[f'blocks.{i}.attn.v.bias'] = p('attn.v.bias')
        w[f'blocks.{i}.attn.o.weight'] = p('attn.o.weight')
        w[f'blocks.{i}.attn.o.bias'] = p('attn.o.bias')
        w[f'blocks.{i}.mlp.fc1.weight'] = p('mlp.fc1.weight')
        w[f'blocks.{i}.mlp.fc1.bias'] = p('mlp.fc1.bias')
        w[f'blocks.{i}.mlp.fc2.weight'] = p('mlp.fc2.weight')
        w[f'blocks.{i}.mlp.fc2.bias'] = p('mlp.fc2.bias')
        w[f'blocks.{i}.norm1.gamma'] = p('norm1.gamma')
        w[f'blocks.{i}.norm1.beta'] = p('norm1.beta')
        w[f'blocks.{i}.norm2.gamma'] = p('norm2.gamma')
        w[f'blocks.{i}.norm2.beta'] = p('norm2.beta')
    w['head.weight'] = read_wts(weights_dir / 'head.weight.wts')
    w['head.bias'] = read_wts(weights_dir / 'head.bias.wts')
    return w


def save_weights_dict(out_dir: Path, weights: dict):
    # Save with same file names
    write_wts(out_dir / 'patch.weight.wts', weights['patch.weight'])
    write_wts(out_dir / 'patch.bias.wts', weights['patch.bias'])
    write_wts(out_dir / 'cls_token.wts', weights['cls_token'].reshape(1,32))
    write_wts(out_dir / 'pos_embed.wts', weights['pos_embed'].reshape(17,32))
    for i in range(2):
        prefix = out_dir
        def w(name, arr):
            write_wts(prefix / f'blocks.{i}.{name}.wts', arr)
        w('attn.q.weight', weights[f'blocks.{i}.attn.q.weight'])
        w('attn.q.bias', weights[f'blocks.{i}.attn.q.bias'])
        w('attn.k.weight', weights[f'blocks.{i}.attn.k.weight'])
        w('attn.k.bias', weights[f'blocks.{i}.attn.k.bias'])
        w('attn.v.weight', weights[f'blocks.{i}.attn.v.weight'])
        w('attn.v.bias', weights[f'blocks.{i}.attn.v.bias'])
        w('attn.o.weight', weights[f'blocks.{i}.attn.o.weight'])
        w('attn.o.bias', weights[f'blocks.{i}.attn.o.bias'])
        w('mlp.fc1.weight', weights[f'blocks.{i}.mlp.fc1.weight'])
        w('mlp.fc1.bias', weights[f'blocks.{i}.mlp.fc1.bias'])
        w('mlp.fc2.weight', weights[f'blocks.{i}.mlp.fc2.weight'])
        w('mlp.fc2.bias', weights[f'blocks.{i}.mlp.fc2.bias'])
        w('norm1.gamma', weights[f'blocks.{i}.norm1.gamma'])
        w('norm1.beta', weights[f'blocks.{i}.norm1.beta'])
        w('norm2.gamma', weights[f'blocks.{i}.norm2.gamma'])
        w('norm2.beta', weights[f'blocks.{i}.norm2.beta'])
    write_wts(out_dir / 'head.weight.wts', weights['head.weight'])
    write_wts(out_dir / 'head.bias.wts', weights['head.bias'])


# PyTorch model definition
MODEL_CODE = r"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_ch, patch_size, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(patch_size*patch_size, hidden_dim)

    def forward(self, x):
        # x: B, H, W
        B, H, W = x.shape
        ph = pw = int(H/4) if False else 7
        patches = []
        for i in range(0, H, ph):
            for j in range(0, W, pw):
                p = x[:, i:i+ph, j:j+pw].contiguous().view(B, -1)
                patches.append(self.proj(p).unsqueeze(1))
        return patches

class MLP(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(0.5 * self.fc1(x) * (1.0 + torch.erf(self.fc1(x) / (2**0.5))))

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_dim)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

    def forward(self, x):
        B, N, C = x.shape
        h = self.norm1(x)
        q = self.q(h).view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k(h).view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.v(h).view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = F.softmax(scores, dim=-1)
        ctx = (att @ v).permute(0,2,1,3).contiguous().view(B, N, C)
        out = self.out(ctx)
        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, hidden_dim=32, num_heads=4, mlp_dim=64, num_layers=2, num_classes=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.cls_token = nn.Parameter(torch.zeros(1,1,hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,17,hidden_dim))
        self.patch_proj = nn.Linear(49, hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, mlp_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        patches = []
        for i in range(0,28,7):
            for j in range(0,28,7):
                p = x[:, i:i+7, j:j+7].contiguous().view(B, -1)
                patches.append(self.patch_proj(p).unsqueeze(1))
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(B, -1, -1)
        tokens = torch.cat([cls] + [p + self.pos_embed[:, idx+1:idx+2, :].expand(B, -1, -1) for idx, p in enumerate(patches)], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        cls_token = tokens[:,0]
        logits = self.head(cls_token)
        return logits
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', default='weights')
    parser.add_argument('--out-dir', default='weights_finetuned')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-samples', type=int, default=2000)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchvision
        import torchvision.transforms as transforms
    except Exception as e:
        print('需要安装 torch 与 torchvision 才能训练: pip install torch torchvision')
        raise

    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.out_dir)

    print('加载原始权重...')
    weights = load_weights_dict(weights_dir)

    print('构建 PyTorch 模型...')
    # create module from MODEL_CODE
    import importlib.util, types, tempfile
    spec = importlib.util.spec_from_loader('vt_model', loader=None)
    vt_mod = types.ModuleType('vt_model')
    exec(MODEL_CODE, vt_mod.__dict__)
    Vit = vt_mod.VisionTransformer

    device = torch.device(args.device)
    model = Vit().to(device)

    # initialize model weights from numpy weights
    def to_t(t):
        return torch.from_numpy(t).to(torch.float32)

    # patch projection expects (out=in_features?); original patch.weight had shape (49,32)
    # PyTorch patch_proj.weight shape (hidden_dim, 49)
    model.patch_proj.weight.data.copy_(to_t(weights['patch.weight']).T)
    model.patch_proj.bias.data.copy_(to_t(weights['patch.bias']).reshape(-1))

    model.cls_token.data.copy_(to_t(weights['cls_token'].reshape(1,1,32)))
    model.pos_embed.data.copy_(to_t(weights['pos_embed']))

    for i in range(2):
        blk = model.blocks[i]
        # qkv: weights are (32,32) per file; PyTorch linear weight is (out,in)
        blk.q.weight.data.copy_(to_t(weights[f'blocks.{i}.attn.q.weight']).T)
        blk.q.bias.data.copy_(to_t(weights[f'blocks.{i}.attn.q.bias']).reshape(-1))
        blk.k.weight.data.copy_(to_t(weights[f'blocks.{i}.attn.k.weight']).T)
        blk.k.bias.data.copy_(to_t(weights[f'blocks.{i}.attn.k.bias']).reshape(-1))
        blk.v.weight.data.copy_(to_t(weights[f'blocks.{i}.attn.v.weight']).T)
        blk.v.bias.data.copy_(to_t(weights[f'blocks.{i}.attn.v.bias']).reshape(-1))
        blk.out.weight.data.copy_(to_t(weights[f'blocks.{i}.attn.o.weight']).T)
        blk.out.bias.data.copy_(to_t(weights[f'blocks.{i}.attn.o.bias']).reshape(-1))

        # mlp
        blk.mlp.fc1.weight.data.copy_(to_t(weights[f'blocks.{i}.mlp.fc1.weight']).T)
        blk.mlp.fc1.bias.data.copy_(to_t(weights[f'blocks.{i}.mlp.fc1.bias']).reshape(-1))
        blk.mlp.fc2.weight.data.copy_(to_t(weights[f'blocks.{i}.mlp.fc2.weight']).T)
        blk.mlp.fc2.bias.data.copy_(to_t(weights[f'blocks.{i}.mlp.fc2.bias']).reshape(-1))

        # layernorm weights: gamma beta in numpy are shape (1,N) or (N,)
        try:
            blk.norm1.weight.data.copy_(to_t(weights[f'blocks.{i}.norm1.gamma']).reshape(-1))
            blk.norm1.bias.data.copy_(to_t(weights[f'blocks.{i}.norm1.beta']).reshape(-1))
            blk.norm2.weight.data.copy_(to_t(weights[f'blocks.{i}.norm2.gamma']).reshape(-1))
            blk.norm2.bias.data.copy_(to_t(weights[f'blocks.{i}.norm2.beta']).reshape(-1))
        except Exception:
            pass

    # head
    # original head.weight is (32,10); PyTorch head.weight is (10,32)
    model.head.weight.data.copy_(to_t(weights['head.weight']).T)
    model.head.bias.data.copy_(to_t(weights['head.bias']).reshape(-1))

    # dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
    if args.max_samples > 0:
        trainset.data = trainset.data[:args.max_samples]
        trainset.targets = trainset.targets[:args.max_samples]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('开始训练（小步进行）...')
    model.train()
    for epoch in range(args.epochs):
        running = 0
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device).squeeze(1)  # B,H,W
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running += loss.item()
            if i % 100 == 0:
                print(f'epoch {epoch+1}/{args.epochs} step {i} loss {running/(i+1):.4f} acc {correct/total:.4f}')

    print('训练完成，导出权重...')
    # extract back to numpy dict
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

    save_weights_dict(out_dir, new_w)
    print('已将微调后的权重保存到', out_dir)


if __name__ == '__main__':
    main()
