# ==========================================
# 文件名: bls_ultimate_ops.py
# 作用: 特征提取 + 协方差 + 幂次变换 (CVPR级预处理)
# ==========================================
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerActivations:
    def __init__(self, backbone, extract_indices):
        self.backbone = backbone
        self.features = {}
        self.hooks = []
        
        # 自动探测层级
        self.layers_ref = None
        for name in ["blocks", "layers", "features"]:
            if hasattr(backbone, name):
                self.layers_ref = getattr(backbone, name)
                break
        if self.layers_ref is None:
            self.layers_ref = list(backbone.children())
            if len(self.layers_ref) == 1 and isinstance(self.layers_ref[0], nn.Sequential):
                self.layers_ref = self.layers_ref[0]

        # 索引对齐
        total_layers = len(self.layers_ref)
        self.abs_indices = set()
        for idx in extract_indices:
            abs_idx = idx if idx >= 0 else total_layers + idx
            if 0 <= abs_idx < total_layers:
                self.abs_indices.add(abs_idx)
        
        self._register_hooks()

    def _register_hooks(self):
        for i, layer in enumerate(self.layers_ref):
            if i in self.abs_indices:
                h = layer.register_forward_hook(self._get_hook(i))
                self.hooks.append(h)

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            self.features[layer_idx] = out
        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

# =========================================================
# 新增: 幂次变换 (Power Transform)
# 作用: 也就是 Tukey's Ladder，强制特征高斯化，解决分布偏斜问题
# =========================================================
class PowerTransform(nn.Module):
    def __init__(self, power=0.5):
        super().__init__()
        self.power = power

    def forward(self, x):
        # sign(x) * |x|^beta
        return torch.sign(x) * torch.pow(torch.abs(x) + 1e-12, self.power)

# =========================================================
# 协方差描述符 (优化数值稳定性版)
# =========================================================
class CovarianceDescriptor(nn.Module):
    def __init__(self, in_dim, reduce_dim=64):
        super().__init__()
        # 使用 1x1 卷积降维，减少参数量
        self.reducer = nn.Conv2d(in_dim, reduce_dim, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.reducer.weight) # 正交初始化
        # 冻结参数，作为随机投影层
        for param in self.reducer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 1. 形状适配
        if x.dim() == 3: # ViT [B, N, C]
            B, N, C = x.shape
            H = int((N-1)**0.5) 
            if H*H == N-1: x = x[:, 1:, :].permute(0, 2, 1).view(B, C, H, H)
            elif H*H == N: x = x.permute(0, 2, 1).view(B, C, H, H)
            else: return x.mean(dim=1) # Fallback

        # 2. 降维
        x_red = self.reducer(x) # [B, 64, H, W]
        B, C, H, W = x_red.shape
        x_flat = x_red.view(B, C, -1) # [B, 64, N]
        
        # 3. 实例去均值 (Instance Centering)
        mean = x_flat.mean(dim=2, keepdim=True)
        x_centered = x_flat - mean
        
        # 4. 计算协方差
        # 加上 1e-5 防止除以零
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (H*W - 1 + 1e-5)
        
        # 5. 矩阵对数/幂次归一化 (Matrix Power Normalization)
        # 这是 DeepBDC 的精髓，让二阶特征更鲁棒
        cov = torch.sign(cov) * torch.sqrt(torch.abs(cov) + 1e-12)
        
        return cov.view(B, -1)