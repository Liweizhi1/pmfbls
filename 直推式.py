import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================================
# Part 1: Layer Activations (保持不变，用于提取多层特征)
# =========================================================================
class LayerActivations:
    def __init__(self, backbone, extract_indices):
        self.backbone = backbone
        self.extract_indices = extract_indices
        self.features = {}
        self.hooks = []
        self.layers_ref = None
        
        # 自动寻找 backbone 中的层容器
        for name in ['blocks', 'layers', 'features', 'module']:
            if hasattr(backbone, name):
                self.layers_ref = getattr(backbone, name)
                break
        if self.layers_ref is None:
            self.layers_ref = list(backbone.children())
        if len(self.layers_ref) == 1 and isinstance(self.layers_ref[0], nn.Sequential):
            self.layers_ref = self.layers_ref[0]
            
        total_layers = len(self.layers_ref)
        self.abs_indices = set()
        for idx in extract_indices:
            self.abs_indices.add(idx if idx >= 0 else total_layers + idx)
        self._register_hooks()

    def _register_hooks(self):
        for i, layer in enumerate(self.layers_ref):
            if i in self.abs_indices:
                h = layer.register_forward_hook(self._get_hook(i))
                self.hooks.append(h)

    def _get_hook(self, idx):
        def hook(module, input, output):
            self.features[idx] = output
        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def run(self, x):
        self.features = {}
        self.backbone.eval() 
        with torch.no_grad():
            _ = self.backbone(x)
        outputs = []
        total_layers = len(self.layers_ref)
        for idx in self.extract_indices:
            abs_idx = idx if idx >= 0 else total_layers + idx
            if abs_idx in self.features:
                outputs.append(self.features[abs_idx])
            else:
                outputs.append(list(self.features.values())[-1])
        return outputs

# =========================================================================
# Part 2: ProtoNet_MiniBLS_Ultimate (Step 2 & Step 3 Integrated)
# =========================================================================
class ProtoNet_MiniBLS_Ultimate(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone
        self.args = args

        # --- 核心配置 ---
        self.use_multi_layer = getattr(args, 'use_multi_layer', True)
        self.use_cov = getattr(args, 'use_cov', True)   # 建议开启
        self.use_tta = getattr(args, 'use_tta', True)
        self.use_bls = getattr(args, 'use_bls', True)   # 开启，但在Step 3中会被降权
        
        # [Step 3 Config] ISIC/ChestX 建议 64，保留更多纹理细节
        self.cov_proj_dim = getattr(args, 'cov_proj_dim', 64) 

        # 层选择
        if self.use_multi_layer:
            layer_str = getattr(args, 'mini_bls_mlf_layers', getattr(args, 'bls_layers', '-2,-5,-8'))
            self.mlf_layers = [int(x) for x in layer_str.split(',') if x.strip()]
        else:
            self.mlf_layers = [-1]

        self.extractor = LayerActivations(backbone, self.mlf_layers)
        
        # 自动探测特征维度
        with torch.no_grad():
            # 兼容 CPU/GPU
            device = next(backbone.parameters()).device
            dummy = torch.zeros(1, 3, 224, 224).to(device)
            feats = self.extractor.run(dummy)
            self.token_dim = feats[0].shape[-1] 
            
        # --- Module 1: Robust Covariance (Orthogonal Projection) ---
        if self.use_cov:
            self.cov_reducer = nn.Parameter(
                torch.empty(self.token_dim, self.cov_proj_dim), 
                requires_grad=False
            )
            nn.init.orthogonal_(self.cov_reducer)
        else:
            self.cov_reducer = None

        # --- Module 2: BLS (Random Registers) ---
        self.bls_width = getattr(args, 'mini_bls_mapping_dim', getattr(args, 'bls_width', 768))
        if self.use_bls:
            self.w_bls = nn.Parameter(
                torch.randn(self.token_dim, self.bls_width) * (1.0 / math.sqrt(self.token_dim)),
                requires_grad=False
            )
            self.b_bls = nn.Parameter(torch.zeros(self.bls_width), requires_grad=False)
        else:
            self.w_bls = None

        # Regression Regularization
        self.reg_lambda = getattr(args, 'mini_bls_reg_lambda', getattr(args, 'reg_lambda', 0.1))
        self.scale_cls = 20.0 

        # =========================================================
        # [Step 3 核心] 自适应加权 (Adaptive Weighting Strategy)
        # 既然第一步拼接是狗屎，这里我们手动控制信噪比
        # =========================================================
        # GAP: 主干特征，最稳定，权重最高 (Anchor)
        self.weight_gap = 1.0 
        # COV: 纹理特征，ISIC很有用，权重次之
        self.weight_cov = 0.8 
        # BLS: 随机映射，作为微扰动 (Perturbation)，权重必须低，否则淹没主信号
        self.weight_bls = 0.1 

    def _augment_image(self, x):
        """Standard 5-Crop / Flip TTA"""
        if not self.use_tta:
            return x, 1
        B_N, C, H, W = x.shape
        # 5 views: Original, Flip, Rot90, Rot-90, CenterCrop-Resize
        aug0 = x
        aug1 = torch.flip(x, dims=[3])
        aug2 = torch.rot90(x, k=1, dims=[2, 3])
        aug3 = torch.rot90(x, k=-1, dims=[2, 3])
        
        pad = 40
        if H > pad*2:
            crop = x[:, :, pad:-pad, pad:-pad]
            aug4 = F.interpolate(crop, size=(H, W), mode='bilinear', align_corners=False)
        else:
            aug4 = x 
            
        x_aug = torch.stack([aug0, aug1, aug2, aug3, aug4], dim=1).view(-1, C, H, W)
        return x_aug, 5

    def _matrix_sqrt(self, x, iter_n=3):
        """Newton-Schulz Iteration for Matrix Square Root"""
        dtype = x.dtype
        device = x.device
        norm = torch.linalg.norm(x, dim=(1,2), keepdim=True)
        Y = x / (norm + 1e-8)
        I = torch.eye(x.size(1), device=device, dtype=dtype).unsqueeze(0).expand_as(x)
        Z = torch.eye(x.size(1), device=device, dtype=dtype).unsqueeze(0).expand_as(x)
        
        for _ in range(iter_n):
            T = 0.5 * (3.0 * I - torch.bmm(Z, Y))
            Y = torch.bmm(Y, T)
            Z = torch.bmm(T, Z)
        
        return Y * torch.sqrt(norm + 1e-8)

    def _compute_robust_covariance(self, patch_tokens):
        """MPN-COV Style: Projection -> Cov -> Matrix Sqrt -> Triu Flatten"""
        # 1. Orthogonal Projection
        x = torch.matmul(patch_tokens, self.cov_reducer)
        x = x - x.mean(dim=1, keepdim=True)
        
        # 2. Covariance Calculation
        denom = max(1, x.size(1) - 1)
        cov = torch.matmul(x.transpose(1, 2), x) / denom
        
        # 3. Regularization
        cov = cov + 1e-5 * torch.eye(cov.size(-1), device=cov.device).unsqueeze(0)
        
        # 4. Matrix Sqrt
        cov_sqrt = self._matrix_sqrt(cov)
        
        # 5. Upper Triangular Flatten
        d = cov_sqrt.size(1)
        idx = torch.triu_indices(d, d, device=cov_sqrt.device)
        cov_flat = cov_sqrt[:, idx[0], idx[1]]
        
        return cov_flat

    def _get_feature_components(self, feats_list):
        """分解提取步骤，返回 GAP, COV, BLS 三个独立分量"""
        results = []
        for feat in feats_list:
            comp = {}
            if feat.dim() == 4: 
                feat = feat.flatten(2).transpose(1, 2) # [N, L, C]
            
            # --- GAP Component ---
            if feat.shape[1] > 1:
                patch_tokens = feat
                gap = feat.mean(dim=1)
            else:
                patch_tokens = feat
                gap = feat.squeeze(1)
            
            comp['gap'] = gap

            # --- Covariance Component ---
            if self.use_cov and self.cov_reducer is not None:
                comp['cov'] = self._compute_robust_covariance(patch_tokens)
            
            # --- BLS Component ---
            if self.use_bls and self.w_bls is not None:
                bls = torch.tanh(patch_tokens @ self.w_bls + self.b_bls).mean(dim=1)
                comp['bls'] = bls
            
            results.append(comp)
        return results

    def _tukey_transform(self, x, beta=0.5):
        """Tukey's Power Transformation"""
        return torch.sign(x) * torch.pow(torch.abs(x) + 1e-8, beta)

    # =====================================================================
    # [Step 3 实现] 带权重的特征融合
    # =====================================================================
    def fuse_and_pool_weighted(self, comp_list, n_views):
        processed_parts = []
        
        # 1. GAP (High Weight)
        gap_raw = torch.cat([layer['gap'] for layer in comp_list], dim=-1)
        gap_pool = gap_raw.view(-1, n_views, gap_raw.shape[-1]).mean(dim=1)
        gap_trans = self._tukey_transform(gap_pool, beta=0.5)
        gap_norm = F.normalize(gap_trans, dim=-1)
        processed_parts.append(gap_norm * self.weight_gap) # <--- Apply Weight

        # 2. Cov (Medium Weight)
        if self.use_cov:
            cov_raw = torch.cat([layer['cov'] for layer in comp_list], dim=-1)
            cov_pool = cov_raw.view(-1, n_views, cov_raw.shape[-1]).mean(dim=1)
            cov_trans = self._tukey_transform(cov_pool, beta=0.5)
            cov_norm = F.normalize(cov_trans, dim=-1)
            processed_parts.append(cov_norm * self.weight_cov) # <--- Apply Weight

        # 3. BLS (Low Weight - Perturbation only)
        if self.use_bls:
            bls_raw = torch.cat([layer['bls'] for layer in comp_list], dim=-1)
            bls_pool = bls_raw.view(-1, n_views, bls_raw.shape[-1]).mean(dim=1)
            bls_trans = self._tukey_transform(bls_pool, beta=0.5)
            bls_norm = F.normalize(bls_trans, dim=-1)
            processed_parts.append(bls_norm * self.weight_bls) # <--- Apply Low Weight

        # Concatenate weighted parts
        return torch.cat(processed_parts, dim=-1)

    def forward(self, x_sup, y_sup, x_qry):
        B = x_sup.shape[0]
        n_sup = x_sup.shape[1]
        n_qry = x_qry.shape[1]

        x_sup_flat = x_sup.view(-1, *x_sup.shape[-3:])
        x_qry_flat = x_qry.view(-1, *x_qry.shape[-3:])
        
        # 1. TTA Augmentation
        x_sup_aug, n_views_sup = self._augment_image(x_sup_flat)
        x_qry_aug, n_views_qry = self._augment_image(x_qry_flat)

        # 2. Extract Raw Features
        feats_sup = self.extractor.run(x_sup_aug)
        feats_qry = self.extractor.run(x_qry_aug)

        # 3. Process Components
        comps_sup = self._get_feature_components(feats_sup)
        comps_qry = self._get_feature_components(feats_qry)

        # 4. Weighted Fusion (Step 3 Applied)
        z_sup = self.fuse_and_pool_weighted(comps_sup, n_views_sup) # [B*N_sup, D_total]
        z_qry = self.fuse_and_pool_weighted(comps_qry, n_views_qry) # [B*N_qry, D_total]

        z_sup = z_sup.view(B, n_sup, -1)
        z_qry = z_qry.view(B, n_qry, -1)
        
        # =================================================================
        # [Step 2 核心] 跨域分布校准 (Distribution Calibration)
        # =================================================================
        # 计算 Support 的统计量 (Mean & Std)
        # [B, n_sup, D] -> [B, 1, D]
        mu_sup = z_sup.mean(dim=1, keepdim=True)
        std_sup = z_sup.std(dim=1, keepdim=True) + 1e-8

        # 计算 Query 的统计量
        mu_qry = z_qry.mean(dim=1, keepdim=True)
        std_qry = z_qry.std(dim=1, keepdim=True) + 1e-8
        
        # [Calibration] 将 Query 分布强行对齐到 Support 分布
        # 这一步是解决跨域（ImageNet -> ISIC）分布漂移的关键
        z_qry = (z_qry - mu_qry) / std_qry * std_sup + mu_sup
        
        # 5. Inductive Centering
        # 既然分布已对齐，现在统一减去 Support 中心
        z_sup = z_sup - mu_sup
        z_qry = z_qry - mu_sup 
        
        # Final Normalize (再次归一化确保 Ridge 数值稳定)
        z_sup = F.normalize(z_sup, dim=-1)
        z_qry = F.normalize(z_qry, dim=-1)

        # 6. Classification (Adaptive Kernel Ridge Regression)
        if y_sup.dim() == 3: y_sup = y_sup.flatten(1)
        num_classes = int(y_sup.max().item() + 1)
        Y_onehot = F.one_hot(y_sup.long(), num_classes=num_classes).float()

        # Adaptive Lambda: 维度越高，正则化越强
        d_feat = z_sup.size(-1)
        curr_lambda = self.reg_lambda * (d_feat / 500.0) # Scaled lambda

        # Solve: W = (Z^T Z + lambda I)^-1 Z^T Y
        # 使用 Kernel Trick: K = Z Z^T
        K = torch.bmm(z_sup, z_sup.transpose(1, 2)) # [B, N_sup, N_sup]
        I = torch.eye(n_sup, device=z_sup.device).unsqueeze(0)
        
        alpha = torch.linalg.solve(K + curr_lambda * I, Y_onehot)

        # Query Predictions
        sim = torch.bmm(z_qry, z_sup.transpose(1, 2))
        logits = torch.bmm(sim, alpha)

        return logits * self.scale_cls