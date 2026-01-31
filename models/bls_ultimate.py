import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================================
# Part 1: Layer Activations (保持高效提取)
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
                # Fallback if layer index not found
                outputs.append(list(self.features.values())[-1])
        return outputs

# =========================================================================
# Part 2: ProtoNet_MiniBLS_Ultimate_V2 (SOTA 级优化)
# =========================================================================
class ProtoNet_MiniBLS_Ultimate(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone
        self.args = args

        # --- 配置 ---
        self.use_multi_layer = getattr(args, 'use_multi_layer', True)
        self.use_cov = getattr(args, 'use_cov', True)
        self.use_tta = getattr(args, 'use_tta', True)
        self.use_bls = getattr(args, 'use_bls', True)
        
        # [Config] 协方差降维维度
        # 32 或 64 是比较好的折中，ISIC/ChestX 推荐 64，EuroSAT 推荐 32
        self.cov_proj_dim = getattr(args, 'cov_proj_dim', 64) 

        # 层选择
        if self.use_multi_layer:
            # 默认取最后几层，捕捉不同尺度的纹理
            layer_str = getattr(args, 'mini_bls_mlf_layers', getattr(args, 'bls_layers', '-2,-5,-8'))
            self.mlf_layers = [int(x) for x in layer_str.split(',') if x.strip()]
        else:
            self.mlf_layers = [-1]

        self.extractor = LayerActivations(backbone, self.mlf_layers)
        
        # 探测特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            if next(backbone.parameters()).is_cuda: dummy = dummy.cuda()
            feats = self.extractor.run(dummy)
            self.token_dim = feats[0].shape[-1] 
            
        # --- Module 1: Robust Covariance (Orthogonal Projection) ---
        if self.use_cov:
            # [Optimization] 使用正交初始化，最大化保留方差信息
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
            aug4 = x # fallback if image too small
            
        x_aug = torch.stack([aug0, aug1, aug2, aug3, aug4], dim=1).view(-1, C, H, W)
        return x_aug, 5

    def _matrix_sqrt(self, x, iter_n=3):
        """
        [Optimization] Newton-Schulz Iteration for Matrix Square Root.
        比简单的 Power Transform 更符合黎曼几何，用于 Covariance Pooling。
        """
        # x: [N, D, D]
        dtype = x.dtype
        device = x.device
        
        # Normalize spectral norm to ensure convergence
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
        """
        MPN-COV Style: Projection -> Cov -> Matrix Sqrt -> Triu Flatten
        """
        # 1. Orthogonal Projection: [N, L, D] -> [N, L, D_red]
        x = torch.matmul(patch_tokens, self.cov_reducer)
        x = x - x.mean(dim=1, keepdim=True)
        
        # 2. Covariance Calculation: [N, D_red, D_red]
        denom = max(1, x.size(1) - 1)
        cov = torch.matmul(x.transpose(1, 2), x) / denom
        
        # 3. Regularization (Conditioning)
        cov = cov + 1e-5 * torch.eye(cov.size(-1), device=cov.device).unsqueeze(0)
        
        # 4. Matrix Sqrt (DeepBDC / MPN-COV Core)
        cov_sqrt = self._matrix_sqrt(cov)
        
        # 5. Upper Triangular Flatten (Remove redundant symmetric parts)
        # indices for upper triangle
        d = cov_sqrt.size(1)
        idx = torch.triu_indices(d, d, device=cov_sqrt.device)
        cov_flat = cov_sqrt[:, idx[0], idx[1]]
        
        return cov_flat

    def _get_feature_components(self, feats_list):
        """
        分解提取步骤，返回 GAP, COV, BLS 三个独立分量
        """
        results = []
        for feat in feats_list:
            comp = {}
            if feat.dim() == 4: 
                feat = feat.flatten(2).transpose(1, 2) # [N, L, C]
            
            # --- GAP Component ---
            if feat.shape[1] > 1:
                # Exclude CLS token if present (usually index 0) for patch stats
                # ViT usually has cls token at 0. ResNet doesn't. 
                # 这里假设纯 patch 或 CNN feature map
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
                # Random projection + Non-linearity
                bls = torch.tanh(patch_tokens @ self.w_bls + self.b_bls).mean(dim=1)
                comp['bls'] = bls
            
            results.append(comp)
        return results

    def _tukey_transform(self, x, beta=0.5):
        """
        [Optimization] Tukey's Power Transformation
        将 Cross-Domain 的偏态分布修正为类高斯分布
        """
        return torch.sign(x) * torch.pow(torch.abs(x) + 1e-8, beta)

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

        # 3. Process into Components (GAP, Cov, BLS)
        # Returns: List[Dict] over layers
        comps_sup = self._get_feature_components(feats_sup)
        comps_qry = self._get_feature_components(feats_qry)

        # 4. Intelligent Fusion & Pooling
        # 策略：先在各自分量上做 Pooling 和 Tukey 变换，最后再拼接
        # 这样避免了量纲不同导致的融合问题
        
        def fuse_and_pool(comp_list, n_views):
            # comp_list: [Layer1_Dict, Layer2_Dict, ...]
            # 我们将所有层的所有分量拼接
            
            processed_parts = []
            keys = comp_list[0].keys() # ['gap', 'cov', 'bls']
            
            for key in keys:
                # Concatenate across layers for this component type
                # e.g., GAP from layer -1, -2, -3 -> Concat
                raw_part = torch.cat([layer[key] for layer in comp_list], dim=-1)
                
                # TTA Mean Pooling
                # [B*N*Views, Dim] -> [B*N, Views, Dim] -> [B*N, Dim]
                raw_part = raw_part.view(B * -1, n_views, raw_part.shape[-1]).mean(dim=1)
                
                # [Optimization] Tukey Transform per component
                # 这种分量级的变换比全局变换更精细
                trans_part = self._tukey_transform(raw_part, beta=0.5)
                
                # [Optimization] L2 Normalize per component
                # 确保 GAP, Cov, BLS 在拼接前具有相同的“能量”
                norm_part = F.normalize(trans_part, dim=-1)
                
                processed_parts.append(norm_part)
            
            # Final Concatenation
            return torch.cat(processed_parts, dim=-1)

        z_sup = fuse_and_pool(comps_sup, n_views_sup) # [B*N_sup, D_total]
        z_qry = fuse_and_pool(comps_qry, n_views_qry) # [B*N_qry, D_total]

        # 5. Inductive Centering
        # 此时特征已经经过 Tukey 变换，分布更接近高斯，减均值才有效
        z_sup = z_sup.view(B, n_sup, -1)
        z_qry = z_qry.view(B, n_qry, -1)
        
        mu_sup = z_sup.mean(dim=1, keepdim=True)
        z_sup = z_sup - mu_sup
        z_qry = z_qry - mu_sup # Inductive: Query uses Support stats
        
        # Final Normalize
        z_sup = F.normalize(z_sup, dim=-1)
        z_qry = F.normalize(z_qry, dim=-1)

        # 6. Classification (Ridge Regression)
        if y_sup.dim() == 3: y_sup = y_sup.flatten(1)
        num_classes = int(y_sup.max().item() + 1)
        Y_onehot = F.one_hot(y_sup.long(), num_classes=num_classes).float()

        # Dynamic Lambda adjustment based on Feature Dimension
        # 维度越高，正则化强度应稍大
        d_feat = z_sup.size(-1)
        curr_lambda = self.reg_lambda * (d_feat / 1000.0) # Adaptive scaling

        # Solve: W = (Z^T Z + lambda I)^-1 Z^T Y
        # Efficient trick: Use Woodbury identity or Kernel trick K = Z Z^T
        # Here we use Kernel trick as N << D
        K = torch.bmm(z_sup, z_sup.transpose(1, 2)) # [B, N_sup, N_sup]
        I = torch.eye(n_sup, device=z_sup.device).unsqueeze(0)
        
        alpha = torch.linalg.solve(K + curr_lambda * I, Y_onehot)

        # Query Predictions
        sim = torch.bmm(z_qry, z_sup.transpose(1, 2))
        logits = torch.bmm(sim, alpha)

        return logits * self.scale_cls