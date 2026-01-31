import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple


# =========================================================================
# Part 1: Layer Activations（只 hook 需要的层，保持高效率）
# =========================================================================
class LayerActivations:
    def __init__(self, backbone, extract_indices):
        self.backbone = backbone
        self.extract_indices = extract_indices
        self.features = {}
        self.hooks = []
        self.layers_ref = None

        # 尽量兼容 ViT / CNN 不同结构
        for name in ["blocks", "layers", "features", "module"]:
            if hasattr(backbone, name):
                self.layers_ref = getattr(backbone, name)
                break

        if self.layers_ref is None:
            self.layers_ref = list(backbone.children())

        # children() 返回 [Sequential] 的情况展开一层
        if isinstance(self.layers_ref, list) and len(self.layers_ref) == 1 and isinstance(self.layers_ref[0], nn.Sequential):
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
        def hook(_module, _input, output):
            self.features[idx] = output
        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

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
                # fallback：取最后一个捕获到的
                outputs.append(list(self.features.values())[-1])
        return outputs


# =========================================================================
# Part 2: ProtoNet_MiniBLS_Ultimate（加入 SACC：Support→Canonical 对齐）
# =========================================================================
class ProtoNet_MiniBLS_Ultimate(nn.Module):
    """
    现有主干：
      - 多层融合（MLF）
      - GAP 自适应门控（patch vs cls）
      - (可选) support-only 标准化
      - (可选) support 驱动 TTA 视角加权/TopK
      - Ridge closed-form

    新增（SACC / Diag-SACC）：
      - 对每个 (layer l, component k) 做 support→canonical 的对角白化-重着色
      - canonical bank 离线统计的 mu0/var0
      - shrinkage 自适应（shot 越少越保守）
      - 完整开关 + debug（避免“悄悄 fallback 导致没变化”）
    """

    # -------------------- 初始化 --------------------
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone
        self.args = args

        # 原始开关
        self.use_multi_layer = getattr(args, "use_multi_layer", True)
        self.use_cov = getattr(args, "use_cov", True)
        self.use_tta = getattr(args, "use_tta", True)
        self.use_bls = getattr(args, "use_bls", True)

        # ViT-like 判断
        self.is_vit_like = hasattr(backbone, "cls_token") or hasattr(backbone, "pos_embed") or hasattr(backbone, "patch_embed")

        # 一键自适应增强（默认不强制改变你基线；你想开再开）
        self.mini_bls_auto = bool(getattr(args, "mini_bls_auto", False))

        # cov/bls统计是否排除CLS
        self.exclude_cls_for_stats = getattr(args, "exclude_cls_for_stats", True)

        # GAP策略：allmean / patch / cls / concat / adaptive_gated
        self.gap_mode = getattr(args, "gap_mode", "allmean")
        self.gap_gate_temp = float(getattr(args, "gap_gate_temp", 1.0))

        # Tukey beta
        self.tukey_beta_gap = float(getattr(args, "tukey_beta_gap", 0.5))
        self.tukey_beta_cov = float(getattr(args, "tukey_beta_cov", 0.5))
        self.tukey_beta_bls = float(getattr(args, "tukey_beta_bls", 0.5))

        # support-only 标准化（归纳式）
        self.use_support_standardize = getattr(args, "use_support_standardize", False)
        self.calib_shrinkage = float(getattr(args, "calib_shrinkage", 0.0))  # <0 自动
        self.calib_eps = float(getattr(args, "calib_eps", 1e-6))

        # ===== TTA 视角池化方式 =====
        self.tta_pool_mode = getattr(args, "tta_pool_mode", "mean")  # mean / support_weighted / support_topk
        self.tta_weight_temp = float(getattr(args, "tta_weight_temp", 1.0))
        self.tta_topk = int(getattr(args, "tta_topk", 2))

        # cov参数
        self.cov_proj_dim = int(getattr(args, "cov_proj_dim", 64))
        self.cov_sqrt_iter = int(getattr(args, "cov_sqrt_iter", 3))
        self.cov_eps = float(getattr(args, "cov_eps", 1e-5))

        # =========================
        # SACC（Support→Canonical）模块开关（重点）
        # =========================
        self.use_sacc = bool(getattr(args, "use_sacc", False))             # 总开关：默认关，避免破坏基线
        self.sacc_mode = str(getattr(args, "sacc_mode", "diag")).lower()   # 当前实现：diag
        self.sacc_bank_path = str(getattr(args, "sacc_bank_path", ""))     # canonical_stats.pt
        self.sacc_strict = bool(getattr(args, "sacc_strict", False))       # True: 缺 key/维度错直接报错
        self.sacc_eps = float(getattr(args, "sacc_eps", 1e-6))             # 白化/重着色 eps
        self.sacc_shrinkage = float(getattr(args, "sacc_shrinkage", -1.0)) # <0 自动；>=0 固定
        self.sacc_max_rho = float(getattr(args, "sacc_max_rho", 0.25))     # 自动 rho 上限（保守）
        self.sacc_use_views_for_stats = bool(getattr(args, "sacc_use_views_for_stats", True))  # support统计是否利用view维度

        # 对各分量单独开关（便于消融）
        self.sacc_on_gap = bool(getattr(args, "sacc_on_gap", True))
        self.sacc_on_cov = bool(getattr(args, "sacc_on_cov", True))
        self.sacc_on_bls = bool(getattr(args, "sacc_on_bls", True))
        self.sacc_on_gap_patch = bool(getattr(args, "sacc_on_gap_patch", True))
        self.sacc_on_gap_cls = bool(getattr(args, "sacc_on_gap_cls", True))

        # bank缺失时回退策略：
        #   identity：啥也不做（最安全）
        #   center：仅去 support 均值
        #   center_scale：去均值 + 按对角方差白化到 unit（无 canonical）
        self.sacc_fallback = str(getattr(args, "sacc_fallback", "identity")).lower()

        # SACC 之后是否还做 post-centering（默认保持你原行为 support-centering）
        #   support：保持原行为
        #   none：不做（便于验证 SACC 的 mu0 贡献）
        self.post_centering = str(getattr(args, "post_centering", "support")).lower()

        # ===== SACC Debug =====
        self.sacc_debug = bool(getattr(args, "sacc_debug", False))
        self.sacc_debug_interval = int(getattr(args, "sacc_debug_interval", 50))
        self._sacc_debug_step = 0
        self._sacc_used = 0
        self._sacc_missed = 0
        self._sacc_rho_sum = 0.0
        self._sacc_scale_abs_sum = 0.0
        self._sacc_scale_cnt = 0
        self._sacc_warned_no_bank = False

        # =========================
        # 选层
        # =========================
        if self.use_multi_layer:
            layer_str = getattr(args, "mini_bls_mlf_layers", getattr(args, "bls_layers", "-2,-5,-8"))
            self.mlf_layers = [int(x) for x in layer_str.split(",") if x.strip()]
        else:
            self.mlf_layers = [-1]

        self.extractor = LayerActivations(backbone, self.mlf_layers)

        # 探测 token_dim（修复 CNN/ViT 兼容）
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            try:
                if next(backbone.parameters()).is_cuda:
                    dummy = dummy.cuda()
            except StopIteration:
                pass

            feats = self.extractor.run(dummy)
            f0 = feats[0]
            if f0.dim() == 4:
                self.token_dim = int(f0.shape[1])   # [N,C,H,W] -> C
            elif f0.dim() in (2, 3):
                self.token_dim = int(f0.shape[-1])  # [N,D] or [N,L,D]
            else:
                raise ValueError(f"无法识别的特征形状：{tuple(f0.shape)}")

        # COV（固定不训练）
        if self.use_cov:
            self.cov_reducer = nn.Parameter(torch.empty(self.token_dim, self.cov_proj_dim), requires_grad=False)
            nn.init.orthogonal_(self.cov_reducer)
        else:
            self.cov_reducer = None

        # BLS（固定不训练）
        self.bls_width = int(getattr(args, "mini_bls_mapping_dim", getattr(args, "bls_width", 768)))
        if self.use_bls:
            self.w_bls = nn.Parameter(
                torch.randn(self.token_dim, self.bls_width) * (1.0 / math.sqrt(self.token_dim)),
                requires_grad=False,
            )
            self.b_bls = nn.Parameter(torch.zeros(self.bls_width), requires_grad=False)
        else:
            self.w_bls = None
            self.b_bls = None

        # Ridge
        self.reg_lambda = float(getattr(args, "mini_bls_reg_lambda", getattr(args, "reg_lambda", 0.1)))
        self.lambda_mode = getattr(args, "lambda_mode", "dim_scaled")  # fixed/dim_scaled/trace_scaled
        self.scale_cls = float(getattr(args, "scale_cls", 20.0))

        # ===== 加载 canonical bank（只在 init 做一次）=====
        self.canonical_bank: Dict[str, Any] = {}
        if self.use_sacc and self.sacc_bank_path:
            self._load_canonical_bank(self.sacc_bank_path, strict=self.sacc_strict)

        # 一键 auto（可选：你自己决定开不开）
        if self.mini_bls_auto:
            if self.is_vit_like:
                self.exclude_cls_for_stats = True
                self.gap_mode = "adaptive_gated"
            self.lambda_mode = "trace_scaled"
            self.use_support_standardize = True
            self.calib_shrinkage = -1.0
            self.tta_pool_mode = "support_weighted"
            # 不强制开启 SACC，避免改变你默认基线
            # 你要开：args.use_sacc=True 并提供 bank

    # -------------------- SACC: bank 处理 --------------------
    def _load_canonical_bank(self, path: str, strict: bool = False) -> None:
        try:
            bank = torch.load(path, map_location="cpu")
            if not isinstance(bank, dict):
                raise ValueError("canonical bank 必须是 dict")
            self.canonical_bank = bank
        except Exception:
            if strict:
                raise
            self.canonical_bank = {}

    @staticmethod
    def _layer_key(layer_idx: int) -> str:
        return f"layer_{int(layer_idx)}"

    def _bank_get_mu_var(
        self, comp: str, layer_idx: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        期望 bank 结构：
          bank[comp][layer_key]["mu"] : [D]
          bank[comp][layer_key]["var"]: [D]
        comp: gap / gap_patch / gap_cls / cov / bls
        """
        if not self.canonical_bank:
            return None, None

        lk = self._layer_key(layer_idx)
        try:
            d = self.canonical_bank[comp][lk]
            mu0 = d["mu"]
            var0 = d["var"]
        except Exception:
            if self.sacc_strict:
                raise KeyError(f"canonical bank 缺失: comp={comp}, {lk}")
            return None, None

        if not torch.is_tensor(mu0) or not torch.is_tensor(var0):
            if self.sacc_strict:
                raise TypeError(f"canonical bank 中 mu/var 必须是 Tensor: comp={comp}, {lk}")
            return None, None

        if mu0.numel() != dim or var0.numel() != dim:
            if self.sacc_strict:
                raise ValueError(f"canonical bank 维度不匹配: comp={comp}, {lk}, bank_dim={mu0.numel()}, need_dim={dim}")
            return None, None

        mu0 = mu0.to(device=device, dtype=dtype).view(1, 1, 1, dim)
        var0 = var0.to(device=device, dtype=dtype).view(1, 1, 1, dim)
        return mu0, var0

    def _auto_rho(self, n_support: int) -> float:
        if n_support <= 0:
            return self.sacc_max_rho
        return float(min(self.sacc_max_rho, 2.0 / max(1.0, float(n_support))))

    def _diag_sacc_pair(
        self,
        sup_view: torch.Tensor,   # [B,Ns,V,D]
        qry_view: torch.Tensor,   # [B,Nq,V,D]
        mu0: Optional[torch.Tensor],   # [1,1,1,D]
        var0: Optional[torch.Tensor],  # [1,1,1,D]
        n_support: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        diag SACC:
          x' = sqrt(var0+eps) * rsqrt(var_s+eps) * (x - mu_s) + mu0

        var_s shrinkage:
          var_s = (1-rho)*var_hat + rho*var0
        """
        assert sup_view.dim() == 4 and qry_view.dim() == 4, "SACC 输入必须是 [B,N,V,D]"
        B, Ns, Vs, D = sup_view.shape

        # ---- 用 support 估计 mu_s / var_hat（对角）----
        if self.sacc_use_views_for_stats:
            xs = sup_view.reshape(B, Ns * Vs, D)
        else:
            xs = sup_view.mean(dim=2)  # [B,Ns,D]

        mu_s = xs.mean(dim=1, keepdim=True)  # [B,1,D]
        var_hat = (xs - mu_s).pow(2).mean(dim=1, keepdim=True).clamp_min(0.0)  # [B,1,D]

        mu_s4 = mu_s.view(B, 1, 1, D)
        var_hat4 = var_hat.view(B, 1, 1, D)

        # ---- 无 canonical：回退 ----
        if (mu0 is None) or (var0 is None):
            fb = self.sacc_fallback
            if fb == "center":
                return sup_view - mu_s4, qry_view - mu_s4
            if fb == "center_scale":
                inv_std = torch.rsqrt(var_hat4 + self.sacc_eps)
                return (sup_view - mu_s4) * inv_std, (qry_view - mu_s4) * inv_std
            return sup_view, qry_view

        # ---- shrinkage ----
        rho = self.sacc_shrinkage
        if rho < 0:
            rho = self._auto_rho(n_support)
        rho = float(max(0.0, min(1.0, rho)))

        var_s = (1.0 - rho) * var_hat4 + rho * var0
        var_s = var_s.clamp_min(0.0)

        # ---- white + recolor ----
        scale = torch.sqrt(var0 + self.sacc_eps) * torch.rsqrt(var_s + self.sacc_eps)  # [B,1,1,D]
        sup_out = (sup_view - mu_s4) * scale + mu0
        qry_out = (qry_view - mu_s4) * scale + mu0

        # ---- debug 统计（只统计一次 scale 的幅度，不影响结果）----
        if self.sacc_debug:
            with torch.no_grad():
                self._sacc_rho_sum += rho
                # 统计平均 |scale-1|，越接近0说明基本没做事
                self._sacc_scale_abs_sum += float((scale - 1.0).abs().mean().item())
                self._sacc_scale_cnt += 1

        return sup_out, qry_out

    # -------------------- TTA 增广 --------------------
    def _augment_image(self, x):
        if not self.use_tta:
            return x, 1

        B_N, C, H, W = x.shape
        aug0 = x
        aug1 = torch.flip(x, dims=[3])
        aug2 = torch.rot90(x, k=1, dims=[2, 3])
        aug3 = torch.rot90(x, k=-1, dims=[2, 3])

        pad = 40
        if H > pad * 2 and W > pad * 2:
            crop = x[:, :, pad:-pad, pad:-pad]
            aug4 = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)
        else:
            aug4 = x

        x_aug = torch.stack([aug0, aug1, aug2, aug3, aug4], dim=1).view(-1, C, H, W)
        return x_aug, 5

    # -------------------- Covariance --------------------
    def _matrix_sqrt(self, x, iter_n=3):
        dtype = x.dtype
        device = x.device
        norm = torch.linalg.norm(x, dim=(1, 2), keepdim=True)
        Y = x / (norm + 1e-8)
        I = torch.eye(x.size(1), device=device, dtype=dtype).unsqueeze(0).expand_as(x)
        Z = torch.eye(x.size(1), device=device, dtype=dtype).unsqueeze(0).expand_as(x)

        for _ in range(int(iter_n)):
            T = 0.5 * (3.0 * I - torch.bmm(Z, Y))
            Y = torch.bmm(Y, T)
            Z = torch.bmm(T, Z)

        return Y * torch.sqrt(norm + 1e-8)

    def _compute_robust_covariance(self, patch_tokens):
        x = torch.matmul(patch_tokens, self.cov_reducer)
        x = x - x.mean(dim=1, keepdim=True)

        denom = max(1, x.size(1) - 1)
        cov = torch.matmul(x.transpose(1, 2), x) / float(denom)

        cov = cov + self.cov_eps * torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype).unsqueeze(0)
        cov_sqrt = self._matrix_sqrt(cov, iter_n=self.cov_sqrt_iter)

        d = cov_sqrt.size(1)
        idx = torch.triu_indices(d, d, device=cov_sqrt.device)
        cov_flat = cov_sqrt[:, idx[0], idx[1]]
        return cov_flat

    # -------------------- Token split --------------------
    def _split_cls_patch(self, tokens):
        if tokens.dim() != 3:
            return None, tokens, tokens
        if self.is_vit_like and tokens.size(1) > 1:
            cls = tokens[:, 0]
            patch = tokens[:, 1:]
            return cls, patch, tokens
        return None, tokens, tokens

    # -------------------- Tukey --------------------
    def _tukey_transform(self, x, beta=0.5):
        return torch.sign(x) * torch.pow(torch.abs(x) + 1e-8, float(beta))

    # -------------------- 可分性分数（向量化）--------------------
    @staticmethod
    def _disc_score_vectorized(x_sup, y_sup, eps=1e-8):
        """
        x_sup: [B,N,D]  y_sup: [B,N] -> score [B]
        """
        B, N, D = x_sup.shape
        C = int(y_sup.max().item() + 1)

        Y = F.one_hot(y_sup.long(), num_classes=C).float()          # [B,N,C]
        n_c = Y.sum(dim=1)                                          # [B,C]
        n_c_safe = n_c.clamp_min(1.0)

        mu_c = torch.bmm(Y.transpose(1, 2), x_sup) / n_c_safe.unsqueeze(-1)  # [B,C,D]
        mu = x_sup.mean(dim=1, keepdim=True)                                # [B,1,D]

        p_c = n_c / float(N)                                                # [B,C]
        diff = mu_c - mu                                                    # [B,C,D]
        between = (p_c.unsqueeze(-1) * diff.pow(2)).sum(dim=1).mean(dim=1)   # [B]

        m2_c = torch.bmm(Y.transpose(1, 2), x_sup.pow(2)) / n_c_safe.unsqueeze(-1)
        var_c = (m2_c - mu_c.pow(2)).clamp_min(0.0)
        within = (p_c * var_c.mean(dim=2)).sum(dim=1)

        return between / (within + eps)

    # -------------------- 第三步：support 驱动 view 加权池化 --------------------
    def _pool_tta_support_weighted(self, sup_view, qry_view, y_sup):
        """
        sup_view: [B,Ns,V,D]
        qry_view: [B,Nq,V,D]
        """
        B, Ns, V, D = sup_view.shape
        if qry_view.shape[2] != V:
            return sup_view.mean(dim=2), qry_view.mean(dim=2)

        mode = str(self.tta_pool_mode).lower()
        if mode == "mean" or V == 1:
            return sup_view.mean(dim=2), qry_view.mean(dim=2)

        # 每个 view 用 support 可分性打分
        scores = []
        for v in range(V):
            sv = F.normalize(sup_view[:, :, v, :], dim=-1)
            scores.append(self._disc_score_vectorized(sv, y_sup))
        S = torch.stack(scores, dim=1)  # [B,V]

        if mode == "support_topk":
            k = max(1, min(int(self.tta_topk), V))
            topk = torch.topk(S, k=k, dim=1).indices
            W = torch.zeros_like(S)
            W.scatter_(1, topk, 1.0)
            W = W / W.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            temp = max(1e-6, float(self.tta_weight_temp))
            W = torch.softmax(S / temp, dim=1)

        W = W.view(B, 1, V, 1)
        sup_pool = (sup_view * W).sum(dim=2)
        qry_pool = (qry_view * W).sum(dim=2)
        return sup_pool, qry_pool

    # -------------------- 组件提取（按层）--------------------
    def _get_feature_components(self, feats_list):
        results = []
        for feat in feats_list:
            comp = {}

            # CNN -> tokens
            if feat.dim() == 4:
                tokens = feat.flatten(2).transpose(1, 2).contiguous()
            elif feat.dim() == 3:
                tokens = feat
            elif feat.dim() == 2:
                tokens = feat.unsqueeze(1)
            else:
                raise ValueError(f"无法识别的特征形状：{tuple(feat.shape)}")

            cls, patch, all_tokens = self._split_cls_patch(tokens)

            # cov/bls 统计 token（可排除 CLS）
            stat_tokens = patch if (cls is not None and self.exclude_cls_for_stats) else all_tokens

            # GAP
            if self.gap_mode == "adaptive_gated" and cls is not None:
                comp["gap_patch"] = patch.mean(dim=1)
                comp["gap_cls"] = cls
            else:
                if self.gap_mode == "patch":
                    gap = patch.mean(dim=1)
                elif self.gap_mode == "cls" and cls is not None:
                    gap = cls
                elif self.gap_mode == "concat" and cls is not None:
                    gap = torch.cat([patch.mean(dim=1), cls], dim=-1)
                else:
                    gap = all_tokens.mean(dim=1)
                comp["gap"] = gap

            # COV
            if self.use_cov and self.cov_reducer is not None:
                comp["cov"] = self._compute_robust_covariance(stat_tokens)

            # BLS
            if self.use_bls and self.w_bls is not None:
                bls = torch.tanh(stat_tokens @ self.w_bls + self.b_bls).mean(dim=1)
                comp["bls"] = bls

            results.append(comp)
        return results

    # -------------------- SACC：按层按分量 fuse + 校准（支持 sup/qry 同变换）--------------------
    def _fuse_layers_view_pair(
        self,
        comps_sup, comps_qry,
        key: str,
        Vsup: int, Vqry: int,
        B: int, Ns: int, Nq: int,
        y_sup: torch.Tensor,
        comp_name_for_bank: str,
        apply_sacc: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对每个层：raw -> [B,N,V,D_l]，可选 SACC(diag)，最后沿特征维拼接：
          sup_cat: [B,Ns,Vsup,D_total]
          qry_cat: [B,Nq,Vqry,D_total]
        """
        sup_chunks = []
        qry_chunks = []

        for li, (layer_sup, layer_qry) in enumerate(zip(comps_sup, comps_qry)):
            if key not in layer_sup or key not in layer_qry:
                continue

            raw_sup = layer_sup[key]  # [B*Ns*Vsup, D_l]
            raw_qry = layer_qry[key]  # [B*Nq*Vqry, D_l]

            if raw_sup.dim() != 2 or raw_qry.dim() != 2:
                raise RuntimeError(f"{key} 特征必须是 2D 向量，但得到 {raw_sup.dim()}D / {raw_qry.dim()}D")

            if (raw_sup.shape[0] % Vsup != 0) or (raw_qry.shape[0] % Vqry != 0):
                raise RuntimeError(f"TTA reshape 失败：样本数不能整除 views，sup={raw_sup.shape[0]}/{Vsup}, qry={raw_qry.shape[0]}/{Vqry}")

            sup_view = raw_sup.view(B, Ns, Vsup, -1)
            qry_view = raw_qry.view(B, Nq, Vqry, -1)

            # SACC(diag)
            if self.use_sacc and apply_sacc and (self.sacc_mode == "diag"):
                if (not self.canonical_bank) and self.sacc_debug and (not self._sacc_warned_no_bank):
                    print("[SACC DEBUG] use_sacc=True 但 canonical_bank 为空：你可能没提供 sacc_bank_path，或加载失败。")
                    self._sacc_warned_no_bank = True

                layer_idx = self.mlf_layers[li]
                D_l = sup_view.shape[-1]
                mu0, var0 = self._bank_get_mu_var(comp_name_for_bank, layer_idx, D_l, sup_view.device, sup_view.dtype)

                # 记录 bank 命中/缺失（debug）
                if self.sacc_debug:
                    if (mu0 is None) or (var0 is None):
                        self._sacc_missed += 1
                    else:
                        self._sacc_used += 1

                sup_view, qry_view = self._diag_sacc_pair(sup_view, qry_view, mu0, var0, n_support=Ns)

            sup_chunks.append(sup_view)
            qry_chunks.append(qry_view)

        if len(sup_chunks) == 0:
            raise RuntimeError(f"_fuse_layers_view_pair: 没有找到 key={key} 的分量输出（检查 gap_mode/use_cov/use_bls 配置）")

        sup_cat = torch.cat(sup_chunks, dim=-1)
        qry_cat = torch.cat(qry_chunks, dim=-1)
        return sup_cat, qry_cat

    # -------------------- forward --------------------
    def forward(self, x_sup, y_sup, x_qry):
        B = x_sup.shape[0]
        Ns = x_sup.shape[1]
        Nq = x_qry.shape[1]

        if y_sup.dim() == 3:
            y_sup = y_sup.flatten(1)
        y_sup = y_sup.long()

        x_sup_flat = x_sup.view(-1, *x_sup.shape[-3:])
        x_qry_flat = x_qry.view(-1, *x_qry.shape[-3:])

        # TTA
        x_sup_aug, Vsup = self._augment_image(x_sup_flat)
        x_qry_aug, Vqry = self._augment_image(x_qry_flat)

        # 抽层特征
        feats_sup = self.extractor.run(x_sup_aug)
        feats_qry = self.extractor.run(x_qry_aug)

        # 组件提取（按层）
        comps_sup = self._get_feature_components(feats_sup)
        comps_qry = self._get_feature_components(feats_qry)

        parts_sup = []
        parts_qry = []

        # =========================
        # GAP：普通 or adaptive_gated
        # =========================
        if ("gap_patch" in comps_sup[0]) and ("gap_cls" in comps_sup[0]):
            # patch / cls 分支分别 fuse + SACC
            sup_patch, qry_patch = self._fuse_layers_view_pair(
                comps_sup, comps_qry,
                key="gap_patch",
                Vsup=Vsup, Vqry=Vqry,
                B=B, Ns=Ns, Nq=Nq,
                y_sup=y_sup,
                comp_name_for_bank="gap_patch",
                apply_sacc=(self.sacc_on_gap and self.sacc_on_gap_patch)
            )
            sup_cls, qry_cls = self._fuse_layers_view_pair(
                comps_sup, comps_qry,
                key="gap_cls",
                Vsup=Vsup, Vqry=Vqry,
                B=B, Ns=Ns, Nq=Nq,
                y_sup=y_sup,
                comp_name_for_bank="gap_cls",
                apply_sacc=(self.sacc_on_gap and self.sacc_on_gap_cls)
            )

            # Tukey + L2（逐 view）
            sup_patch = F.normalize(self._tukey_transform(sup_patch, self.tukey_beta_gap), dim=-1)
            qry_patch = F.normalize(self._tukey_transform(qry_patch, self.tukey_beta_gap), dim=-1)
            sup_cls = F.normalize(self._tukey_transform(sup_cls, self.tukey_beta_gap), dim=-1)
            qry_cls = F.normalize(self._tukey_transform(qry_cls, self.tukey_beta_gap), dim=-1)

            # view pooling（支持加权/TopK/mean）
            sp, qp = self._pool_tta_support_weighted(sup_patch, qry_patch, y_sup)
            sc, qc = self._pool_tta_support_weighted(sup_cls, qry_cls, y_sup)

            # support 驱动 patch/cls 门控
            score_p = self._disc_score_vectorized(F.normalize(sp, dim=-1), y_sup)
            score_c = self._disc_score_vectorized(F.normalize(sc, dim=-1), y_sup)
            g = torch.sigmoid((score_p - score_c) / max(1e-6, self.gap_gate_temp)).view(B, 1, 1)

            sup_gap = g * sp + (1.0 - g) * sc
            qry_gap = g * qp + (1.0 - g) * qc

            parts_sup.append(F.normalize(sup_gap, dim=-1).view(B * Ns, -1))
            parts_qry.append(F.normalize(qry_gap, dim=-1).view(B * Nq, -1))

        else:
            sup_gap, qry_gap = self._fuse_layers_view_pair(
                comps_sup, comps_qry,
                key="gap",
                Vsup=Vsup, Vqry=Vqry,
                B=B, Ns=Ns, Nq=Nq,
                y_sup=y_sup,
                comp_name_for_bank="gap",
                apply_sacc=self.sacc_on_gap
            )

            sup_gap = F.normalize(self._tukey_transform(sup_gap, self.tukey_beta_gap), dim=-1)
            qry_gap = F.normalize(self._tukey_transform(qry_gap, self.tukey_beta_gap), dim=-1)

            sup_gap, qry_gap = self._pool_tta_support_weighted(sup_gap, qry_gap, y_sup)

            parts_sup.append(F.normalize(sup_gap, dim=-1).view(B * Ns, -1))
            parts_qry.append(F.normalize(qry_gap, dim=-1).view(B * Nq, -1))

        # =========================
        # COV
        # =========================
        if "cov" in comps_sup[0]:
            sup_cov, qry_cov = self._fuse_layers_view_pair(
                comps_sup, comps_qry,
                key="cov",
                Vsup=Vsup, Vqry=Vqry,
                B=B, Ns=Ns, Nq=Nq,
                y_sup=y_sup,
                comp_name_for_bank="cov",
                apply_sacc=self.sacc_on_cov
            )

            sup_cov = F.normalize(self._tukey_transform(sup_cov, self.tukey_beta_cov), dim=-1)
            qry_cov = F.normalize(self._tukey_transform(qry_cov, self.tukey_beta_cov), dim=-1)

            sup_cov, qry_cov = self._pool_tta_support_weighted(sup_cov, qry_cov, y_sup)

            parts_sup.append(F.normalize(sup_cov, dim=-1).view(B * Ns, -1))
            parts_qry.append(F.normalize(qry_cov, dim=-1).view(B * Nq, -1))

        # =========================
        # BLS
        # =========================
        if "bls" in comps_sup[0]:
            sup_bls, qry_bls = self._fuse_layers_view_pair(
                comps_sup, comps_qry,
                key="bls",
                Vsup=Vsup, Vqry=Vqry,
                B=B, Ns=Ns, Nq=Nq,
                y_sup=y_sup,
                comp_name_for_bank="bls",
                apply_sacc=self.sacc_on_bls
            )

            sup_bls = F.normalize(self._tukey_transform(sup_bls, self.tukey_beta_bls), dim=-1)
            qry_bls = F.normalize(self._tukey_transform(qry_bls, self.tukey_beta_bls), dim=-1)

            sup_bls, qry_bls = self._pool_tta_support_weighted(sup_bls, qry_bls, y_sup)

            parts_sup.append(F.normalize(sup_bls, dim=-1).view(B * Ns, -1))
            parts_qry.append(F.normalize(qry_bls, dim=-1).view(B * Nq, -1))

        # 拼接
        z_sup = torch.cat(parts_sup, dim=-1).view(B, Ns, -1)
        z_qry = torch.cat(parts_qry, dim=-1).view(B, Nq, -1)

        # post centering（保持可控）
        if self.post_centering == "support":
            mu = z_sup.mean(dim=1, keepdim=True)
            z_sup = z_sup - mu
            z_qry = z_qry - mu
        elif self.post_centering == "none":
            pass
        else:
            mu = z_sup.mean(dim=1, keepdim=True)
            z_sup = z_sup - mu
            z_qry = z_qry - mu

        # 可选：support-only 标准化（保留）
        if self.use_support_standardize:
            var_hat = (z_sup ** 2).mean(dim=1, keepdim=True)
            rho = self.calib_shrinkage
            if rho < 0:
                rho = min(0.25, 2.0 / max(1.0, float(Ns)))
            var = (1.0 - rho) * var_hat + rho * torch.ones_like(var_hat)
            std = torch.sqrt(var + self.calib_eps)
            z_sup = z_sup / std
            z_qry = z_qry / std

        z_sup = F.normalize(z_sup, dim=-1)
        z_qry = F.normalize(z_qry, dim=-1)

        # Ridge Regression
        C = int(y_sup.max().item() + 1)
        Y = F.one_hot(y_sup, num_classes=C).float()

        K = torch.bmm(z_sup, z_sup.transpose(1, 2))

        mode = str(self.lambda_mode).lower()
        if mode == "fixed":
            lam = self.reg_lambda
        elif mode == "trace_scaled":
            tr = torch.diagonal(K, dim1=-2, dim2=-1).mean(dim=-1)
            lam = tr.mean().item() * self.reg_lambda
        else:
            d_feat = z_sup.size(-1)
            lam = self.reg_lambda * (d_feat / 1000.0)

        I = torch.eye(Ns, device=z_sup.device, dtype=z_sup.dtype).unsqueeze(0)
        alpha = torch.linalg.solve(K + lam * I, Y)

        sim = torch.bmm(z_qry, z_sup.transpose(1, 2))
        logits = torch.bmm(sim, alpha)

        # ===== SACC debug 打印 =====
        if self.sacc_debug:
            self._sacc_debug_step += 1
            if self._sacc_debug_step % max(1, self.sacc_debug_interval) == 0:
                total = self._sacc_used + self._sacc_missed
                used_ratio = 0.0 if total == 0 else (self._sacc_used / float(total))
                mean_rho = 0.0 if self._sacc_scale_cnt == 0 else (self._sacc_rho_sum / float(self._sacc_scale_cnt))
                mean_abs_scale = 0.0 if self._sacc_scale_cnt == 0 else (self._sacc_scale_abs_sum / float(self._sacc_scale_cnt))

                print(
                    f"[SACC DEBUG] used={self._sacc_used}, missed={self._sacc_missed}, used_ratio={used_ratio:.3f}, "
                    f"mean_rho={mean_rho:.4f}, mean|scale-1|={mean_abs_scale:.6f}, "
                    f"post_centering={self.post_centering}"
                )
                # 清零（避免累积影响观察）
                self._sacc_used = 0
                self._sacc_missed = 0
                self._sacc_rho_sum = 0.0
                self._sacc_scale_abs_sum = 0.0
                self._sacc_scale_cnt = 0

        return logits * self.scale_cls
