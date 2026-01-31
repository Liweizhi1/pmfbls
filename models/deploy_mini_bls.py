import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import math
from typing import Optional
##这个文件是之前的屎山代码，已经不用了

class ProtoNet_MiniBLS(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        args=None,
        input_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        mapping_dim: Optional[int] = None,
        reg_lambda: float = 1e-3,
    ):
        super().__init__()
        self.backbone = backbone

        if args is not None:
            self._debug_raw_args_mapping_dim = getattr(args, 'mini_bls_mapping_dim', None)
            self._debug_raw_args_reg_lambda = getattr(args, 'mini_bls_reg_lambda', None)
            if num_classes is None:
                num_classes = getattr(args, 'test_n_way', None)
            if num_classes is None:
                num_classes = getattr(args, 'nClsEpisode', 5)
            if mapping_dim is None:
                mapping_dim = getattr(args, 'mini_bls_mapping_dim', None)

            reg_lambda = getattr(args, 'mini_bls_reg_lambda', reg_lambda)
            self.robust_level = int(getattr(args, 'mini_bls_robust_level', 0))
            self.irls_iters = int(getattr(args, 'mini_bls_irls_iters', 3))
            self.huber_delta = float(getattr(args, 'mini_bls_huber_delta', 1.0))
            self.margin_tau = float(getattr(args, 'mini_bls_margin_tau', 0.5))
            self.weight_min = float(getattr(args, 'mini_bls_weight_min', 0.2))
            self.mcc_sigma = float(getattr(args, 'mini_bls_mcc_sigma', 0.5))
            # Graph regularization (episode-level, transductive via query features)
            self.graph_lambda = float(getattr(args, 'mini_bls_graph_lambda', 0.0))
            self.graph_k = int(getattr(args, 'mini_bls_graph_k', 10))
            self.graph_sigma = float(getattr(args, 'mini_bls_graph_sigma', 1.0))

            # SVD singular value truncation (optional denoising for wide mappings)
            self.svd_enable = bool(getattr(args, 'mini_bls_svd_enable', False))
            self.svd_drop = float(getattr(args, 'mini_bls_svd_drop', 0.0))
            self.svd_energy = float(getattr(args, 'mini_bls_svd_energy', 0.9))
            self.svd_min_rank = int(getattr(args, 'mini_bls_svd_min_rank', 1))

            # TCF Adapter (TransT-style cross-attention over ViT patch tokens)
            self.tcf_enable = bool(getattr(args, 'mini_bls_tcf_enable', False))
            self.tcf_k = int(getattr(args, 'mini_bls_tcf_k', 4))
            self.tcf_mode = str(getattr(args, 'mini_bls_tcf_mode', 'learnable'))
            self.tcf_out = str(getattr(args, 'mini_bls_tcf_out', 'flat'))
            # Episode-level refinement for TCF prototypes (refine mode)
            self.tcf_refine_steps = int(getattr(args, 'mini_bls_tcf_refine_steps', 0))
            self.tcf_refine_lr = float(getattr(args, 'mini_bls_tcf_refine_lr', 0.5))
            self.tcf_refine_temp = float(getattr(args, 'mini_bls_tcf_refine_temp', 0.05))
            self.tcf_refine_kmeans_iters = int(getattr(args, 'mini_bls_tcf_refine_kmeans_iters', 10))
            self.tcf_refine_max_points = int(getattr(args, 'mini_bls_tcf_refine_max_points', 1024))

            # Correlated Fuzzy Mapping (CorF)
            self.corf_enable = bool(getattr(args, 'mini_bls_corf_enable', False))
            self.corf_num_subsystems = int(getattr(args, 'mini_bls_corf_num_subsystems', 8))
            self.corf_num_rules = int(getattr(args, 'mini_bls_corf_num_rules', 2))
            self.corf_sub_dim = int(getattr(args, 'mini_bls_corf_sub_dim', 64))
            self.corf_kmeans_iters = int(getattr(args, 'mini_bls_corf_kmeans_iters', 10))
            self.corf_sigma = float(getattr(args, 'mini_bls_corf_sigma', 1.0))
            self.corf_cov_eps = float(getattr(args, 'mini_bls_corf_cov_eps', 1e-4))

            # Multi-Layer Feature Fusion (MLF)
            self.mlf_enable = bool(getattr(args, 'mini_bls_mlf_enable', False))
            mlf_layers_raw = str(getattr(args, 'mini_bls_mlf_layers', '-1,-3,-6'))
            try:
                self.mlf_layers = [int(x) for x in mlf_layers_raw.split(',') if x.strip() != '']
            except Exception:
                self.mlf_layers = [-1, -3, -6]
            if len(self.mlf_layers) == 0:
                self.mlf_layers = [-1]
            self.mlf_max_n = max([abs(x) for x in self.mlf_layers])
            self.mlf_k = len(self.mlf_layers)
            # If both TCF and MLF are requested, prefer TCF and disable MLF (TCF operates on patch tokens).
            if bool(self.tcf_enable) and bool(self.mlf_enable):
                try:
                    print('[MiniBLS] Warning: both TCF and MLF enabled; disabling MLF and using TCF.')
                except Exception:
                    pass
                self.mlf_enable = False

            # Feature Orthogonalization (background direction removal)
            # Episode-level: compute mu from support set and remove projection along mu.
            self.ortho_enable = bool(getattr(args, 'mini_bls_ortho_enable', False))
            self.ortho_eps = float(getattr(args, 'mini_bls_ortho_eps', 1e-6))
            self.ortho_mode = str(getattr(args, 'mini_bls_ortho_mode', 'mu'))
            self.ortho_k = int(getattr(args, 'mini_bls_ortho_k', 1))

            # Second-order statistics (covariance pooling over patch tokens) for MLF layers.
            self.cov_enable = bool(getattr(args, 'mini_bls_cov_enable', False))
            self.cov_proj_dim = int(getattr(args, 'mini_bls_cov_proj_dim', 16))
            self.cov_power = float(getattr(args, 'mini_bls_cov_power', 0.5))
            self.cov_eps = float(getattr(args, 'mini_bls_cov_eps', 1e-4))
            self.cov_seed_offset = int(getattr(args, 'mini_bls_cov_seed_offset', 2021))

            # Optional feature-space non-linear calibration (power transform)
            # Applied to Z_support/Z_query after initial L2 norm and before random mapping.
            self.power_transform_enable = bool(getattr(args, 'mini_bls_power_transform', False))
            self.power_transform_gamma = float(getattr(args, 'mini_bls_power_gamma', 0.5))
            self.power_transform_eps = float(getattr(args, 'mini_bls_power_eps', 1e-6))
            self.power_transform_mode = str(getattr(args, 'mini_bls_power_mode', 'signed'))

            # Double relaxation (lightweight)
            self.label_relax = float(getattr(args, 'mini_bls_label_relax', 0.0))
            self.graph_relax = float(getattr(args, 'mini_bls_graph_relax', 0.0))

            # Transductive self-training (query pseudo-label refinement)
            self.self_train_alpha = float(getattr(args, 'mini_bls_self_train_alpha', 0.0))
            self.self_train_iters = int(getattr(args, 'mini_bls_self_train_iters', 1))
            self.self_train_temp = float(getattr(args, 'mini_bls_self_train_temp', 1.0))
            self.self_train_conf_thr = float(getattr(args, 'mini_bls_self_train_conf_thr', 0.0))
            seed_base = int(getattr(args, 'seed', 0))
            seed_offset = int(getattr(args, 'mini_bls_map_seed_offset', 777))
            self.map_seed = seed_base + seed_offset

            # Ensemble over multiple independent random mappings (average logits)
            self.ensemble = int(getattr(args, 'mini_bls_ensemble', 1))
            self.ensemble_seed_stride = int(getattr(args, 'mini_bls_ensemble_seed_stride', 1000))

            # Enhancement nodes
            self.enhance_type = str(getattr(args, 'mini_bls_enhance_type', 'tanh'))
            self.gauss_sigma_mode = str(getattr(args, 'mini_bls_gauss_sigma_mode', 'adaptive'))
            self.gauss_sigma_mode = 'fixed'
            self.gauss_sigma = float(getattr(args, 'mini_bls_gauss_sigma', 1.0))
            self.gauss_sigma_scale = float(getattr(args, 'mini_bls_gauss_sigma_scale', 1.0))
            self.gauss_sigma_eps = float(getattr(args, 'mini_bls_gauss_sigma_eps', 1e-3))

            # Ridge solver controls (numerical/repro)
            self.solve_device = str(getattr(args, 'mini_bls_solve_device', 'auto'))
            self.solve_dtype = str(getattr(args, 'mini_bls_solve_dtype', 'float32'))

            # Orthogonal mapping init
            self.map_orthogonal = bool(getattr(args, 'mini_bls_map_orthogonal', False))

            # Virtual sample synthesis
            self.virtual_samples = int(getattr(args, 'mini_bls_virtual_samples', 0))
            self.virtual_scale = float(getattr(args, 'mini_bls_virtual_scale', 1.0))
            self.virtual_weight = float(getattr(args, 'mini_bls_virtual_weight', 0.5))
            # Class-balanced weighting
            self.class_balanced = bool(getattr(args, 'mini_bls_class_balanced', False))
            self.class_balanced_mode = str(getattr(args, 'mini_bls_cb_mode', 'inv_sqrt'))

            # Residual Feature Refinement / Residual Classification (optional)
            # - refine: refine query features via residual iterations (keep support unchanged)
            # - classify_residual: replace features with residuals and run the usual MiniBLS
            self.residual_mode = str(getattr(args, 'mini_bls_residual_mode', 'none'))
            self.residual_base = str(getattr(args, 'mini_bls_residual_base', 'ridge'))
            self.residual_ridge_lambda = float(getattr(args, 'mini_bls_residual_ridge_lambda', 1e-2))
            self.residual_temp = float(getattr(args, 'mini_bls_residual_temp', 1.0))
            self.residual_alpha = float(getattr(args, 'mini_bls_residual_alpha', 0.5))
            self.residual_iters = int(getattr(args, 'mini_bls_residual_iters', 1))
        else:
            self.robust_level = 0
            self.irls_iters = 3
            self.huber_delta = 1.0
            self.margin_tau = 0.5
            self.weight_min = 0.2
            self.mcc_sigma = 0.5
            self.graph_lambda = 0.0
            self.graph_k = 10
            self.graph_sigma = 1.0
            self.svd_enable = False
            self.svd_drop = 0.0
            self.svd_energy = 0.9
            self.svd_min_rank = 1
            self.map_seed = 777
            self.ensemble = 1
            self.ensemble_seed_stride = 1000
            self.enhance_type = 'tanh'
            self.gauss_sigma_mode = 'fixed'
            self.gauss_sigma = 1.0
            self.gauss_sigma_scale = 1.0
            self.gauss_sigma_eps = 1e-3
            self.solve_device = 'auto'
            self.solve_dtype = 'float32'
            self.map_orthogonal = False
            self.virtual_samples = 0
            self.virtual_scale = 1.0
            self.virtual_weight = 0.5
            self.tcf_enable = False
            self.tcf_k = 4
            self.tcf_mode = 'learnable'
            self.tcf_out = 'flat'
            self.tcf_refine_steps = 0
            self.tcf_refine_lr = 0.5
            self.tcf_refine_temp = 0.05
            self.tcf_refine_kmeans_iters = 10
            self.tcf_refine_max_points = 1024
            self.corf_enable = False
            self.corf_num_subsystems = 8
            self.corf_num_rules = 2
            self.corf_sub_dim = 64
            self.corf_kmeans_iters = 10
            self.corf_sigma = 1.0
            self.corf_cov_eps = 1e-4
            self.mlf_enable = False
            self.mlf_layers = [-1]
            self.mlf_max_n = 1
            self.mlf_k = 1
            self.ortho_enable = False
            self.ortho_eps = 1e-6
            self.ortho_mode = 'mu'
            self.ortho_k = 1
            self.cov_enable = False
            self.cov_proj_dim = 16
            self.cov_power = 0.5
            self.cov_eps = 1e-4
            self.cov_seed_offset = 2021
            self.power_transform_enable = False
            self.power_transform_gamma = 0.5
            self.power_transform_eps = 1e-6
            self.power_transform_mode = 'signed'
            self.label_relax = 0.0
            self.graph_relax = 0.0
            self.self_train_alpha = 0.0
            self.self_train_iters = 1
            self.self_train_temp = 1.0
            self.self_train_conf_thr = 0.0
            self.class_balanced = False
            self.class_balanced_mode = 'inv_sqrt'

            self.residual_mode = 'none'
            self.residual_base = 'ridge'
            self.residual_ridge_lambda = 1e-2
            self.residual_temp = 1.0
            self.residual_alpha = 0.5
            self.residual_iters = 1

        if num_classes is None:
            num_classes = 5
        if mapping_dim is None:
            mapping_dim = 100

        if input_dim is None:
            input_dim = getattr(backbone, 'embed_dim', None)
        if input_dim is None:
            input_dim = getattr(backbone, 'num_features', None)
        if input_dim is None:
            # Fallback: infer feature dim by a cheap forward.
            try:
                param = next(backbone.parameters())
                device = param.device
            except StopIteration:
                device = torch.device('cpu')
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=device)
                feat = backbone(dummy)
                input_dim = int(feat.shape[-1])

        token_dim = int(input_dim)

        # Base MiniBLS input dim (before optional CorF concatenation)
        base_input_dim = token_dim

        # If MLF is enabled (and not using TCF), concatenate pooled features from requested layers.
        if bool(getattr(self, 'mlf_enable', False)) and not bool(getattr(self, 'tcf_enable', False)):
            k = max(1, int(getattr(self, 'mlf_k', 1)))
            base_input_dim = int(token_dim * k)
            if bool(getattr(self, 'cov_enable', False)):
                r = max(1, int(getattr(self, 'cov_proj_dim', 16)))
                base_input_dim = int(base_input_dim + (r * r) * k)

        # If TCF is enabled, the MiniBLS input becomes the flattened K*d TCF output.
        if bool(self.tcf_enable):
            K = max(1, int(self.tcf_k))
            out_mode = str(getattr(self, 'tcf_out', 'flat'))
            if out_mode == 'mean':
                base_input_dim = int(token_dim)
            else:
                base_input_dim = int(K * token_dim)

            # Learnable lesion prototypes Q and key/value projections
            self.tcf_Q = nn.Parameter(torch.randn(K, token_dim) * 0.02)
            self.tcf_Wk = nn.Linear(token_dim, token_dim, bias=False)
            self.tcf_Wv = nn.Linear(token_dim, token_dim, bias=False)

            # Evaluation-friendly default: identity projections (random projections hurt when head is not trained).
            with torch.no_grad():
                nn.init.eye_(self.tcf_Wk.weight)
                nn.init.eye_(self.tcf_Wv.weight)
            self.tcf_Wk.weight.requires_grad_(False)
            self.tcf_Wv.weight.requires_grad_(False)
        else:
            self.tcf_Q = None
            self.tcf_Wk = None
            self.tcf_Wv = None

        # CorF output dim (K_f * R) and frozen random projections for subsystems
        if bool(self.corf_enable):
            self.corf_dim = int(max(1, self.corf_num_subsystems) * max(1, self.corf_num_rules))
        else:
            self.corf_dim = 0

        input_dim = int(base_input_dim + self.corf_dim)
        self.base_input_dim = int(base_input_dim)

        # Covariance pooling projection (frozen). Only used when cov_enable=True.
        if bool(getattr(self, 'cov_enable', False)):
            r = max(1, int(getattr(self, 'cov_proj_dim', 16)))
            gen_cov = torch.Generator(device='cpu')
            gen_cov.manual_seed(int(self.map_seed) + int(getattr(self, 'cov_seed_offset', 2021)))
            self.cov_proj = nn.Parameter(
                torch.randn(int(token_dim), int(r), generator=gen_cov) * (1.0 / math.sqrt(max(1, int(token_dim)))),
                requires_grad=False,
            )
        else:
            self.cov_proj = None

        # Cache for ensemble mapping weights (keyed by (z_dim, mapping_dim, member, device, dtype))
        self._ens_map_cache: dict[tuple, torch.Tensor] = {}

        # CorF subsystem projections (frozen)
        if bool(self.corf_enable):
            kf = max(1, int(self.corf_num_subsystems))
            sub_dim = max(1, int(self.corf_sub_dim))
            gen_corf = torch.Generator(device='cpu')
            gen_corf.manual_seed(int(self.map_seed) + 999)
            self.corf_proj = nn.Parameter(
                torch.randn(kf, token_dim, sub_dim, generator=gen_corf),
                requires_grad=False,
            )
        else:
            self.corf_proj = None

        # 1) Random feature mapping (frozen)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(int(self.map_seed))
        W0 = torch.randn(int(input_dim), int(mapping_dim), generator=gen)
        # Always scale at init; optional orthogonalization will be performed lazily on-device.
        W0 = W0 * (1.0 / math.sqrt(max(1, int(input_dim))))
        self.mapping_weight = nn.Parameter(W0, requires_grad=False)
        self._map_orthogonal_pending = bool(getattr(self, 'map_orthogonal', False))
        self.bls_mapping_bias = nn.Parameter(
            torch.zeros(int(mapping_dim)),
            requires_grad=False,
        )

        self.num_classes = int(num_classes)
        self.reg_lambda = float(reg_lambda)

        # Debug print: make sure CLI hyper-params are actually applied.
        try:
            print(
                "[MiniBLS] cfg: "
                f"raw_args_mapping_dim={getattr(self, '_debug_raw_args_mapping_dim', None)}, "
                f"raw_args_reg_lambda={getattr(self, '_debug_raw_args_reg_lambda', None)}, "
                f"num_classes={self.num_classes}, reg_lambda={self.reg_lambda}, "
                f"robust_level={int(self.robust_level)}, irls_iters={int(self.irls_iters)}, "
                f"graph_lambda={float(self.graph_lambda)}, svd_enable={bool(getattr(self, 'svd_enable', False))}, "
                f"svd_drop={float(getattr(self, 'svd_drop', 0.0))}, svd_min_rank={int(getattr(self, 'svd_min_rank', 1))}, "
                f"tcf_enable={bool(self.tcf_enable)}, "
                f"corf_enable={bool(self.corf_enable)}, map_seed={int(self.map_seed)}, "
                f"mapping_weight_shape={tuple(self.mapping_weight.shape)}"
            )
            # Extra: what args object did we actually receive?
            _args_type = type(args).__name__ if args is not None else None
            _args_has_dict = hasattr(args, '__dict__')
            _args_keys = None
            if _args_has_dict:
                try:
                    _args_keys = sorted(list(args.__dict__.keys()))[:25]
                except Exception:
                    _args_keys = None
            print(f"[MiniBLS] args_type={_args_type}, args_has_dict={_args_has_dict}, args_keys_head={_args_keys}")
        except Exception:
            pass

    def _get_ensemble_mapping_weight(
        self,
        z_dim: int,
        member: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mapping_dim = int(self.mapping_weight.shape[1])
        key = (int(z_dim), int(mapping_dim), int(member), str(device), str(dtype))
        w = self._ens_map_cache.get(key, None)
        if w is not None:
            return w

        gen = torch.Generator(device='cpu')
        stride = int(getattr(self, 'ensemble_seed_stride', 1000))
        gen.manual_seed(int(getattr(self, 'map_seed', 777)) + int(member) * stride)
        w_cpu = torch.randn(int(z_dim), int(mapping_dim), generator=gen)
        w_cpu = w_cpu * (1.0 / math.sqrt(max(1, int(z_dim))))
        w = w_cpu.to(device=device, dtype=dtype)
        if bool(getattr(self, 'map_orthogonal', False)):
            w = self._orthogonalize_columns(w)
        self._ens_map_cache[key] = w
        return w

    def _orthogonalize_columns(self, W: torch.Tensor) -> torch.Tensor:
        """Orthonormalize columns of W so that W^T W ~= I (on current device)."""
        with torch.cuda.amp.autocast(enabled=False):
            Wf = W.float()
            if int(Wf.shape[0]) >= int(Wf.shape[1]):
                Q, _ = torch.linalg.qr(Wf, mode='reduced')
                return Q.to(dtype=W.dtype)
            # If fewer rows than cols, orthonormalize rows and transpose back.
            Q, _ = torch.linalg.qr(Wf.transpose(0, 1), mode='reduced')
            return Q.transpose(0, 1).to(dtype=W.dtype)

    def _compute_enhancement(
        self,
        a_support: torch.Tensor,
        a_query: torch.Tensor,
        sigma_ref: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute enhancement nodes from pre-activations a = ZW + b.

        Args:
            a_support: [B, Ns, M]
            a_query:   [B, Nq, M]
        Returns:
            H_support, H_query
        """
        et = str(getattr(self, 'enhance_type', 'tanh')).lower().strip()
        if et == 'relu':
            return torch.relu(a_support), torch.relu(a_query)
        if et == 'sine':
            return torch.sin(a_support), torch.sin(a_query)
        if et == 'gaussian':
            mode = str(getattr(self, 'gauss_sigma_mode', 'adaptive')).lower().strip()
            eps = float(getattr(self, 'gauss_sigma_eps', 1e-3))
            if mode == 'fixed':
                sigma = float(getattr(self, 'gauss_sigma', 1.0))
                sigma = max(eps, float(sigma))
                return torch.exp(-0.5 * (a_support / sigma) ** 2), torch.exp(-0.5 * (a_query / sigma) ** 2)

            # adaptive per neuron, per episode: sigma_j = std(a_support[..., j])
            # - mode=adaptive: compute from the current call's a_support (original behavior)
            # - mode=adaptive_shared/adaptive_joint: optionally reuse external sigma_ref
            with torch.cuda.amp.autocast(enabled=False):
                use_ref = (mode in ('adaptive_shared', 'adaptive_joint')) and (sigma_ref is not None)
                if use_ref:
                    sig = sigma_ref.float().clamp_min(eps)
                else:
                    sig = a_support.float().std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
                    sig = sig * float(getattr(self, 'gauss_sigma_scale', 1.0))
                hs = torch.exp(-0.5 * (a_support.float() / sig) ** 2)
                hq = torch.exp(-0.5 * (a_query.float() / sig) ** 2)
            return hs.to(dtype=a_support.dtype), hq.to(dtype=a_query.dtype)

        if et == 'multi':
            hs_relu = torch.relu(a_support)
            hq_relu = torch.relu(a_query)
            hs_sin = torch.sin(a_support)
            hq_sin = torch.sin(a_query)
            # gaussian in float32 for stability
            eps = float(getattr(self, 'gauss_sigma_eps', 1e-3))
            with torch.cuda.amp.autocast(enabled=False):
                mode = str(getattr(self, 'gauss_sigma_mode', 'fixed')).lower().strip()
                if mode == 'fixed':
                    sig = float(getattr(self, 'gauss_sigma', 1.0))
                    sig = max(eps, float(sig))
                    hs_g = torch.exp(-0.5 * (a_support.float() / sig) ** 2)
                    hq_g = torch.exp(-0.5 * (a_query.float() / sig) ** 2)
                else:
                    use_ref = (mode in ('adaptive_shared', 'adaptive_joint')) and (sigma_ref is not None)
                    if use_ref:
                        sig = sigma_ref.float().clamp_min(eps)
                    else:
                        sig = a_support.float().std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
                        sig = sig * float(getattr(self, 'gauss_sigma_scale', 1.0))
                    hs_g = torch.exp(-0.5 * (a_support.float() / sig) ** 2)
                    hq_g = torch.exp(-0.5 * (a_query.float() / sig) ** 2)
            hs_g = hs_g.to(dtype=a_support.dtype)
            hq_g = hq_g.to(dtype=a_query.dtype)
            return torch.cat([hs_relu, hs_sin, hs_g], dim=-1), torch.cat([hq_relu, hq_sin, hq_g], dim=-1)

        # default: tanh
        return torch.tanh(a_support), torch.tanh(a_query)

    def _compute_class_prototypes(
        self,
        Z_support: torch.Tensor,
        support_y: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Compute class prototypes from support features.

        Args:
            Z_support: [B, Ns, D]
            support_y: [B, Ns]
        Returns:
            protos: [B, C, D]
        """
        with torch.cuda.amp.autocast(enabled=False):
            Zs = Z_support.float()
            y = support_y.long()
            B, Ns, D = Zs.shape
            C = int(num_classes)
            protos = torch.zeros((B, C, D), device=Zs.device, dtype=Zs.dtype)
            counts = torch.zeros((B, C), device=Zs.device, dtype=Zs.dtype)

            idx = y.unsqueeze(-1).expand(-1, -1, D)
            protos.scatter_add_(1, idx, Zs)
            ones = torch.ones((B, Ns), device=Zs.device, dtype=Zs.dtype)
            counts.scatter_add_(1, y, ones)
            protos = protos / counts.clamp_min(1.0).unsqueeze(-1)
        return protos.to(dtype=Z_support.dtype)

    def _base_prob_from_ridge(
        self,
        Z_support: torch.Tensor,
        Y_support: torch.Tensor,
        Z_query: torch.Tensor,
        ridge_lambda: float,
        temp: float,
    ) -> torch.Tensor:
        """Compute base class probabilities using a cheap episode-wise ridge classifier.

        Uses the dual form to avoid inverting DxD (D is large, Ns is small).
        """
        with torch.cuda.amp.autocast(enabled=False):
            X = Z_support.float()  # [B, Ns, D]
            Y = Y_support.float()  # [B, Ns, C]
            Zq = Z_query.float()  # [B, Nq, D]
            B, Ns, _ = X.shape
            lam = float(max(1e-8, ridge_lambda))

            K = torch.bmm(X, X.transpose(1, 2))  # [B, Ns, Ns]
            I = torch.eye(Ns, device=K.device, dtype=K.dtype).unsqueeze(0).expand(B, -1, -1)
            A = torch.linalg.solve(K + lam * I, Y)  # [B, Ns, C]
            W = torch.bmm(X.transpose(1, 2), A)  # [B, D, C]
            logits = torch.bmm(Zq, W)  # [B, Nq, C]
            t = float(max(1e-6, temp))
            prob = torch.softmax(logits / t, dim=-1)
        return prob.to(dtype=Z_query.dtype)

    def _base_prob_from_cosine(
        self,
        Z_query: torch.Tensor,
        protos: torch.Tensor,
        temp: float,
    ) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            Zq = F.normalize(Z_query.float(), p=2, dim=-1)
            P = F.normalize(protos.float(), p=2, dim=-1)
            sim = torch.bmm(Zq, P.transpose(1, 2))  # [B, Nq, C]
            t = float(max(1e-6, temp))
            prob = torch.softmax(sim / t, dim=-1)
        return prob.to(dtype=Z_query.dtype)

    def _apply_residual_mode(
        self,
        Z_support: torch.Tensor,
        Z_query: torch.Tensor,
        support_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply residual feature refinement or residual classification.

        Modes:
            - none: no-op
            - refine: iteratively refine query features using residuals to support prototypes
            - classify_residual: replace features with residuals (support residuals use true labels)
        """
        mode = 'none'
        if mode in ('', 'none', 'off', 'false', '0'):
            return Z_support, Z_query

        B, Ns, D = Z_support.shape
        C = int(self.num_classes)
        y = support_y.long()

        # Prototypes from current support features
        protos = self._compute_class_prototypes(Z_support, support_y, num_classes=C)  # [B, C, D]

        # Support residuals are always defined by true labels
        proto_y = protos.gather(1, y.unsqueeze(-1).expand(-1, -1, D))  # [B, Ns, D]
        resid_s = (Z_support - proto_y)

        base = str(getattr(self, 'residual_base', 'ridge')).lower().strip()
        temp = float(getattr(self, 'residual_temp', 1.0))

        if base == 'ridge':
            Y_sup = F.one_hot(y, num_classes=C).to(dtype=Z_support.dtype)
            prob_q = self._base_prob_from_ridge(
                Z_support=Z_support,
                Y_support=Y_sup,
                Z_query=Z_query,
                ridge_lambda=float(getattr(self, 'residual_ridge_lambda', 1e-2)),
                temp=temp,
            )
        else:
            prob_q = self._base_prob_from_cosine(Z_query=Z_query, protos=protos, temp=temp)

        proto_hat_q = torch.bmm(prob_q, protos)  # [B, Nq, D]
        resid_q = (Z_query - proto_hat_q)

        if mode in ('classify_residual', 'residual', 'resid'):
            Zs_new = F.normalize(resid_s, p=2, dim=-1)
            Zq_new = F.normalize(resid_q, p=2, dim=-1)
            return Zs_new, Zq_new

        # mode == refine
        iters = max(1, int(getattr(self, 'residual_iters', 1)))
        alpha = float(getattr(self, 'residual_alpha', 0.5))
        alpha = float(max(0.0, min(2.0, alpha)))
        Zq_new = Z_query
        for _ in range(iters):
            # keep support fixed to avoid drifting the ridge solve / prototypes
            protos = self._compute_class_prototypes(Z_support, support_y, num_classes=C)
            if base == 'ridge':
                Y_sup = F.one_hot(y, num_classes=C).to(dtype=Z_support.dtype)
                prob_q = self._base_prob_from_ridge(
                    Z_support=Z_support,
                    Y_support=Y_sup,
                    Z_query=Zq_new,
                    ridge_lambda=float(getattr(self, 'residual_ridge_lambda', 1e-2)),
                    temp=temp,
                )
            else:
                prob_q = self._base_prob_from_cosine(Z_query=Zq_new, protos=protos, temp=temp)
            proto_hat_q = torch.bmm(prob_q, protos)
            resid_q = (Zq_new - proto_hat_q)
            # Residual convergence: move toward the explained component (prototype mixture)
            # z <- normalize(z - alpha*(z - proto_hat)) = normalize((1-alpha)z + alpha*proto_hat)
            Zq_new = F.normalize(Zq_new - alpha * resid_q, p=2, dim=-1)
        return Z_support, Zq_new

    def _virtual_synthesize(
        self,
        Z_support: torch.Tensor,
        support_feat: torch.Tensor,
        support_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate virtual samples per class using support covariance (implicit sqrt via Xc).

        Returns:
            Z_virtual: [B, C*M, Dz]
            feat_virtual: [B, C*M, Df]
            y_virtual: [B, C*M]
        """
        M = int(getattr(self, 'virtual_samples', 0))
        if M <= 0:
            return (
                Z_support.new_zeros((Z_support.size(0), 0, Z_support.size(-1))),
                support_feat.new_zeros((support_feat.size(0), 0, support_feat.size(-1))),
                support_y.new_zeros((support_y.size(0), 0), dtype=support_y.dtype),
            )

        scale = float(getattr(self, 'virtual_scale', 1.0))
        eps = 1e-6
        B, Ns, Dz = Z_support.shape
        _, _, Df = support_feat.shape
        C = int(self.num_classes)

        z_out = []
        f_out = []
        y_out = []
        for b in range(B):
            zb = Z_support[b]
            fb = support_feat[b]
            yb = support_y[b].long()
            z_virt_all = []
            f_virt_all = []
            y_virt_all = []
            for c in range(C):
                idx = (yb == c)
                if not torch.any(idx):
                    continue
                Xz = zb[idx]  # [K, Dz]
                Xf = fb[idx]  # [K, Df]
                K = int(Xz.size(0))
                if K <= 1:
                    # if only one sample, no covariance; just replicate with tiny noise
                    mu_z = Xz.mean(dim=0, keepdim=True)
                    mu_f = Xf.mean(dim=0, keepdim=True)
                    noise_z = torch.randn((M, Dz), device=zb.device, dtype=zb.dtype) * 0.01
                    noise_f = torch.randn((M, Df), device=fb.device, dtype=fb.dtype) * 0.01
                    zc = mu_z + noise_z
                    fc = mu_f + noise_f
                else:
                    mu_z = Xz.mean(dim=0, keepdim=True)
                    mu_f = Xf.mean(dim=0, keepdim=True)
                    Xcz = Xz - mu_z
                    Xcf = Xf - mu_f

                    # Implicit sampling: perturb = (G @ Xc)/sqrt(K-1), G~N(0,I_K)
                    G = torch.randn((M, K), device=zb.device, dtype=zb.dtype)
                    denom = math.sqrt(max(1.0, float(K - 1)))
                    dz = (G @ Xcz) / denom
                    df = (G @ Xcf) / denom
                    zc = mu_z + scale * dz
                    fc = mu_f + scale * df

                zc = F.normalize(zc, p=2, dim=-1)
                fc = F.normalize(fc, p=2, dim=-1)
                z_virt_all.append(zc)
                f_virt_all.append(fc)
                y_virt_all.append(torch.full((M,), int(c), device=yb.device, dtype=yb.dtype))

            if len(z_virt_all) == 0:
                z_out.append(zb.new_zeros((0, Dz)))
                f_out.append(fb.new_zeros((0, Df)))
                y_out.append(yb.new_zeros((0,), dtype=yb.dtype))
            else:
                z_out.append(torch.cat(z_virt_all, dim=0))
                f_out.append(torch.cat(f_virt_all, dim=0))
                y_out.append(torch.cat(y_virt_all, dim=0))

        # pad to same length per batch (rare; but keep tensorized)
        max_n = max([t.size(0) for t in z_out]) if len(z_out) > 0 else 0
        if max_n == 0:
            return (
                Z_support.new_zeros((B, 0, Dz)),
                support_feat.new_zeros((B, 0, Df)),
                support_y.new_zeros((B, 0), dtype=support_y.dtype),
            )

        ZV = Z_support.new_zeros((B, max_n, Dz))
        FV = support_feat.new_zeros((B, max_n, Df))
        YV = support_y.new_zeros((B, max_n), dtype=support_y.dtype)
        for b in range(B):
            n = int(z_out[b].size(0))
            if n > 0:
                ZV[b, :n] = z_out[b]
                FV[b, :n] = f_out[b]
                YV[b, :n] = y_out[b]
        return ZV, FV, YV

    def _apply_power_transform(self, z: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self, 'power_transform_enable', False)):
            return z

        orig_dtype = z.dtype
        z32 = z.float()

        # Power transform to compress outliers.
        # gamma=0.5 is a sqrt. Mode controls how negatives are handled.
        eps = float(getattr(self, 'power_transform_eps', 1e-6))
        gamma = float(getattr(self, 'power_transform_gamma', 0.5))
        mode = str(getattr(self, 'power_transform_mode', 'signed')).lower().strip()
        if mode == 'relu':
            z32 = torch.pow(torch.relu(z32) + eps, gamma)
        elif mode == 'shift':
            # Shift per vector to be non-negative, then apply power.
            # Keeps ordering information from negatives (unlike ReLU) while avoiding NaNs.
            zmin = z32.amin(dim=-1, keepdim=True)
            z32 = torch.pow((z32 - zmin) + eps, gamma)
        elif mode == 'abs':
            z32 = torch.pow(torch.abs(z32) + eps, gamma)
        else:
            # signed
            z32 = torch.sign(z32) * torch.pow(torch.abs(z32) + eps, gamma)
        return z32.to(dtype=orig_dtype)

    def _apply_ortho_background(self, z_support: torch.Tensor, z_query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(getattr(self, 'ortho_enable', False)):
            return z_support, z_query

        eps = float(getattr(self, 'ortho_eps', 1e-6))
        mode = str(getattr(self, 'ortho_mode', 'mu')).lower().strip()
        k = max(1, int(getattr(self, 'ortho_k', 1)))

        # z_support/z_query: [B, S, D] / [B, Q, D]
        B, S, D = z_support.shape

        def remove_subspace(z: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
            # basis: [B, K, D] with orthonormal rows
            # proj = (z @ basis^T) @ basis
            coef = torch.matmul(z, basis.transpose(1, 2))  # [B, N, K]
            proj = torch.matmul(coef, basis)  # [B, N, D]
            return z - proj

        with torch.cuda.amp.autocast(enabled=False):
            zs = z_support.float()
            zq = z_query.float()

            if mode == 'pca':
                # Compute top-k principal directions from centered support features.
                xs = zs - zs.mean(dim=1, keepdim=True)  # [B, S, D]
                denom = max(1, S - 1)
                C = torch.matmul(xs.transpose(1, 2), xs) / float(denom)  # [B, D, D]
                C = 0.5 * (C + C.transpose(1, 2))
                # SPD jitter
                I = torch.eye(D, device=C.device, dtype=C.dtype).unsqueeze(0)
                C = C + eps * I
                evals, evecs = torch.linalg.eigh(C)  # ascending
                # take largest k eigenvectors
                basis = evecs[:, :, -k:].transpose(1, 2).contiguous()  # [B, K, D]
            else:
                # mode == 'mu' (default)
                mu = zs.mean(dim=1, keepdim=False)  # [B, D]
                mu = mu / (mu.norm(dim=-1, keepdim=True).clamp_min(eps))
                basis = mu.unsqueeze(1)  # [B, 1, D]

            zs_out = remove_subspace(zs, basis)
            zq_out = remove_subspace(zq, basis)

        return zs_out.to(dtype=z_support.dtype), zq_out.to(dtype=z_query.dtype)

    def _cov_pool_sqrtm(self, tokens: torch.Tensor) -> torch.Tensor:
        """Second-order pooling over patch tokens + matrix power on SPD.

        Args:
            tokens: [N, P, D] patch tokens (exclude CLS), float/half ok.
        Returns:
            vec: [N, r*r] flattened sqrtm covariance in projected space.
        """
        if self.cov_proj is None:
            raise RuntimeError('cov_proj is None but cov pooling requested')

        r = int(getattr(self, 'cov_proj_dim', 16))
        eps = float(getattr(self, 'cov_eps', 1e-4))
        power = float(getattr(self, 'cov_power', 0.5))

        with torch.cuda.amp.autocast(enabled=False):
            x = tokens.float()  # [N, P, D]
            # project to r dims
            x = x @ self.cov_proj.to(device=x.device, dtype=x.dtype)  # [N, P, r]
            # center over tokens
            x = x - x.mean(dim=1, keepdim=True)
            P = x.size(1)
            denom = max(1, P - 1)
            C = (x.transpose(1, 2) @ x) / float(denom)  # [N, r, r]
            C = 0.5 * (C + C.transpose(1, 2))
            I = torch.eye(r, device=C.device, dtype=C.dtype).unsqueeze(0)
            C = C + eps * I

            # Matrix power via eigen-decomposition (SPD)
            evals, evecs = torch.linalg.eigh(C)
            evals = evals.clamp_min(eps)
            evals_p = torch.pow(evals, power)
            Cp = (evecs * evals_p.unsqueeze(1)) @ evecs.transpose(1, 2)
            vec = Cp.reshape(Cp.size(0), r * r)

        return vec.to(dtype=tokens.dtype)

    def _kmeans_batch(
        self,
        x: torch.Tensor,
        n_clusters: int,
        iters: int,
        max_points: Optional[int] = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """Simple batched KMeans (L2) in float32.

        Args:
            x: [B, N, D]
        Returns:
            centers: [B, K, D]
        """
        B, N, D = x.shape
        K = max(1, int(n_clusters))
        iters = max(1, int(iters))

        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()
            if max_points is not None and N > int(max_points):
                # Deterministic subsample for speed
                gen = torch.Generator(device='cpu')
                gen.manual_seed(int(seed))
                idx = torch.randperm(N, generator=gen)[: int(max_points)].to(x.device)
                x32 = x32.index_select(1, idx)
                N = x32.size(1)

            # Init by random points
            gen_init = torch.Generator(device='cpu')
            gen_init.manual_seed(int(seed) + 17)
            init_idx = torch.randint(low=0, high=N, size=(B, K), generator=gen_init, device='cpu').to(x.device)
            centers = torch.gather(x32, dim=1, index=init_idx.unsqueeze(-1).expand(-1, -1, D)).contiguous()

            for _ in range(iters):
                # dist2: [B, N, K]
                x2 = (x32 ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
                c2 = (centers ** 2).sum(dim=-1).unsqueeze(1)  # [B, 1, K]
                dist2 = (x2 + c2 - 2.0 * torch.bmm(x32, centers.transpose(1, 2))).clamp_min(0.0)
                assign = dist2.argmin(dim=-1)  # [B, N]

                # Update centers
                new_centers = torch.zeros_like(centers)
                counts = torch.zeros((B, K), device=x.device, dtype=torch.float32)
                for k in range(K):
                    mask = (assign == k).to(torch.float32)  # [B, N]
                    counts[:, k] = mask.sum(dim=1)
                    new_centers[:, k, :] = torch.bmm(mask.unsqueeze(1), x32).squeeze(1)
                denom = counts.clamp_min(1.0).unsqueeze(-1)
                new_centers = new_centers / denom

                # Handle empty clusters by keeping previous centers
                empty = (counts < 1.0).unsqueeze(-1)
                centers = torch.where(empty, centers, new_centers)

        return centers

    def _whiten_apply(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        """Whiten per-episode with covariance from x itself.

        Args:
            x: [B, N, D]
        Returns:
            xw: [B, N, D]
        """
        B, N, D = x.shape
        eps = float(max(eps, 1e-8))
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()
            xw_all = []
            for b in range(B):
                xb = x32[b]  # [N, D]
                mu = xb.mean(dim=0, keepdim=True)
                xc = xb - mu
                cov = (xc.t() @ xc) / max(1.0, float(N))
                cov = cov + eps * torch.eye(D, device=xb.device, dtype=xb.dtype)
                L = torch.linalg.cholesky(cov)
                # Solve L * y = xc^T  => y = L^{-1} xc^T
                y = torch.linalg.solve_triangular(L, xc.t(), upper=False)
                xw = y.t()
                xw_all.append(xw)
            xw32 = torch.stack(xw_all, dim=0)
        return xw32

    def _corf_fit_transform(
        self,
        support_feat: torch.Tensor,
        query_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Episode-level correlated fuzzy mapping.

        Builds per-subsystem whitened subspaces on support features, KMeans centers,
        then computes Gaussian memberships for both support and query.

        Args:
            support_feat: [B, S, D]
            query_feat: [B, Q, D]

        Returns:
            Zs: [B, S, Kf*R]
            Zq: [B, Q, Kf*R]
        """
        if not bool(self.corf_enable):
            raise RuntimeError('CorF requested but corf_enable is False')
        kf = max(1, int(self.corf_num_subsystems))
        r = max(1, int(self.corf_num_rules))
        sigma = max(float(self.corf_sigma), 1e-6)
        eps = float(self.corf_cov_eps)

        B, S, D = support_feat.shape
        Qn = query_feat.size(1)

        with torch.cuda.amp.autocast(enabled=False):
            sf = support_feat.float()
            qf = query_feat.float()

            out_s = []
            out_q = []
            for j in range(kf):
                P = self.corf_proj[j].to(device=sf.device, dtype=sf.dtype)  # [D, sub_dim]
                s_sub = torch.matmul(sf, P)  # [B, S, sub_dim]
                q_sub = torch.matmul(qf, P)  # [B, Q, sub_dim]

                # Whiten using support covariance per episode
                s_w = self._whiten_apply(s_sub, eps=eps)  # [B, S, sub_dim]
                # Apply the same whitening transform approximately by whitening query with its own stats is wrong.
                # To keep it simple/stable, center query by its mean and whiten by its own cov is avoided; instead,
                # we just center by support mean and reuse support whitening via solving triangular systems.
                # Implement reuse exactly by recomputing per-episode L from support and applying to query.
                q_w_all = []
                for b in range(B):
                    sb = s_sub[b]
                    qb = q_sub[b]
                    mu = sb.mean(dim=0, keepdim=True)
                    sc = (sb - mu)
                    qc = (qb - mu)
                    cov = (sc.t() @ sc) / max(1.0, float(S))
                    cov = cov + eps * torch.eye(cov.size(0), device=sb.device, dtype=sb.dtype)
                    L = torch.linalg.cholesky(cov)
                    yq = torch.linalg.solve_triangular(L, qc.t(), upper=False)
                    q_w_all.append(yq.t())
                q_w = torch.stack(q_w_all, dim=0)  # [B, Q, sub_dim]

                centers = self._kmeans_batch(
                    s_w,
                    n_clusters=r,
                    iters=int(self.corf_kmeans_iters),
                    max_points=None,
                    seed=int(self.map_seed) + 1234 + j,
                )  # [B, R, sub_dim]

                # Memberships
                def memberships(xw: torch.Tensor) -> torch.Tensor:
                    # xw: [B, N, d]
                    x2 = (xw ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
                    c2 = (centers ** 2).sum(dim=-1).unsqueeze(1)  # [B, 1, R]
                    dist2 = (x2 + c2 - 2.0 * torch.bmm(xw, centers.transpose(1, 2))).clamp_min(0.0)
                    m = torch.exp(-dist2 / (2.0 * (sigma ** 2)))
                    m = m / (m.sum(dim=-1, keepdim=True).clamp_min(1e-8))
                    return m

                ms = memberships(s_w)  # [B, S, R]
                mq = memberships(q_w)  # [B, Q, R]
                out_s.append(ms)
                out_q.append(mq)

            Zs = torch.cat(out_s, dim=-1)  # [B, S, Kf*R]
            Zq = torch.cat(out_q, dim=-1)  # [B, Q, Kf*R]

        # Normalize memberships (treat as features)
        Zs = F.normalize(Zs, p=2, dim=-1)
        Zq = F.normalize(Zq, p=2, dim=-1)
        return Zs, Zq

    def _tcf_forward_with_Q(self, patches: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """TCF forward with explicit per-episode prototypes.

        Args:
            patches: [N, Np, D]
            Q: [K, D]

        Returns:
            x_flat: [N, K*D]
            pooled: [N, D]
        """
        N, Np, D = patches.shape
        pooled = patches.mean(dim=1)

        E = F.normalize(patches, p=2, dim=-1)
        Qn0 = F.normalize(Q, p=2, dim=-1)
        Qn = Qn0.unsqueeze(0).expand(N, -1, -1)  # [N, K, D]
        K_tok = self.tcf_Wk(E)  # [N, Np, D]
        V_tok = self.tcf_Wv(E)  # [N, Np, D]

        scale = 1.0 / math.sqrt(float(D))
        attn_logits = torch.bmm(Qn, K_tok.transpose(1, 2)) * scale  # [N, K, Np]
        attn = torch.softmax(attn_logits, dim=-1)
        Z = torch.bmm(attn, V_tok)  # [N, K, D]
        x_flat = Z.reshape(N, -1)
        return x_flat, pooled

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Extract ViT patch tokens.

        Returns:
            patches: [B, Np, D]
        """
        # Our VisionTransformer supports `use_patches=True` and returns x[:, 1:]
        try:
            patches = self.backbone(x, use_patches=True)
        except TypeError:
            raise RuntimeError('TCF Adapter requires a ViT backbone that supports patch token extraction (use_patches=True).')
        if patches.dim() != 3:
            raise RuntimeError(f'Unexpected patch token shape: {tuple(patches.shape)}')
        return patches

    def _tcf_forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """TCF Adapter forward.

        Args:
            patches: [B, Np, D]

        Returns:
            x_flat: [B, K*D] flattened TCF features (MiniBLS input)
            pooled: [B, D] mean pooled patch feature (for weighting/graph)
        """
        B, Np, D = patches.shape
        pooled = patches.mean(dim=1)

        # Cross-attention: A = softmax(Q (E Wk)^T / sqrt(D))
        E = F.normalize(patches, p=2, dim=-1)
        Q0 = F.normalize(self.tcf_Q, p=2, dim=-1)
        Q = Q0.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        K_tok = self.tcf_Wk(E)  # [B, Np, D]
        V_tok = self.tcf_Wv(E)  # [B, Np, D]

        scale = 1.0 / math.sqrt(float(D))
        attn_logits = torch.bmm(Q, K_tok.transpose(1, 2)) * scale  # [B, K, Np]
        attn = torch.softmax(attn_logits, dim=-1)
        Z = torch.bmm(attn, V_tok)  # [B, K, D]
        x_flat = Z.reshape(B, -1)  # [B, K*D]
        return x_flat, pooled

    def _tcf_refine_Q_episode(
        self,
        support_patches_ep: torch.Tensor,
        support_labels_ep: torch.Tensor,
    ) -> torch.Tensor:
        """Refine TCF prototypes Q for a single episode using only the support set.

        We do a few steps of gradient descent on an auxiliary cosine-prototype
        classification loss computed on the TCF features of the support set.

        Args:
            support_patches_ep: [S, Np, D]
            support_labels_ep: [S]

        Returns:
            Q: [K, D] refined prototypes
        """
        steps = max(0, int(getattr(self, 'tcf_refine_steps', 0)))
        K = max(1, int(self.tcf_k))
        if steps <= 0:
            return self.tcf_Q

        lr = float(getattr(self, 'tcf_refine_lr', 0.5))
        lr = max(lr, 0.0)
        temp = float(getattr(self, 'tcf_refine_temp', 0.05))
        temp = max(temp, 1e-6)
        scale = 1.0 / temp

        # Detach patches to avoid any backbone gradient / graph bloat under eval.
        patches = support_patches_ep.detach()
        y = support_labels_ep.detach().long()
        C = int(self.num_classes)

        # Initialize Q by KMeans on support patch tokens (more stable than random global Q)
        with torch.cuda.amp.autocast(enabled=False):
            Np = int(patches.size(1))
            Dtok = int(patches.size(2))
            x_tok = patches.reshape(1, -1, Dtok)  # [1, S*Np, D]
            x_tok = F.normalize(x_tok.float(), p=2, dim=-1)
            Q0 = self._kmeans_batch(
                x_tok,
                n_clusters=K,
                iters=int(getattr(self, 'tcf_refine_kmeans_iters', 10)),
                max_points=int(getattr(self, 'tcf_refine_max_points', 1024)),
                seed=int(self.map_seed) + 333,
            )[0]
            Q = F.normalize(Q0, p=2, dim=-1).detach().clone().requires_grad_(True)

        # Inner-loop refinement on Q
        with torch.enable_grad():
            for _ in range(steps):
                x_sup, _ = self._tcf_forward_with_Q(patches, Q)  # [S, K*D]
                x_sup = F.normalize(x_sup.float(), p=2, dim=-1)

                # Episode prototypes in the TCF feature space
                protos = torch.zeros((C, x_sup.size(-1)), device=x_sup.device, dtype=x_sup.dtype)
                for c in range(C):
                    mask = (y == c)
                    if mask.any():
                        protos[c] = x_sup[mask].mean(dim=0)
                protos = F.normalize(protos, p=2, dim=-1)

                logits_aux = (x_sup @ protos.t()) * float(scale)
                loss = F.cross_entropy(logits_aux, y)
                grad = torch.autograd.grad(loss, Q, retain_graph=False, create_graph=False)[0]

                with torch.no_grad():
                    Q = (Q - lr * grad).detach()
                    Q = F.normalize(Q, p=2, dim=-1)
                    Q.requires_grad_(True)

        return Q.detach()

    def _solve_weighted_ridge(
        self,
        H: torch.Tensor,
        Y: torch.Tensor,
        weights: torch.Tensor,
        extra_reg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve weighted ridge regression in batch.

        Args:
            H: [B, N, D]
            Y: [B, N, C]
            weights: [B, N] (non-negative)

        Returns:
            W: [B, D, C]
        """
        eps = 1e-8
        w = weights.clamp_min(eps)

        # AMP-safe: do the linear algebra in float32
        # Auto choose primal/dual:
        # - If D >> N and no extra_reg, use the dual (N x N) system to enable huge widths.
        # - Otherwise, solve the primal (D x D) system.
        with torch.cuda.amp.autocast(enabled=False):
            H32 = H.float()
            Y32 = Y.float()
            sqrtw = torch.sqrt(w.float()).unsqueeze(-1)  # [B, N, 1]

            Hs = H32 * sqrtw
            Ys = Y32 * sqrtw

            # Optional SVD singular-value truncation on the weighted design matrix.
            # Motivation: when D is large and N is tiny (e.g., 25 supports), small singular
            # values often correspond to high-frequency/noisy directions. Hard-truncating
            # them can act like a denoiser complementary to ridge.
            if bool(getattr(self, 'svd_enable', False)) and extra_reg is None:
                drop = float(getattr(self, 'svd_drop', 0.0))
                drop = float(min(max(drop, 0.0), 0.95))
                energy = float(getattr(self, 'svd_energy', 0.0))
                min_rank = int(getattr(self, 'svd_min_rank', 1))
                min_rank = max(1, min_rank)

                # Batched economy SVD: Hs = U S Vh
                # Shapes: U [B, N, r], S [B, r], Vh [B, r, D], r=min(N,D)
                U, S, Vh = torch.linalg.svd(Hs, full_matrices=False)
                r = int(S.size(-1))

                keep = r
                if 0.0 < energy < 1.0:
                    # Energy ratio: sum_{i<=k} s_i^2 / sum s_i^2
                    s2 = S * S
                    cum = torch.cumsum(s2, dim=1)
                    total = cum[:, -1].clamp_min(1e-12)
                    ratio = cum / total.unsqueeze(1)
                    mask = ratio >= float(energy)
                    # first True index (+1) per batch; fallback to r
                    first_idx = mask.float().argmax(dim=1) + 1
                    k_b = torch.where(mask.any(dim=1), first_idx, torch.full_like(first_idx, r))
                    keep = int(max(min_rank, int(k_b.max().item())))
                elif drop > 0.0:
                    keep = int(max(min_rank, math.floor((1.0 - drop) * float(r))))

                keep = max(1, min(keep, r))
                U_k = U[:, :, :keep]
                S_k = S[:, :keep]
                Vh_k = Vh[:, :keep, :]
                Hs = torch.bmm(U_k * S_k.unsqueeze(1), Vh_k)

            B, N, D = Hs.shape

            # Static diagonal loading (baseline): lambda is shared across episodes.
            lam_b = torch.full((B,), float(self.reg_lambda), device=Hs.device, dtype=Hs.dtype)

            use_dual = (extra_reg is None) and (D > N)
            if use_dual:
                # Dual: W = Hs^T (Hs Hs^T + lam I)^{-1} Ys
                K = torch.bmm(Hs, Hs.transpose(1, 2))
                I = torch.eye(N, device=K.device, dtype=K.dtype).unsqueeze(0)
                K = K + lam_b.view(B, 1, 1) * I

                solve_device = str(getattr(self, 'solve_device', 'auto')).lower().strip()
                solve_dtype = str(getattr(self, 'solve_dtype', 'float32')).lower().strip()
                if solve_dtype == 'float64':
                    target_dtype = torch.float64
                else:
                    target_dtype = torch.float32

                if solve_device == 'cpu':
                    K_s = K.detach().to(device='cpu', dtype=target_dtype)
                    Ys_s = Ys.detach().to(device='cpu', dtype=target_dtype)
                    Lk, info = torch.linalg.cholesky_ex(K_s)
                    if torch.any(info != 0):
                        alpha = torch.linalg.solve(K_s, Ys_s)
                    else:
                        alpha = torch.cholesky_solve(Ys_s, Lk)
                    alpha = alpha.to(device=Hs.device, dtype=Hs.dtype)
                else:
                    # auto / cuda: solve on the current device
                    Lk, info = torch.linalg.cholesky_ex(K)
                    if torch.any(info != 0):
                        alpha = torch.linalg.solve(K, Ys)
                    else:
                        alpha = torch.cholesky_solve(Ys, Lk)

                W = torch.bmm(Hs.transpose(1, 2), alpha)  # [B, D, C]
            else:
                # Primal: W = (Hs^T Hs + lam I + extra_reg)^{-1} (Hs^T Ys)
                HT = Hs.transpose(1, 2)
                A = torch.bmm(HT, Hs)
                I = torch.eye(D, device=A.device, dtype=A.dtype).unsqueeze(0)
                A = A + lam_b.view(B, 1, 1) * I
                if extra_reg is not None:
                    A = A + extra_reg.float()
                Bmat = torch.bmm(HT, Ys)

                solve_device = str(getattr(self, 'solve_device', 'auto')).lower().strip()
                solve_dtype = str(getattr(self, 'solve_dtype', 'float32')).lower().strip()
                if solve_dtype == 'float64':
                    target_dtype = torch.float64
                else:
                    target_dtype = torch.float32

                if solve_device == 'cpu':
                    A_s = A.detach().to(device='cpu', dtype=target_dtype)
                    B_s = Bmat.detach().to(device='cpu', dtype=target_dtype)
                    L, info = torch.linalg.cholesky_ex(A_s)
                    if torch.any(info != 0):
                        W_s = torch.linalg.solve(A_s, B_s)
                    else:
                        W_s = torch.cholesky_solve(B_s, L)
                    W = W_s.to(device=Hs.device, dtype=Hs.dtype)
                else:
                    L, info = torch.linalg.cholesky_ex(A)
                    if torch.any(info != 0):
                        W = torch.linalg.solve(A, Bmat)
                    else:
                        W = torch.cholesky_solve(Bmat, L)

        return W

    def _knn_rbf_laplacian(self, feats: torch.Tensor) -> torch.Tensor:
        """Build a symmetric KNN-RBF graph Laplacian for each episode.

        Args:
            feats: [B, N, D] feature vectors (will be L2-normalized internally)

        Returns:
            L: [B, N, N] symmetric normalized Laplacian (float32)
        """
        B, N, _ = feats.shape
        k = int(self.graph_k)
        k = max(1, min(k, N - 1))
        sigma = float(self.graph_sigma)
        sigma = max(sigma, 1e-6)

        with torch.cuda.amp.autocast(enabled=False):
            x = F.normalize(feats.float(), p=2, dim=-1)  # [B, N, D]
            L_all = []
            for b in range(B):
                xb = x[b]  # [N, D]
                sim = xb @ xb.t()  # [N, N]
                dist2 = (2.0 - 2.0 * sim).clamp_min(0.0)
                dist2.fill_diagonal_(float('inf'))

                vals, idx = torch.topk(dist2, k=k, largest=False, dim=-1)
                w = torch.exp(-vals / (2.0 * (sigma ** 2)))

                S = torch.zeros((N, N), device=xb.device, dtype=xb.dtype)
                S.scatter_(1, idx, w)
                S = 0.5 * (S + S.t())
                # Symmetric normalized Laplacian: L = I - D^{-1/2} S D^{-1/2}
                d = S.sum(dim=1)  # [N]
                inv_sqrt_d = torch.rsqrt(d.clamp_min(1e-6))
                S_norm = S * inv_sqrt_d.unsqueeze(1) * inv_sqrt_d.unsqueeze(0)
                L = torch.eye(N, device=xb.device, dtype=xb.dtype) - S_norm
                L_all.append(L)
            L = torch.stack(L_all, dim=0)
        return L

    def _welsch_weights_from_residual(self, resid: torch.Tensor) -> torch.Tensor:
        """Welsch M-estimator weights: w = exp(-e^2 / (2*sigma^2)).

        Args:
            resid: [B, N] residual magnitudes e_i (non-negative)

        Returns:
            w: [B, N] in (0, 1]
        """
        sigma = float(self.mcc_sigma)
        sigma = max(sigma, 1e-6)
        return torch.exp(-(resid ** 2) / (2.0 * (sigma ** 2)))

    def _proto_cos_weights(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Prototype cosine similarity weights.

        Args:
            feats: [B, N, D] (assumed L2-normalized)
            labels: [B, N] in [0, C)

        Returns:
            weights: [B, N] in (0, 1]
        """
        B, N, D = feats.shape
        C = self.num_classes
        weights = torch.empty((B, N), device=feats.device, dtype=feats.dtype)
        for k in range(C):
            mask = (labels == k)
            if not mask.any():
                continue
            mask_f = mask.to(feats.dtype).unsqueeze(-1)  # [B, N, 1]
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            proto = (feats * mask_f).sum(dim=1) / denom  # [B, D]
            proto = F.normalize(proto, p=2, dim=-1)
            sim = (feats * proto.unsqueeze(1)).sum(dim=-1)  # [B, N]
            w = ((sim + 1.0) * 0.5).clamp_min(1e-4)
            weights = torch.where(mask, w, weights)
        return weights

    def _class_balance_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-sample class-balance multipliers based on support label counts.

        Args:
            labels: [B, N] long

        Returns:
            cb_w: [B, N] float, normalized so mean per episode = 1.0
        """
        B, N = labels.shape
        C = int(self.num_classes)
        # one-hot counts: [B, N, C]
        one = F.one_hot(labels.long().clamp_min(0).clamp_max(C - 1), num_classes=C).to(dtype=torch.float32)
        counts = one.sum(dim=1)  # [B, C]
        # Avoid div by zero
        counts = counts.clamp_min(1.0)
        if self.class_balanced_mode == 'inv':
            class_w = 1.0 / counts
        else:
            # default inv_sqrt
            class_w = 1.0 / torch.sqrt(counts)

        # per-sample multiplier
        idx = labels.long().unsqueeze(-1).clamp_min(0).clamp_max(C - 1)
        cb = class_w.gather(1, idx.squeeze(-1))  # gathering will broadcast shapes
        # ensure shape [B, N]
        cb = cb.view(B, N)
        # normalize mean per episode to 1 to keep regularization scale similar
        mean_cb = cb.mean(dim=1, keepdim=True).clamp_min(1e-6)
        cb = cb / mean_cb
        return cb.to(dtype=torch.float32)

    def _fuzzy_margin_weights(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Fuzzy confidence weights based on margin between nearest negative prototype and positive prototype.

        Following the scheme in docs: for each support sample i:
          d_pos = ||x_i - mu_{y_i}||
          d_neg = min_{c!=y_i} ||x_i - mu_c||
          m = d_neg - d_pos
          w = sigmoid(m / tau), then clamp to [w_min, 1]

        Args:
            feats: [B, N, D] (assumed L2-normalized)
            labels: [B, N]

        Returns:
            weights: [B, N]
        """
        B, N, D = feats.shape
        C = self.num_classes

        # Build prototypes per class
        protos = torch.zeros((B, C, D), device=feats.device, dtype=feats.dtype)
        counts = torch.zeros((B, C), device=feats.device, dtype=feats.dtype)
        for k in range(C):
            mask = (labels == k)  # [B, N]
            if not mask.any():
                continue
            mask_f = mask.to(feats.dtype).unsqueeze(-1)  # [B, N, 1]
            sum_k = (feats * mask_f).sum(dim=1)  # [B, D]
            cnt_k = mask_f.sum(dim=1).squeeze(-1)  # [B]
            protos[:, k, :] = sum_k / cnt_k.clamp_min(1.0).unsqueeze(-1)
            counts[:, k] = cnt_k
        protos = F.normalize(protos, p=2, dim=-1)

        # Distances from each sample to each prototype: [B, N, C]
        # Use Euclidean distance; feats and protos are normalized so this is monotonic w.r.t cosine.
        dist = torch.cdist(feats, protos, p=2)

        labels_long = labels.long().clamp_min(0).clamp_max(C - 1)
        idx = labels_long.unsqueeze(-1)  # [B, N, 1]
        d_pos = dist.gather(dim=-1, index=idx).squeeze(-1)  # [B, N]

        # Mask out the positive class when computing nearest negative
        pos_mask = F.one_hot(labels_long, num_classes=C).to(dist.dtype)  # [B, N, C]
        dist_neg = dist + pos_mask * 1e6
        d_neg = dist_neg.min(dim=-1).values  # [B, N]

        tau = float(self.margin_tau)
        tau = max(tau, 1e-6)
        margin = d_neg - d_pos
        w = torch.sigmoid(margin / tau)
        w = w.clamp(min=float(self.weight_min), max=1.0)
        return w

    def _mcc_weights_from_pred(self, pred_logits: torch.Tensor, Y_onehot: torch.Tensor) -> torch.Tensor:
        """Maximum correntropy (MCC) weights from residuals in probability space.

        Args:
            pred_logits: [B, N, C]
            Y_onehot: [B, N, C]

        Returns:
            weights: [B, N] in (0, 1]
        """
        # Residuals in softmax space are bounded and sigma is easier to tune.
        pred_prob = torch.softmax(pred_logits, dim=-1)
        resid = torch.linalg.vector_norm(pred_prob - Y_onehot, ord=2, dim=-1)  # [B, N]
        sigma = max(float(self.mcc_sigma), 1e-6)
        w = torch.exp(-(resid ** 2) / (2.0 * (sigma ** 2)))
        return w.clamp_min(1e-4)

    def _mcc_weights_from_trueclass(self, pred_logits: torch.Tensor, Y_onehot: torch.Tensor) -> torch.Tensor:
        """MCC weights using only the true-class probability residual.

        This tends to be better-behaved than vector residuals for few-shot heads.

        Args:
            pred_logits: [B, N, C]
            Y_onehot: [B, N, C]

        Returns:
            weights: [B, N] in (0, 1]
        """
        pred_prob = torch.softmax(pred_logits, dim=-1)
        p_true = (pred_prob * Y_onehot).sum(dim=-1)  # [B, N]
        resid = (1.0 - p_true).clamp_min(0.0)  # [B, N]
        sigma = max(float(self.mcc_sigma), 1e-6)
        w = torch.exp(-(resid ** 2) / (2.0 * (sigma ** 2)))
        return w.clamp_min(1e-4)

    def _fuzzy_cos_margin_weights(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Fuzzy confidence weights based on cosine margin (pos - nearest neg).

        Compared with `_fuzzy_margin_weights` (cdist-based), this is faster and the
        margin scale is more stable on L2-normalized features.

        Args:
            feats: [B, N, D] (L2-normalized)
            labels: [B, N]

        Returns:
            weights: [B, N]
        """
        B, N, D = feats.shape
        C = self.num_classes

        # Prototypes per class
        protos = torch.zeros((B, C, D), device=feats.device, dtype=feats.dtype)
        for k in range(C):
            mask = (labels == k)
            if not mask.any():
                continue
            mask_f = mask.to(feats.dtype).unsqueeze(-1)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            proto = (feats * mask_f).sum(dim=1) / denom
            protos[:, k, :] = proto
        protos = F.normalize(protos, p=2, dim=-1)

        # Cosine similarities: [B, N, C]
        sims = torch.bmm(feats, protos.transpose(1, 2))
        labels_long = labels.long().clamp_min(0).clamp_max(C - 1)
        s_pos = sims.gather(dim=-1, index=labels_long.unsqueeze(-1)).squeeze(-1)  # [B, N]

        pos_mask = F.one_hot(labels_long, num_classes=C).to(sims.dtype)
        sims_neg = sims - pos_mask * 1e6
        s_neg = sims_neg.max(dim=-1).values  # [B, N]

        tau = max(float(self.margin_tau), 1e-6)
        margin = s_pos - s_neg
        w = torch.sigmoid(margin / tau)
        w = w.clamp(min=float(self.weight_min), max=1.0)
        return w

    def forward(self, support_x, support_y, query_x):
        """
        这里面的逻辑是：先提取特征，然后【现场解方程】算出分类权重 W
        """
        # --- 步骤 0: 提取特征 (ViT Backbone) ---
        # 假设输入是图片 [Batch, N, C, H, W]
        # 我们要把 Batch 和 N 维度合并来过 Backbone，然后再拆开
        b, s_shot, c, h, w = support_x.shape
        q_shot = query_x.size(1)
        
        if bool(self.tcf_enable):
            # Use ViT patch tokens + TCF to build MiniBLS input.
            support_patches = self._extract_patch_tokens(support_x.view(-1, c, h, w))  # [B*s, Np, D]
            query_patches = self._extract_patch_tokens(query_x.view(-1, c, h, w))  # [B*q, Np, D]

            out_mode = str(getattr(self, 'tcf_out', 'flat'))
            K = max(1, int(self.tcf_k))

            if str(self.tcf_mode) == 'episode_kmeans':
                # Initialize per-episode Q by KMeans on support patch tokens (no training)
                Btot, Np, Dtok = support_patches.shape
                support_ep = support_patches.view(b, s_shot, Np, Dtok).reshape(b, s_shot * Np, Dtok)
                support_ep = F.normalize(support_ep, p=2, dim=-1)
                Q_ep = self._kmeans_batch(
                    support_ep,
                    n_clusters=int(self.tcf_k),
                    iters=10,
                    max_points=1024,
                    seed=int(self.map_seed) + 2025,
                )  # [b, K, D]

                # Run TCF per episode to apply the correct Q
                support_x_list = []
                support_pool_list = []
                query_x_list = []
                query_pool_list = []
                sp = support_patches.view(b, s_shot, Np, Dtok)
                qp = query_patches.view(b, q_shot, Np, Dtok)
                for bi in range(b):
                    xs, ps = self._tcf_forward_with_Q(sp[bi], Q_ep[bi])
                    xq, pq = self._tcf_forward_with_Q(qp[bi], Q_ep[bi])
                    if out_mode == 'mean':
                        xs = xs.view(xs.size(0), K, -1).mean(dim=1)
                        xq = xq.view(xq.size(0), K, -1).mean(dim=1)
                    support_x_list.append(xs)
                    support_pool_list.append(ps)
                    query_x_list.append(xq)
                    query_pool_list.append(pq)
                support_x_flat = torch.stack(support_x_list, dim=0)  # [b, s, K*D]
                query_x_flat = torch.stack(query_x_list, dim=0)  # [b, q, K*D]
                support_pooled = torch.stack(support_pool_list, dim=0)  # [b, s, D]
                query_pooled = torch.stack(query_pool_list, dim=0)  # [b, q, D]

                Z_support = support_x_flat
                Z_query = query_x_flat
                support_feat = support_pooled
                query_feat = query_pooled
            elif str(self.tcf_mode) == 'refine':
                # Refine per-episode Q using support labels (few inner steps), then apply to support+query.
                Btot, Np, Dtok = support_patches.shape
                sp = support_patches.view(b, s_shot, Np, Dtok)
                qp = query_patches.view(b, q_shot, Np, Dtok)

                support_x_list = []
                support_pool_list = []
                query_x_list = []
                query_pool_list = []

                for bi in range(b):
                    Q_ep = self._tcf_refine_Q_episode(sp[bi], support_y[bi])
                    xs, ps = self._tcf_forward_with_Q(sp[bi], Q_ep)
                    xq, pq = self._tcf_forward_with_Q(qp[bi], Q_ep)
                    if out_mode == 'mean':
                        xs = xs.view(xs.size(0), K, -1).mean(dim=1)
                        xq = xq.view(xq.size(0), K, -1).mean(dim=1)
                    support_x_list.append(xs)
                    support_pool_list.append(ps)
                    query_x_list.append(xq)
                    query_pool_list.append(pq)

                support_x_flat = torch.stack(support_x_list, dim=0)  # [b, s, K*D]
                query_x_flat = torch.stack(query_x_list, dim=0)  # [b, q, K*D]
                support_pooled = torch.stack(support_pool_list, dim=0)  # [b, s, D]
                query_pooled = torch.stack(query_pool_list, dim=0)  # [b, q, D]

                Z_support = support_x_flat
                Z_query = query_x_flat
                support_feat = support_pooled
                query_feat = query_pooled
            else:
                support_x_flat, support_pooled = self._tcf_forward(support_patches)  # [B*s, K*D], [B*s, D]
                query_x_flat, query_pooled = self._tcf_forward(query_patches)

                if out_mode == 'mean':
                    support_x_flat = support_x_flat.view(support_x_flat.size(0), K, -1).mean(dim=1)
                    query_x_flat = query_x_flat.view(query_x_flat.size(0), K, -1).mean(dim=1)

                # Restore episode shapes
                Z_support = support_x_flat.view(b, s_shot, -1)
                Z_query = query_x_flat.view(b, q_shot, -1)

                # Separate feature for robust weighting / graph (stable, lower-dim)
                support_feat = support_pooled.view(b, s_shot, -1)
                query_feat = query_pooled.view(b, q_shot, -1)
        else:
            # Standard: use CLS token feature as MiniBLS input
            if bool(getattr(self, 'mlf_enable', False)):
                # Robust Multi-Layer Feature Fusion:
                # - obtain all intermediate layer token outputs from the backbone
                # - support arbitrary positive/negative indices (clamped)
                # - GAP over patch tokens (exclude CLS) and concat in user-specified order
                sx = support_x.view(-1, c, h, w)
                qx = query_x.view(-1, c, h, w)
                try:
                    if not hasattr(self.backbone, 'get_intermediate_layers'):
                        raise RuntimeError('backbone does not support get_intermediate_layers')

                    # Request full depth to make indexing explicit and robust
                    depth = len(getattr(self.backbone, 'blocks', []))
                    if depth is None or depth <= 0:
                        depth = max(1, int(getattr(self, 'mlf_max_n', 1)))

                    layers_all_s = self.backbone.get_intermediate_layers(sx, n=depth)
                    layers_all_q = self.backbone.get_intermediate_layers(qx, n=depth)

                    # layers_all_* is a list of length `depth`, ordered from early->late blocks
                    d_avail = len(layers_all_s)

                    def clamp_idx(ind: int) -> int:
                        if ind >= 0:
                            return min(ind, d_avail - 1)
                        else:
                            # negative indices: clamp to [-d_avail, -1]
                            return max(ind, -d_avail)

                    pooled_s_list = []
                    pooled_q_list = []
                    cov_s_list = []
                    cov_q_list = []
                    used_layer_idxs = []
                    for user_idx in getattr(self, 'mlf_layers', [-1]):
                        si = clamp_idx(int(user_idx))
                        # Python list negative indexing supported directly
                        layer_s = layers_all_s[si]
                        layer_q = layers_all_q[si]
                        # exclude CLS token (index 0), GAP over patch tokens
                        ps = layer_s[:, 1:, :].mean(dim=1)
                        pq = layer_q[:, 1:, :].mean(dim=1)
                        pooled_s_list.append(ps)
                        pooled_q_list.append(pq)

                        # Optional: second-order covariance pooling over patch tokens
                        if bool(getattr(self, 'cov_enable', False)):
                            cov_s = self._cov_pool_sqrtm(layer_s[:, 1:, :])
                            cov_q = self._cov_pool_sqrtm(layer_q[:, 1:, :])
                            cov_s_list.append(cov_s)
                            cov_q_list.append(cov_q)
                        used_layer_idxs.append(si)

                    concat_s = torch.cat(pooled_s_list, dim=-1)
                    concat_q = torch.cat(pooled_q_list, dim=-1)

                    if bool(getattr(self, 'cov_enable', False)) and (len(cov_s_list) > 0):
                        concat_s = torch.cat([concat_s, torch.cat(cov_s_list, dim=-1)], dim=-1)
                        concat_q = torch.cat([concat_q, torch.cat(cov_q_list, dim=-1)], dim=-1)
                    Z_support = concat_s.view(b, s_shot, -1)
                    Z_query = concat_q.view(b, q_shot, -1)

                    # For weighting/graph use the semantic last-layer pooled (most semantic)
                    last_layer = layers_all_s[-1]
                    last_pooled_s = last_layer[:, 1:, :].mean(dim=1)
                    last_layer_q = layers_all_q[-1]
                    last_pooled_q = last_layer_q[:, 1:, :].mean(dim=1)
                    support_feat = last_pooled_s.view(b, s_shot, -1)
                    query_feat = last_pooled_q.view(b, q_shot, -1)

                    try:
                        _cov_en = bool(getattr(self, 'cov_enable', False))
                        _cov_proj_ok = self.cov_proj is not None
                        _cov_r = int(getattr(self, 'cov_proj_dim', 0))
                        _cov_added = int(len(cov_s_list))
                        if not getattr(self, '_mlf_debug_printed', False):
                            print(
                                f'[MiniBLS][MLF] fused_layers={getattr(self, "mlf_layers", None)}, '
                                f'used_idxs={used_layer_idxs}, concat_dim={Z_support.shape[-1]}, '
                                f'cov_enable={_cov_en}, cov_proj={_cov_proj_ok}, cov_r={_cov_r}, cov_layers_added={_cov_added}'
                            )
                            setattr(self, '_mlf_debug_printed', True)
                    except Exception:
                        pass
                except Exception:
                    # Fallback to CLS token if backbone doesn't support intermediate extraction
                    support_feat = self.backbone(support_x.view(-1, c, h, w))
                    query_feat = self.backbone(query_x.view(-1, c, h, w))
                    support_feat = support_feat.view(b, s_shot, -1)
                    query_feat = query_feat.view(b, q_shot, -1)
                    Z_support = support_feat
                    Z_query = query_feat
            else:
                support_feat = self.backbone(support_x.view(-1, c, h, w))
                query_feat = self.backbone(query_x.view(-1, c, h, w))

                support_feat = support_feat.view(b, s_shot, -1)
                query_feat = query_feat.view(b, q_shot, -1)
                Z_support = support_feat
                Z_query = query_feat

        # Keep raw pooled features for CorF (covariance/whitening needs pre-normalization structure)
        support_feat_raw = support_feat
        query_feat_raw = query_feat

        # --- 步骤 A: 不做L2归一化/去均值 ---
        # 保持原始尺度：用任务内方差正则(lambda随episode变化)来稳定闭式解。

        # --- (Optional) Feature Orthogonalization: remove common background direction ---
        if bool(getattr(self, 'ortho_enable', False)):
            Z_support, Z_query = self._apply_ortho_background(Z_support, Z_query)

        # --- (Optional) Power transform: L2 -> signed power -> L2 ---
        # Helps suppress heavy-tailed / outlier feature dimensions in cross-domain settings.
        if bool(getattr(self, 'power_transform_enable', False)):
            Z_support = self._apply_power_transform(Z_support)
            Z_query = self._apply_power_transform(Z_query)

        # --- (Optional) Correlated Fuzzy Mapping: concat memberships to Z_* ---
        if bool(self.corf_enable):
            Zs_corf, Zq_corf = self._corf_fit_transform(support_feat_raw, query_feat_raw)
            Z_support = torch.cat([Z_support, Zs_corf.to(dtype=Z_support.dtype, device=Z_support.device)], dim=-1)
            Z_query = torch.cat([Z_query, Zq_corf.to(dtype=Z_query.dtype, device=Z_query.device)], dim=-1)

        # --- (Optional) Residual Feature Refinement / Residual Classification ---
        # Applied after all feature-space transforms (ortho/power/corf) and before random mapping.
        # NOTE: refine mode only changes query features; classify_residual changes both.
        if str(getattr(self, 'residual_mode', 'none')).lower().strip() not in ('', 'none', 'off', 'false', '0'):
            Z_support, Z_query = self._apply_residual_mode(Z_support, Z_query, support_y)

        # --- Ensure mapping weights match current Z dim ---
        # Z dim can change when enabling MLF / covariance pooling / CorF.
        # mapping_weight is frozen random projection; if dims mismatch, re-init deterministically.
        z_dim = int(Z_support.shape[-1])
        if int(self.mapping_weight.shape[0]) != z_dim:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(int(getattr(self, 'map_seed', 777)))
            mapping_dim = int(self.mapping_weight.shape[1])
            W0 = torch.randn(z_dim, mapping_dim, generator=gen)
            W0 = W0 * (1.0 / math.sqrt(max(1, z_dim)))
            self.mapping_weight = nn.Parameter(W0, requires_grad=False)
            self._map_orthogonal_pending = bool(getattr(self, 'map_orthogonal', False))
            # Keep for debugging / consistency
            self.base_input_dim = int(z_dim)
            # Mapping dims changed: clear cached ensemble weights.
            try:
                self._ens_map_cache.clear()
            except Exception:
                pass

        # --- 步骤 C: 准备标签 (One-Hot) ---
        # 将标签转为 One-Hot 矩阵 [Batch, N, 5]
        Y_support = F.one_hot(support_y.long(), num_classes=self.num_classes).float()
        if float(self.label_relax) > 0.0:
            r = float(self.label_relax)
            C = int(self.num_classes)
            neg = -r / max(1, (C - 1))
            Y_support = Y_support * (1.0 + r - neg) + neg

        def _predict_logits_with_mapping(mapping_w: torch.Tensor) -> torch.Tensor:
            # --- Step B: random feature enhancement and wide design matrix ---
            if bool(getattr(self, 'map_orthogonal', False)) and bool(getattr(self, '_map_orthogonal_pending', False)):
                # Perform orthogonalization once, on-device (avoids heavy CPU QR at init).
                try:
                    with torch.no_grad():
                        self.mapping_weight.data = self._orthogonalize_columns(self.mapping_weight.data)
                    self._map_orthogonal_pending = False
                except Exception:
                    pass
            a_support = Z_support @ mapping_w + self.bls_mapping_bias
            a_query = Z_query @ mapping_w + self.bls_mapping_bias

            # --- (Optional) Virtual sample synthesis (inductive): augment support only ---
            # We generate ZV early so adaptive-Gaussian sigma can be estimated from real+virtual activations.
            ZV, FV, YV = self._virtual_synthesize(Z_support, support_feat, support_y)
            Nv = int(ZV.size(1))

            a_v = None
            if Nv > 0:
                a_v = ZV @ mapping_w + self.bls_mapping_bias

            # For adaptive gaussian/multi enhancement, some modes compute sigma once and reuse it.
            # Default mode 'adaptive' keeps the original per-call behavior (no sharing).
            sigma_ref = None
            try:
                et = str(getattr(self, 'enhance_type', 'tanh')).lower().strip()
                mode = str(getattr(self, 'gauss_sigma_mode', 'adaptive')).lower().strip()
                if et in ('gaussian', 'multi') and mode in ('adaptive_shared', 'adaptive_joint'):
                    eps = float(getattr(self, 'gauss_sigma_eps', 1e-3))
                    with torch.cuda.amp.autocast(enabled=False):
                        if mode == 'adaptive_joint':
                            a_cat = a_support.float()
                            # include query (unlabeled) for better match between support/query feature spaces
                            a_cat = torch.cat([a_cat, a_query.float()], dim=1)
                            if (a_v is not None) and (int(a_v.size(1)) > 0):
                                a_cat = torch.cat([a_cat, a_v.float()], dim=1)
                            sigma_ref = a_cat.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
                        else:
                            # adaptive_shared: estimate from (real support [+ virtual])
                            if (a_v is not None) and (int(a_v.size(1)) > 0):
                                a_cat = torch.cat([a_support.float(), a_v.float()], dim=1)
                                sigma_ref = a_cat.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
                            else:
                                sigma_ref = a_support.float().std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
                        sigma_ref = sigma_ref * float(getattr(self, 'gauss_sigma_scale', 1.0))
            except Exception:
                sigma_ref = None

            H_support, H_query = self._compute_enhancement(a_support, a_query, sigma_ref=sigma_ref)
            A_support = torch.cat([Z_support, H_support], dim=-1)
            A_query = torch.cat([Z_query, H_query], dim=-1)
            if Nv > 0:
                # Compute enhancement for virtual samples
                # a_v already computed above
                assert a_v is not None
                # Reuse the same enhancement rule; query tensor is dummy
                HV, _ = self._compute_enhancement(a_v, a_query[:, :0, :], sigma_ref=sigma_ref)
                A_v = torch.cat([ZV, HV], dim=-1)
                # Augment support matrices and labels
                A_support = torch.cat([A_support, A_v], dim=1)
                support_y_aug = torch.cat([support_y, YV.to(dtype=support_y.dtype)], dim=1)
                support_feat_aug = torch.cat([support_feat, FV.to(dtype=support_feat.dtype)], dim=1)
            else:
                support_y_aug = support_y
                support_feat_aug = support_feat

            # Episode-wise standardization (transductive: fit on support+query, apply to both)
            with torch.cuda.amp.autocast(enabled=False):
                A_all32 = torch.cat([A_support, A_query], dim=1).float()
                mu = A_all32.mean(dim=1, keepdim=True)
                std = A_all32.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
                A_support = ((A_support.float() - mu) / std).to(dtype=A_support.dtype)
                A_query = ((A_query.float() - mu) / std).to(dtype=A_query.dtype)

            # --- (Optional) Graph regularization term: lambda_1 * A_all^T L A_all ---
            extra_reg = None
            L_manifold = None
            if float(self.graph_lambda) > 0.0:
                feats_all = torch.cat([support_feat_aug, query_feat], dim=1)  # [B, N_all, D]
                L = self._knn_rbf_laplacian(feats_all)  # [B, N_all, N_all] (normalized)
                if float(self.graph_relax) > 0.0:
                    r = float(self.graph_relax)
                    N_all = L.size(-1)
                    I = torch.eye(N_all, device=L.device, dtype=L.dtype).unsqueeze(0)
                    L = L + r * I

                L_manifold = L
                p_dim = int(A_support.size(-1))
                if p_dim <= 1024:
                    A_all = torch.cat([A_support, A_query], dim=1)  # [B, N_all, p]
                    with torch.cuda.amp.autocast(enabled=False):
                        A_all_32 = A_all.float()
                        LA = torch.bmm(L, A_all_32)  # [B, N_all, p]
                        M = torch.bmm(A_all_32.transpose(1, 2), LA)  # [B, p, p]
                        extra_reg = float(self.graph_lambda) * M

            # --- Step D: ridge solve (supports robust variants) ---
            # Prepare one-hot for (possibly augmented) support
            Y_support_local = F.one_hot(support_y_aug.long(), num_classes=self.num_classes).float()
            if float(self.label_relax) > 0.0:
                rr = float(self.label_relax)
                Cc = int(self.num_classes)
                neg = -rr / max(1, (Cc - 1))
                Y_support_local = Y_support_local * (1.0 + rr - neg) + neg

            if self.robust_level <= 0:
                weights = torch.ones_like(support_y_aug, dtype=support_feat_aug.dtype, device=support_feat_aug.device)
                # Down-weight virtual samples (if any)
                if int(Nv) > 0:
                    vw = float(getattr(self, 'virtual_weight', 0.5))
                    weights[:, -Nv:] = weights[:, -Nv:] * vw
                if bool(getattr(self, 'class_balanced', False)):
                    cb = self._class_balance_weights(support_y_aug)
                    weights = weights * cb.to(dtype=weights.dtype, device=weights.device)
                W_out = self._solve_weighted_ridge(A_support, Y_support_local, weights, extra_reg=extra_reg)
            elif self.robust_level == 1:
                weights = self._proto_cos_weights(support_feat_aug, support_y_aug.long())
                if int(Nv) > 0:
                    vw = float(getattr(self, 'virtual_weight', 0.5))
                    weights[:, -Nv:] = weights[:, -Nv:] * vw
                if bool(getattr(self, 'class_balanced', False)):
                    cb = self._class_balance_weights(support_y_aug)
                    weights = weights * cb.to(dtype=weights.dtype, device=weights.device)
                W_out = self._solve_weighted_ridge(A_support, Y_support_local, weights, extra_reg=extra_reg)
            elif self.robust_level == 2:
                weights = torch.ones_like(support_y_aug, dtype=support_feat_aug.dtype, device=support_feat_aug.device)
                if int(Nv) > 0:
                    vw = float(getattr(self, 'virtual_weight', 0.5))
                    weights[:, -Nv:] = weights[:, -Nv:] * vw
                for _ in range(max(1, int(self.irls_iters))):
                    W_out = self._solve_weighted_ridge(A_support, Y_support_local, weights, extra_reg=extra_reg)
                    pred = torch.bmm(A_support, W_out)  # [B, N, C]
                    resid = torch.linalg.vector_norm(pred - Y_support_local, ord=2, dim=-1)  # [B, N]
                    delta = float(self.huber_delta)
                    w_new = torch.where(resid <= delta, torch.ones_like(resid), delta / (resid + 1e-8))
                    weights = w_new.clamp_min(1e-4)
                if bool(getattr(self, 'class_balanced', False)):
                    cb = self._class_balance_weights(support_y_aug)
                    weights = weights * cb.to(dtype=weights.dtype, device=weights.device)
            elif self.robust_level == 3:
                weights = self._fuzzy_margin_weights(support_feat_aug, support_y_aug.long())
                if int(Nv) > 0:
                    vw = float(getattr(self, 'virtual_weight', 0.5))
                    weights[:, -Nv:] = weights[:, -Nv:] * vw
                if bool(getattr(self, 'class_balanced', False)):
                    cb = self._class_balance_weights(support_y_aug)
                    weights = weights * cb.to(dtype=weights.dtype, device=weights.device)
                W_out = self._solve_weighted_ridge(A_support, Y_support_local, weights, extra_reg=extra_reg)
            elif self.robust_level == 4:
                weights_prior = self._fuzzy_margin_weights(support_feat_aug, support_y_aug.long())
                weights = weights_prior
                if int(Nv) > 0:
                    vw = float(getattr(self, 'virtual_weight', 0.5))
                    weights[:, -Nv:] = weights[:, -Nv:] * vw
                for _ in range(max(1, int(self.irls_iters))):
                    W_out = self._solve_weighted_ridge(A_support, Y_support_local, weights, extra_reg=extra_reg)
                    pred = torch.bmm(A_support, W_out)  # [B, N, C]
                    w_mcc = self._mcc_weights_from_pred(pred, Y_support_local)
                    weights = (weights_prior * w_mcc).clamp_min(1e-4)
                if bool(getattr(self, 'class_balanced', False)):
                    cb = self._class_balance_weights(support_y_aug)
                    weights = weights * cb.to(dtype=weights.dtype, device=weights.device)
            elif self.robust_level == 5:
                weights_prior = self._fuzzy_cos_margin_weights(support_feat_aug, support_y_aug.long())
                weights = weights_prior
                if int(Nv) > 0:
                    vw = float(getattr(self, 'virtual_weight', 0.5))
                    weights[:, -Nv:] = weights[:, -Nv:] * vw
                for _ in range(max(1, int(self.irls_iters))):
                    W_out = self._solve_weighted_ridge(A_support, Y_support_local, weights, extra_reg=extra_reg)
                    pred = torch.bmm(A_support, W_out)  # [B, N, C]
                    w_mcc = self._mcc_weights_from_trueclass(pred, Y_support_local)
                    weights = (weights_prior * w_mcc).clamp_min(1e-4)
                    w_mean = weights.mean(dim=1, keepdim=True).clamp_min(1e-6)
                    weights = weights / w_mean
                if bool(getattr(self, 'class_balanced', False)):
                    cb = self._class_balance_weights(support_y_aug)
                    weights = weights * cb.to(dtype=weights.dtype, device=weights.device)
            else:
                weights = torch.ones_like(support_y_aug, dtype=support_feat_aug.dtype, device=support_feat_aug.device)
                if int(Nv) > 0:
                    vw = float(getattr(self, 'virtual_weight', 0.5))
                    weights[:, -Nv:] = weights[:, -Nv:] * vw
                for _ in range(max(1, int(self.irls_iters))):
                    W_out = self._solve_weighted_ridge(A_support, Y_support_local, weights, extra_reg=extra_reg)
                    pred = torch.bmm(A_support, W_out)  # [B, N, C]
                    resid = torch.linalg.vector_norm(pred - Y_support_local, ord=2, dim=-1)  # [B, N]
                    w_new = self._welsch_weights_from_residual(resid)
                    weights = w_new.clamp_min(1e-4)
                if bool(getattr(self, 'class_balanced', False)):
                    cb = self._class_balance_weights(support_y_aug)
                    weights = weights * cb.to(dtype=weights.dtype, device=weights.device)

            # --- (Optional) Transductive self-training refinement ---
            if float(getattr(self, 'self_train_alpha', 0.0)) > 0.0:
                alpha = float(self.self_train_alpha)
                iters = max(1, int(getattr(self, 'self_train_iters', 1)))
                temp = float(getattr(self, 'self_train_temp', 1.0))
                conf_thr = float(getattr(self, 'self_train_conf_thr', 0.0))

                w_sup = weights
                A_all = torch.cat([A_support, A_query], dim=1)
                for _ in range(iters):
                    with torch.cuda.amp.autocast(enabled=False):
                        logits_q = torch.bmm(A_query.float(), W_out.float())  # [B, Nq, C]
                        if temp != 1.0:
                            logits_q = logits_q / max(1e-6, temp)
                        prob_q = torch.softmax(logits_q, dim=-1)
                        conf_q = prob_q.max(dim=-1).values  # [B, Nq]
                        pred_q = prob_q.argmax(dim=-1)  # [B, Nq]

                        if conf_thr > 0.0:
                            mask = (conf_q >= conf_thr).float()
                        else:
                            mask = torch.ones_like(conf_q)

                        w_q = (alpha * conf_q * mask).clamp_min(0.0)
                        Y_q = torch.nn.functional.one_hot(pred_q, num_classes=prob_q.shape[-1]).to(prob_q.dtype)

                        sum_w_per_class = torch.zeros(
                            (w_q.shape[0], prob_q.shape[-1]), device=w_q.device, dtype=w_q.dtype
                        )
                        sum_w_per_class.scatter_add_(1, pred_q, w_q)
                        mean_sum_w = sum_w_per_class.mean(dim=1, keepdim=True)
                        scale = mean_sum_w / (sum_w_per_class + 1e-6)
                        w_q = w_q * scale.gather(1, pred_q)

                        Y_all = torch.cat([Y_support_local.float(), Y_q.float()], dim=1)
                        w_all = torch.cat([w_sup.float(), w_q.float()], dim=1)

                        W_out = self._solve_weighted_ridge(
                            A_all.to(dtype=A_support.dtype),
                            Y_all.to(dtype=Y_support_local.dtype),
                            w_all.to(dtype=w_sup.dtype),
                            extra_reg=extra_reg,
                        )

            logits = torch.bmm(A_query, W_out.to(dtype=A_query.dtype))

            # --- (Optional) Manifold refinement ---
            if (L_manifold is not None) and (float(self.graph_lambda) > 0.0):
                with torch.cuda.amp.autocast(enabled=False):
                    Ns = int(A_support.size(1))
                    Nq = int(A_query.size(1))
                    L32 = L_manifold.float()
                    L_uu = L32[:, Ns:, Ns:]  # [B, Nq, Nq]
                    L_ul = L32[:, Ns:, :Ns]  # [B, Nq, Ns]
                    rhs = -torch.bmm(L_ul, Y_support.float())  # [B, Nq, C]

                    eps = 1e-3
                    Iu = torch.eye(Nq, device=L_uu.device, dtype=L_uu.dtype).unsqueeze(0)
                    Fu = torch.linalg.solve(L_uu + eps * Iu, rhs)  # [B, Nq, C]
                    Fu = Fu.clamp_min(0.0)
                    lp_prob = Fu / Fu.sum(dim=-1, keepdim=True).clamp_min(1e-6)

                    cls_prob = torch.softmax(logits.float(), dim=-1)
                    if extra_reg is not None:
                        beta = float(max(0.0, min(0.2, float(self.graph_lambda) * 20.0)))
                    else:
                        beta = float(max(0.0, min(0.5, float(self.graph_lambda) * 50.0)))
                    prob = (1.0 - beta) * cls_prob + beta * lp_prob
                    logits = torch.log(prob.clamp_min(1e-8)).to(dtype=logits.dtype)

            return logits

        ens = max(1, int(getattr(self, 'ensemble', 1)))
        if ens <= 1:
            logits = _predict_logits_with_mapping(self.mapping_weight)
        else:
            logits_acc = None
            for m in range(ens):
                if m == 0:
                    mw = self.mapping_weight
                else:
                    mw = self._get_ensemble_mapping_weight(
                        z_dim=z_dim,
                        member=m,
                        device=Z_support.device,
                        dtype=Z_support.dtype,
                    )
                l = _predict_logits_with_mapping(mw)
                if logits_acc is None:
                    logits_acc = l.float()
                else:
                    logits_acc = logits_acc + l.float()
            logits = (logits_acc / float(ens)).to(dtype=Z_support.dtype)
        
        # 如果原来的代码期待 [Batch*Query, 5] 的形状，这里展平
        return logits.view(-1, self.num_classes)
