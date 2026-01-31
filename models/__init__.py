import os
import numpy as np
import torch
#from timm.models import create_model
from .protonet import ProtoNet
# Defer importing heavy deploy variants (which may pull timm/torchvision)
# to runtime inside `get_model` to avoid importing optional heavy deps at module import time.


def get_backbone(args):
    if args.arch == 'vit_base_patch16_224_in21k':
        from .vit_google import VisionTransformer, CONFIGS

        config = CONFIGS['ViT-B_16']
        model = VisionTransformer(config, 224)

        url = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'
        pretrained_weights = 'pretrained_ckpts/vit_base_patch16_224_in21k.npz'

        if not os.path.exists(pretrained_weights):
            try:
                import wget
                os.makedirs('pretrained_ckpts', exist_ok=True)
                wget.download(url, pretrained_weights)
            except:
                print(f'Cannot download pretrained weights from {url}. Check if `pip install wget` works.')

        model.load_from(np.load(pretrained_weights))
        print('Pretrained weights found at {}'.format(pretrained_weights))

    elif args.arch == 'vit_base_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)

        if not args.no_pretrain:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

            model.load_state_dict(state_dict, strict=True)
            print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'dino_base_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))

    # Accept architecture name with explicit resolution suffix used elsewhere
    elif args.arch == 'dino_base_patch16_224':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'deit_base_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        url = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'deit_small_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)
        url = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'dino_small_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)

        if not args.no_pretrain:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

            model.load_state_dict(state_dict, strict=True)
            print('Pretrained weights found at {}'.format(url))

    elif args.arch == 'beit_base_patch16_224_pt22k':
        from .beit import default_pretrained_model
        model = default_pretrained_model(args)
        print('Pretrained BEiT loaded')

    elif args.arch == 'clip_base_patch16_224':
        from . import clip
        model, _ = clip.load('ViT-B/16', 'cpu')

    elif args.arch == 'clip_resnet50':
        from . import clip
        model, _ = clip.load('RN50', 'cpu')

    elif args.arch == 'dino_resnet50':
        from torchvision.models.resnet import resnet50

        model = resnet50(pretrained=False)
        model.fc = torch.nn.Identity()

        if not args.no_pretrain:
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
                map_location="cpu",
            )
            model.load_state_dict(state_dict, strict=False)

    elif args.arch == 'resnet50':
        from torchvision.models.resnet import resnet50

        pretrained = not args.no_pretrain
        model = resnet50(pretrained=pretrained)
        model.fc = torch.nn.Identity()

    elif args.arch == 'resnet18':
        from torchvision.models.resnet import resnet18

        pretrained = not args.no_pretrain
        model = resnet18(pretrained=pretrained)
        model.fc = torch.nn.Identity()

    elif args.arch == 'dino_xcit_medium_24_p16':
        model = torch.hub.load('facebookresearch/xcit:main', 'xcit_medium_24_p16')
        model.head = torch.nn.Identity()
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)

    elif args.arch == 'dino_xcit_medium_24_p8':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8')

    elif args.arch == 'simclrv2_resnet50':
        import sys
        sys.path.insert(
            0,
            'cog',
        )
        import model_utils

        model_utils.MODELS_ROOT_DIR = 'cog/models'
        ckpt_file = os.path.join(args.pretrained_checkpoint_path, 'pretrained_ckpts/simclrv2_resnet50.pth')
        resnet, _ = model_utils.load_pretrained_backbone(args.arch, ckpt_file)

        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super(Wrapper, self).__init__()
                self.model = model

            def forward(self, x):
                return self.model(x, apply_fc=False)

        model = Wrapper(resnet)

    elif args.arch in ['mocov2_resnet50', 'swav_resnet50', 'barlow_resnet50']:
        from torchvision.models.resnet import resnet50

        model = resnet50(pretrained=False)
        ckpt_file = os.path.join(args.pretrained_checkpoint_path, 'pretrained_ckpts_converted/{}.pth'.format(args.arch))
        ckpt = torch.load(ckpt_file)

        msg = model.load_state_dict(ckpt, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        # remove the fully-connected layer
        model.fc = torch.nn.Identity()

    else:
        raise ValueError(f'{args.arch} is not conisdered in the current code.')

    return model


def get_model(args):
    backbone = get_backbone(args)

    if args.deploy == 'vanilla':
        model = ProtoNet(backbone)
    elif args.deploy == 'finetune':
        from .deploy import ProtoNet_Finetune
        model = ProtoNet_Finetune(backbone, args.ada_steps, args.ada_lr, args.aug_prob, args.aug_types)
    elif args.deploy == 'finetune_autolr':
        from .deploy import ProtoNet_Auto_Finetune
        model = ProtoNet_Auto_Finetune(backbone, args.ada_steps, args.aug_prob, args.aug_types)
    elif args.deploy == 'ada_tokens':
        from .deploy import ProtoNet_AdaTok
        model = ProtoNet_AdaTok(backbone, args.num_adapters,
                                args.ada_steps, args.ada_lr)
    elif args.deploy == 'ada_tokens_entmin':
        from .deploy import ProtoNet_AdaTok_EntMin
        model = ProtoNet_AdaTok_EntMin(backbone, args.num_adapters,
                                       args.ada_steps, args.ada_lr)
    elif args.deploy == 'drbls':
        from .deploy_drbls import ProtoNet_DRBLS
        lam1 = getattr(args, 'drbls_lam1', 1.0)
        lam2 = getattr(args, 'drbls_lam2', 1.0)
        lam3 = getattr(args, 'drbls_lam3', 0.1)
        model = ProtoNet_DRBLS(backbone, lam1, lam2, lam3)
    elif args.deploy == 'dpbml':
        from .deploy_dpbml import ProtoNet_DPBML
        n_z = getattr(args, 'dpbml_n_z', 10)
        n_h = getattr(args, 'dpbml_n_h', 10)
        n_feature_map = getattr(args, 'dpbml_n_feature_map', 10)
        n_enhance_map = getattr(args, 'dpbml_n_enhance_map', 10)
        lam1 = getattr(args, 'dpbml_lam1', 1.0)
        lam2 = getattr(args, 'dpbml_lam2', 1.0)
        lam3 = getattr(args, 'dpbml_lam3', 0.1)
        max_iter = getattr(args, 'dpbml_max_iter', 10)
        model = ProtoNet_DPBML(backbone, n_z, n_h, n_feature_map, n_enhance_map,
                               lam1, lam2, lam3, max_iter,
                               use_fuzzy=True, fuzzy_tau=1.0, fuzzy_w_min=0.2)
    elif args.deploy == 'tgbls':
        from .deploy_tgbls import ProtoNet_TGBLS
        n_z = getattr(args, 'tgbls_n_z', 10)
        n_h = getattr(args, 'tgbls_n_h', 10)
        n_feature_map = getattr(args, 'tgbls_n_feature_map', 10)
        n_enhance_map = getattr(args, 'tgbls_n_enhance_map', 10)
        lam1 = getattr(args, 'tgbls_lam1', 1.0)
        lam2 = getattr(args, 'tgbls_lam2', 10.0)
        k_neighbor = getattr(args, 'tgbls_k', 5)
        model = ProtoNet_TGBLS(backbone, n_z, n_h, n_feature_map, n_enhance_map, lam1, lam2, k_neighbor)
    elif args.deploy == 'bls_online':
        from .deploy_bls_online import ProtoNet_BLSOnline
        ridge_lambda = getattr(args, 'bls_lambda', 1.0)
        model = ProtoNet_BLSOnline(backbone, ridge_lambda=ridge_lambda)
    elif args.deploy == 'bls':
        from .deploy_bls import ProtoNet_BLS
        num_win = getattr(args, 'bls_num_win', 10)
        num_feat = getattr(args, 'bls_num_feat', 10)
        num_enhan = getattr(args, 'bls_num_enhan', 100)
        s = getattr(args, 'bls_s', 0.5)
        c = getattr(args, 'bls_c', 2**-30)
        model = ProtoNet_BLS(backbone, num_win, num_feat, num_enhan, s, c)
    elif args.deploy == 'mini_bls':
        # --- 替换为 bls_ultimate ---
        from .bls_ultimate import ProtoNet_MiniBLS_Ultimate
        
        # 自动补全默认参数 (防止命令行没传参报错)
        if not hasattr(args, 'mini_bls_mlf_layers'):
            args.mini_bls_mlf_layers = "-1,-3,-6"
        if not hasattr(args, 'mini_bls_mapping_dim'):
            args.mini_bls_mapping_dim = 1000
        if not hasattr(args, 'mini_bls_reg_lambda'):
            args.mini_bls_reg_lambda = 1e-3
        
        # 处理流开关（优先级：disable > enable）
        if not hasattr(args, 'mini_bls_stream1_enable'):
            args.mini_bls_stream1_enable = True
        if hasattr(args, 'mini_bls_stream1_disable') and args.mini_bls_stream1_disable:
            args.mini_bls_stream1_enable = False
            
        if not hasattr(args, 'mini_bls_stream2_enable'):
            args.mini_bls_stream2_enable = True
        if hasattr(args, 'mini_bls_stream2_disable') and args.mini_bls_stream2_disable:
            args.mini_bls_stream2_enable = False
            
        if not hasattr(args, 'mini_bls_stream3_enable'):
            args.mini_bls_stream3_enable = True
        if hasattr(args, 'mini_bls_stream3_disable') and args.mini_bls_stream3_disable:
            args.mini_bls_stream3_enable = False
            
        if not hasattr(args, 'mini_bls_beta_correction'):
            args.mini_bls_beta_correction = 0.5
        if not hasattr(args, 'mini_bls_activation'):
            args.mini_bls_activation = 'relu'
        if not hasattr(args, 'mini_bls_scale'):
            args.mini_bls_scale = 1.0

        if not hasattr(args, 'mini_bls_bls_w_scale'):
            args.mini_bls_bls_w_scale = 2.0
        if not hasattr(args, 'mini_bls_bls_noise'):
            args.mini_bls_bls_noise = 0.05

        if not hasattr(args, 'mini_bls_s3_mode'):
            args.mini_bls_s3_mode = 'concat'
        if not hasattr(args, 'mini_bls_s3_gamma'):
            args.mini_bls_s3_gamma = 1.0
            
        model = ProtoNet_MiniBLS_Ultimate(backbone, args)
    else:
        raise ValueError(f'deploy method {args.deploy} is not supported.')
    return model