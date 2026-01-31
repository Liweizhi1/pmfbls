import os
os.environ[
"KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import time
import random
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path
from tabulate import tabulate

from engine import evaluate
import utils.deit_util as utils
from utils.args import get_args_parser
from models import get_model
from datasets import get_bscd_loader


def main(args):
    args.distributed = False # CDFSL dataloader doesn't support DDP
    args.eval = True
    args.fp16 = False
    print(">>> 强制使用 FP32 精度 (已关闭 FP16)")
    # Make main() runnable when called programmatically (e.g., from scripts/run_repro.py)
    # where the __main__ block (that usually sets these) is not executed.
    global output_dir
    if 'output_dir' not in globals() or output_dir is None:
        output_dir = Path(getattr(args, 'output_dir', 'outputs/tmp'))
        output_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(args, 'train_tag'):
        args.train_tag = 'pt' if getattr(args, 'resume', '') == '' else 'ep'
        args.train_tag += f"_step{getattr(args, 'ada_steps', 0)}_lr{getattr(args, 'ada_lr', 0)}_prob{getattr(args, 'aug_prob', 0)}"

    # print(args)  # 关闭参数整坨打印
    device = torch.device(args.device)

    # Make evaluation deterministic for reproducibility
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    ##############################################
    # Model
    print(f"Creating model: {args.deploy} {args.arch}")

    model = get_model(args)
    model.to(device)

    # TTA support expansion has been removed; no propagation necessary.

    # Note: backbone parameter freezing removed — backbone kept as originally defined

    # --- [Correction 1: Smart Weight Loading Logic] ---
    if args.resume:
        print(f"Loading checkpoint from: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        state_dict = checkpoint['model']
        
        # Smartly correct Key names
        # Strip common prefixes to align with the current model structure
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith("module."):
                new_key = new_key[7:]
            if new_key.startswith("backbone."):
                new_key = new_key[9:]
            if new_key.startswith("feature_extractor."):
                new_key = new_key[18:]
            new_state_dict[new_key] = v

        # Load weights specifically into the backbone
        # This avoids conflicts with the new BLS head
        if hasattr(model, 'backbone'):
            msg = model.backbone.load_state_dict(new_state_dict, strict=False)
            print("✅ Backbone loaded via model.backbone")
        else:
            # Fallback if the model itself is just the backbone
            msg = model.load_state_dict(new_state_dict, strict=False)
            print("✅ Model loaded directly")

        # Validation check
        backbone_missing = [k for k in msg.missing_keys if 'blocks' in k or 'patch_embed' in k]
        if len(backbone_missing) > 0:
            print(f"❌ WARNING: Backbone weights might be missing! Count: {len(backbone_missing)}")
            print(f"Sample missing keys: {backbone_missing[:3]}")
        else:
            print(f"🎉 Backbone loaded successfully! (Missing keys for Head/BLS are expected)")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    ##############################################
    # Test
    criterion = torch.nn.CrossEntropyLoss()
    #datasets = ["EuroSAT", "ISIC", "CropDisease", "ChestX"]
    datasets = args.cdfsl_domains
    var_accs = {}

    for domain in datasets:
        print(f'\n# Testing {domain} starts...\n')

        data_loader_val = get_bscd_loader(
            domain,
            args.test_n_way,
            args.n_shot,
            args.image_size,
            iter_num=args.bscd_iter_num,
            seed=int(args.seed) + 10000,
            fixed_episode_file=getattr(args, 'fixed_episode_file', ''),
        )

        # validate lr
        best_lr = args.ada_lr
        if args.deploy == 'finetune':
            print("Start selecting the best lr...")
            best_acc = 0
            # For finetuning, we need gradients, so no_grad is NOT used here
            for lr in [0, 0.0001, 0.0005, 0.001]:
                model.lr = lr
                test_stats = evaluate(data_loader_val, model, criterion, device, seed=1234, ep=5)
                acc = test_stats['acc1']
                print(f"*lr = {lr}: acc1 = {acc}")
                if acc > best_acc:
                    best_acc = acc
                    best_lr = lr
            model.lr = best_lr
            print(f"### Selected lr = {best_lr}")

        # --- [Correction 2: Fast Evaluation Mode] ---
        # final classification (loader already seeded above)
        
        # Force model to eval mode and disable gradient calculation
        model.eval() 
        with torch.no_grad():
            model = model.cuda() # Ensure model is on the correct device
            test_stats = evaluate(data_loader_val, model, criterion, device)
            
        var_accs[domain] = (test_stats['acc1'], test_stats['acc_std'], best_lr)

        print(f"{domain}: acc1 on {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%")

        if args.output_dir and utils.is_main_process():
            test_stats['domain'] = domain
            test_stats['lr'] = best_lr
            with (output_dir / f"log_test_{args.deploy}_{args.train_tag}.txt").open("a") as f:
                f.write(json.dumps(test_stats) + "\n")

    # print results as a table
    if utils.is_main_process():
        rows = []
        for dataset_name in datasets:
            row = [dataset_name]
            acc, std, lr = var_accs[dataset_name]
            conf = (1.96 * std) / np.sqrt(len(data_loader_val.dataset))
            row.append(f"{acc:0.2f} +- {conf:0.2f}")
            row.append(f"{lr}")
            rows.append(row)
        np.save(os.path.join(output_dir, f'test_results_{args.deploy}_{args.train_tag}.npy'), {'rows': rows})

        table = tabulate(rows, headers=['Domain', args.arch, 'lr'], floatfmt=".2f")
        print(table)
        print("\n")

        if args.output_dir:
            with (output_dir / f"log_test_{args.deploy}_{args.train_tag}.txt").open("a") as f:
                f.write(table)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    #from utils.args import set_global_args
    #set_global_args(args)
    args.train_tag = 'pt' if args.resume == '' else 'ep'
    args.train_tag += f'_step{args.ada_steps}_lr{args.ada_lr}_prob{args.aug_prob}'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import sys
    with (output_dir / f"log_test_{args.deploy}_{args.train_tag}.txt").open("a") as f:
        f.write(" ".join(sys.argv) + "\n")

    main(args)
