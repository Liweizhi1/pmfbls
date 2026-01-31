import math
import sys
import warnings
import os
import time
import numpy as np
from typing import Iterable, Optional
from contextlib import nullcontext  # 必需：用于禁用 autocast

import torch
from torch.utils.tensorboard import SummaryWriter

# Protect heavy timm imports which may trigger torchvision import errors
try:
    from timm.data import Mixup
    from timm.utils import ModelEma, accuracy
except Exception:
    Mixup = None
    ModelEma = None

    def accuracy(output, target, topk=(1,)):
        """Local fallback top-k accuracy compatible with timm.utils.accuracy."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return tuple(res)

import utils.deit_util as utils
from utils import AverageMeter, to_device

def train_one_epoch(data_loader: Iterable,
                    model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    device: torch.device,
                    loss_scaler = None,
                    fp16: bool = False,
                    max_norm: float = 0, # clip_grad
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    writer: Optional[SummaryWriter] = None,
                    set_training_mode=True):

    global_step = epoch * len(data_loader)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    model.train(set_training_mode)

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch = to_device(batch, device)
        SupportTensor, SupportLabel, x, y = batch

        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        # forward
        with torch.cuda.amp.autocast(enabled=fp16):
            output = model(SupportTensor, SupportLabel, x)

        output = output.view(x.shape[0] * x.shape[1], -1)
        y = y.view(-1)
        loss = criterion(output, y)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if fp16:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)
        metric_logger.update(n_ways=SupportLabel.max()+1)
        metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])

        if utils.is_main_process() and global_step % print_freq == 0:
            writer.add_scalar("train/loss", scalar_value=loss_value, global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=lr, global_step=global_step)

        global_step += 1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(data_loaders, model, criterion, device, seed=None, ep=None):
    if isinstance(data_loaders, dict):
        test_stats_lst = {}
        test_stats_glb = {}

        for j, (source, data_loader) in enumerate(data_loaders.items()):
            print(f'* Evaluating {source}:')
            seed_j = seed + j if seed else None
            test_stats = _evaluate(data_loader, model, criterion, device, seed_j)
            test_stats_lst[source] = test_stats
            test_stats_glb[source] = test_stats['acc1']

        for k in test_stats_lst[source].keys():
            test_stats_glb[k] = torch.tensor([test_stats[k] for test_stats in test_stats_lst.values()]).mean().item()

        return test_stats_glb
    elif isinstance(data_loaders, torch.utils.data.DataLoader):
        return _evaluate(data_loaders, model, criterion, device, seed, ep)
    else:
        # 处理那个 annoying 的 warning，直接忽略它继续运行
        return _evaluate(data_loaders, model, criterion, device, seed)


@torch.no_grad()
def _evaluate(data_loader, model, criterion, device, seed=None, ep=None):
    # ==============================================================
    # 核心修复：深度强制搬运
    # 针对 bls_ultimate 这种嵌套结构，手动把内部模块搬到 GPU
    # ==============================================================
    if torch.cuda.is_available():
        # 1. 搬运主模型
        model = model.cuda()
        
        # 2. 检查并搬运 model.extractor
        if hasattr(model, 'extractor'):
            # 如果 extractor 有 .to 方法（是 nn.Module），搬运它
            if hasattr(model.extractor, 'to'):
                model.extractor.to('cuda')
            
            # 3. 检查并搬运 model.extractor.backbone (这里是报错的根源)
            if hasattr(model.extractor, 'backbone') and hasattr(model.extractor.backbone, 'to'):
                # print("Detected internal backbone, moving to CUDA...") 
                model.extractor.backbone.to('cuda')
    # ==============================================================

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('acc1', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    metric_logger.add_meter('acc5', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    header = 'Test:'

    model.eval()

    if seed is not None:
        data_loader.generator.manual_seed(seed)

    per_episode_accs = []
    
    # 确保 data_loader 是可迭代的
    if not isinstance(data_loader, Iterable):
         print(f"Warning: data_loader {type(data_loader)} might not be iterable properly.")

    for ii, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if ep is not None:
            if ii > ep:
                break
        
        # 将数据搬到 device (GPU)
        batch = to_device(batch, device)
        SupportTensor, SupportLabel, x, y = batch

        # 再次确认 input 的位置，用于 debug
        # if ii == 0:
        #     print(f"DEBUG: Input x device: {x.device}")
        
        # 计算结果 - 使用 nullcontext 彻底禁用 autocast
        with nullcontext():
            try:
                output = model(SupportTensor, SupportLabel, x)
            except RuntimeError as e:
                # 如果还是报错，打印出详细信息帮助排查，而不是直接崩溃
                if "Input type" in str(e) and "weight type" in str(e):
                    print("\n!!! CRITICAL ERROR DETECTED !!!")
                    print(f"Input tensor device: {x.device}")
                    try:
                        # 尝试打印模型参数位置
                        print(f"Model param device: {next(model.parameters()).device}")
                        if hasattr(model, 'extractor') and hasattr(model.extractor, 'backbone'):
                            print(f"Backbone param device: {next(model.extractor.backbone.parameters()).device}")
                    except:
                        pass
                    print("Try restarting kernel or checking model definition.")
                raise e

        output = output.view(-1, output.shape[-1])
        y = y.view(-1)
        loss = criterion(output, y)
        acc1, acc5 = accuracy(output, y, topk=(1, 5))

        batch_size = x.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        try:
            per_episode_accs.append(float(acc1.item()))
        except Exception:
            pass
        metric_logger.update(n_ways=SupportLabel.max()+1)
        metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    ret_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    ret_dict['acc_std'] = metric_logger.meters['acc1'].std
    
    per_out = os.getenv('PER_EPISODE_OUT', '')
    if per_out:
        try:
            os.makedirs(per_out, exist_ok=True)
            seed_str = str(seed) if seed is not None else 'nos'
            fn = os.path.join(per_out, f'per_episode_accs_seed{seed_str}_{int(time.time())}.npy')
            np.save(fn, np.array(per_episode_accs, dtype=float))
        except Exception as e:
            print('Failed to save per-episode accs:', e)

    return ret_dict