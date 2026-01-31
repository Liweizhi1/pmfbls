# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from .additional_transforms import ImageJitter
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================================================================
# 路径配置：已设为你的本地路径，使用 r 确保路径解析正确
# =========================================================================
CropDisease_path = r"C:\Users\李维志\Desktop\pmfbls\bls\pmf_cvpr22\data\CropDisease"
identity = lambda x:x

# =========================================================================
# 普通数据集：用于测试全集
# =========================================================================
class SimpleDataset:
    def __init__(self, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform
        self.meta = {'image_names': [], 'image_labels': []}

        if not os.path.exists(CropDisease_path):
            raise RuntimeError(f"未找到数据集路径: {CropDisease_path}")

        # 【核心优化】使用 d.imgs 只获取路径清单，不加载图片本身
        d = ImageFolder(CropDisease_path)
        for img_path, label in d.imgs:
            self.meta['image_names'].append(img_path)
            self.meta['image_labels'].append(label)

    def __getitem__(self, i):
        # 只有在这里被调用时，才真正打开硬盘上的图片
        img = Image.open(self.meta['image_names'][i]).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

# =========================================================================
# Few-Shot 数据集：用于采样 Episodic 任务
# =========================================================================
class SetDataset:
    def __init__(self, batch_size, transform):
        self.sub_meta = {}
        self.cl_list = range(38) # CropDisease 固定 38 类

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        if not os.path.exists(CropDisease_path):
            raise RuntimeError(f"未找到数据集路径: {CropDisease_path}")

        print(f"--- 正在初始化 CropDisease 索引 (路径: {CropDisease_path}) ---")
        
        # 【核心优化】严禁使用 for i, (data, label) in enumerate(d)，否则会导致内存溢出
        d = ImageFolder(CropDisease_path)
        for img_path, label in d.imgs:
            self.sub_meta[label].append(img_path)

        for key in sorted(self.sub_meta.keys()):
            print(f"类别 {key} 样本数: {len(self.sub_meta[key])}")

        self.sub_dataloader = []
        # Windows 环境下 num_workers 建议设为 0，防止内存占用过高导致 VS Code 卡死
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, 
                                  pin_memory = False)
        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

# =========================================================================
# 子数据集：实际推理时才读取图片
# =========================================================================
class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta # 这里存放的是路径列表
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        # 延迟加载逻辑：从硬盘读取图片并转换
        img = Image.open(self.sub_meta[i]).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

# =========================================================================
# 采样器与变换逻辑 (保持不变)
# =========================================================================
class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class TransformLoader:
    def __init__(self, image_size,
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = ImageJitter( self.jitter_param )
            return method
        name = transform_type
        if name == 'RandomSizedCrop':
            return transforms.RandomResizedCrop(self.image_size)
        if name == 'CenterCrop':
            return transforms.CenterCrop(self.image_size)
        if name == 'Scale':
            return transforms.Resize([int(self.image_size*1.15), int(self.image_size*1.15)])
        if name == 'Normalize':
            return transforms.Normalize(**self.normalize_param)
        if name == 'ToTensor' or name == 'ToTensor()' or name == 'to_tensor':
            return transforms.ToTensor()
        if name == 'RandomHorizontalFlip' or name == 'RandomFlip' or name == 'random_horizontal_flip':
            return transforms.RandomHorizontalFlip()

        if hasattr(transforms, transform_type):
            method = getattr(transforms, transform_type)
            try:
                return method()
            except TypeError:
                return method(self.image_size)
        raise AttributeError(f"Unknown transform: {transform_type}")

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

# =========================================================================
# 数据管理器 (DataManager)
# =========================================================================
class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform)
        # Windows 下 num_workers 设为 0 以防卡死
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 0, pin_memory = False)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide = 100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): 
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )
        # 关键：num_workers=0 和 pin_memory=False 确保系统流畅运行
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 0, pin_memory = False)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader