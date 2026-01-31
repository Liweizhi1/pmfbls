# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from PIL import ImageFile

# 解决图片截断报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================================================================
# 路径配置
# =========================================================================
EuroSAT_path = r"C:\Users\李维志\Desktop\pmfbls\bls\pmf_cvpr22\data\EuroSAT\2750"

def identity(x):
    return x

# =========================================================================
# 补全缺失的 SubDataset 类 (这是给 SetDataset 用的)
# =========================================================================
class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl # 当前类别索引
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = self.sub_meta[i]
        try:
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            # 容错处理，万一图坏了，返回一张全黑图防止崩盘
            print(f"Error loading {image_path}: {e}")
            img = self.transform(Image.new('RGB', (224, 224)))
            
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

# =========================================================================
# 普通数据集 (用于验证/测试整个集)
# =========================================================================
class SimpleDataset:
    def __init__(self, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform
        self.meta = {}
        self.meta['image_names'] = []
        self.meta['image_labels'] = []
        
        if not os.path.exists(EuroSAT_path):
            raise RuntimeError(f"Dataset path not found: {EuroSAT_path}")

        classes = sorted([d for d in os.listdir(EuroSAT_path) if os.path.isdir(os.path.join(EuroSAT_path, d))])
        
        for label, class_name in enumerate(classes):
            class_dir = os.path.join(EuroSAT_path, class_name)
            for f in os.listdir(class_dir):
                if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    self.meta['image_names'].append(os.path.join(class_dir, f))
                    self.meta['image_labels'].append(label)

    def __getitem__(self, i):
        img = self.transform(Image.open(self.meta['image_names'][i]).convert('RGB'))
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

# =========================================================================
# 修复后的 SetDataset (用于 Few-Shot 采样)
# =========================================================================
class SetDataset:
    def __init__(self, batch_size, transform, n_way=5):
        self.sub_meta = {}
        # EuroSAT 固定 10 类
        self.cl_list = range(10) 
        self.n_way = n_way
        
        if not os.path.exists(EuroSAT_path):
             raise RuntimeError(f"Dataset path not found: {EuroSAT_path}")

        classes = sorted([d for d in os.listdir(EuroSAT_path) if os.path.isdir(os.path.join(EuroSAT_path, d))])
        
        # 1. 扫描所有图片
        for target, class_name in enumerate(classes):
            self.sub_meta[target] = []
            class_dir = os.path.join(EuroSAT_path, class_name)
            for f in os.listdir(class_dir):
                if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    self.sub_meta[target].append(os.path.join(class_dir, f))

        # 2. 【关键修复】创建子数据加载器列表
        self.sub_dataloader = [] 
        self.batch_size = batch_size
        
        for cl in self.cl_list:
            # 为每个类别创建一个 SubDataset
            sub_dset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            # 包装成 DataLoader
            # num_workers=0 是为了防止 Windows 下多进程死锁
            sub_loader = DataLoader(sub_dset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
            self.sub_dataloader.append(sub_loader)

    def __getitem__(self, i):
        # i 是类别索引 (0-9)
        # 这里的 next(iter(...)) 会从对应类别的 DataLoader 里取出一个 batch (Support + Query)
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

# =========================================================================
# 采样器与变换
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
                 normalize_param = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def get_composed_transform(self, aug = False):
        if aug:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=self.jitter_param['Brightness'], 
                    contrast=self.jitter_param['Contrast'], 
                    saturation=self.jitter_param['Color']),
                transforms.ToTensor(),
                transforms.Normalize(**self.normalize_param)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize([int(self.image_size*1.15), int(self.image_size*1.15)]),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(**self.normalize_param)
            ])
        return transform

# =========================================================================
# DataManager
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
        # Windows下建议 num_workers=0
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 0, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide=100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform, n_way=self.n_way)
        sampler = EpisodicBatchSampler(len(dataset.cl_list), self.n_way, self.n_eposide)
        
        # Windows下建议 num_workers=0
        data_loader_params = dict(batch_sampler=sampler, num_workers=0, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':
    # 简单测试逻辑
    if os.path.exists(EuroSAT_path):
        print("Path exists, testing loader construction...")
        try:
            mgr = SetDataManager(224)
            loader = mgr.get_data_loader(aug=False)
            print("Loader built. Fetching one batch...")
            for x, y in loader:
                print(f"Success! x: {x.shape}, y: {y.shape}")
                break
        except Exception as e:
            print(f"Error: {e}")