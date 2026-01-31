# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
from typing import Optional

# ================= 1. 路径配置区 (最终修正版) =================
ISIC_task3_train_path = r"C:\Users\李维志\Desktop\pmfbls\bls\pmf_cvpr22\data\ISIC2018\images"
ISIC_csv_path = r"C:\Users\李维志\Desktop\pmfbls\bls\pmf_cvpr22\data\ISIC2018\ISIC2018_Task3_Training_GroundTruth.csv"

# 评测裁剪模式：默认 448crop（更接近你原来 54+ 的那种“放大再裁中心”的行为）
# 可选：448crop / 336crop / 256crop / 224squash
ISIC_EVAL_MODE = os.environ.get("ISIC_EVAL_MODE", "448crop").lower()

WORKERS = int(os.environ.get('ISIC_NUM_WORKERS', '0'))
VERBOSE = True
# ====================================================================

print(f"[DEBUG] 图片路径设为: {ISIC_task3_train_path}")
print(f"[DEBUG] CSV 路径设为: {ISIC_csv_path}")
print(f"[DEBUG] ISIC_EVAL_MODE={ISIC_EVAL_MODE}, WORKERS={WORKERS}")

try:
    import torchvision.transforms as transforms
except Exception:
    import torch as _torch

    class _FallbackTransforms:
        class ToTensor:
            def __call__(self, img):
                arr = np.array(img).astype(np.float32) / 255.0
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                arr = np.transpose(arr, (2, 0, 1))
                return _torch.from_numpy(arr)

        class Compose:
            def __init__(self, funcs):
                self.funcs = funcs

            def __call__(self, img):
                for f in self.funcs:
                    img = f(img)
                return img

        class Resize:
            def __init__(self, size):
                if isinstance(size, (list, tuple)):
                    self.size = (int(size[0]), int(size[1]))
                else:
                    self.size = (int(size), int(size))

            def __call__(self, img):
                return img.resize(self.size)

        class CenterCrop:
            def __init__(self, size):
                self.size = int(size)

            def __call__(self, img):
                w, h = img.size
                th, tw = self.size, self.size
                left = max(0, (w - tw) // 2)
                top = max(0, (h - th) // 2)
                return img.crop((left, top, left + tw, top + th))

        class RandomResizedCrop:
            def __init__(self, size):
                self.size = int(size)

            def __call__(self, img):
                return img.resize((self.size, self.size))

        class RandomHorizontalFlip:
            def __call__(self, img):
                import random
                if random.random() < 0.5:
                    return img.transpose(Image.FLIP_LEFT_RIGHT)
                return img

        class Normalize:
            def __init__(self, mean, std):
                self.mean = _torch.tensor(mean).view(-1, 1, 1)
                self.std = _torch.tensor(std).view(-1, 1, 1)

            def __call__(self, tensor):
                return (tensor - self.mean) / self.std

    transforms = _FallbackTransforms()

from .additional_transforms import ImageJitter
from torch.utils.data import Dataset
from abc import abstractmethod
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CustomDatasetFromImages(Dataset):
    _CACHE = {}

    def __init__(
        self,
        csv_path=ISIC_csv_path,
        image_path=ISIC_task3_train_path,
        min_samples_per_class: Optional[int] = None,
        min_classes: Optional[int] = None,
    ):
        self.img_path = image_path
        self.csv_path = csv_path
        self.to_tensor = transforms.ToTensor()

        if VERBOSE:
            print(f"[ISIC] 开始读取 CSV: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"CSV 读取失败: {e}")

        class_cols = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

        missing_cols = [c for c in class_cols if c not in df.columns]
        if missing_cols:
            print(f"[ISIC] 当前列名: {df.columns.tolist()}")
            raise KeyError(f"CSV 格式不匹配，找不到类别列: {missing_cols}")

        labels_idx = df[class_cols].values.argmax(axis=1)
        self.labels = labels_idx.astype(np.int64)

        if 'image' in df.columns:
            self.image_name = df['image'].astype(str).values
        elif 'image_id' in df.columns:
            self.image_name = df['image_id'].astype(str).values
        else:
            self.image_name = df.iloc[:, 0].astype(str).values

        self.class2idx = {name: i for i, name in enumerate(class_cols)}
        self.data_len = len(self.image_name)

        if VERBOSE:
            print(f"[ISIC] CSV 读取成功, 共 {self.data_len} 张图片")
            print(f"[ISIC] 类别映射: {self.class2idx}")
            unique_labels = np.unique(self.labels)
            print(f"[ISIC] 实际存在的类别索引: {unique_labels}")

    def __getitem__(self, index):
        single_image_name = self.image_name[index]
        if single_image_name.endswith('.jpg'):
            img_name = single_image_name
        else:
            img_name = f"{single_image_name}.jpg"
        img_path = os.path.join(self.img_path, img_name)
        single_image_label = self.labels[index]
        return img_path, single_image_label

    def __len__(self):
        return self.data_len


def identity(x):
    return x


class SimpleDataset:
    def __init__(self, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {'image_names': [], 'image_labels': []}

        d = CustomDatasetFromImages()
        if VERBOSE:
            print(f"[ISIC] 开始构建 SimpleDataset meta, total_entries={len(d)}")

        for data, label in d:
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label)

        if VERBOSE:
            print(f"[ISIC] SimpleDataset meta 构建完成, collected={len(self.meta['image_names'])}")

    def __getitem__(self, i):
        img_path = self.meta['image_names'][i]
        try:
            with Image.open(img_path) as im:
                img = self.transform(im.convert('RGB'))
        except Exception:
            img = self.transform(Image.new('RGB', (224, 224)))
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, batch_size, transform, n_way=5):
        self.sub_meta = {}

        dset = CustomDatasetFromImages(
            min_samples_per_class=int(batch_size),
            min_classes=int(n_way) if n_way is not None else None
        )

        if VERBOSE:
            print(f"[ISIC] 开始扫描所有样本以构建 class->paths 映射, entries={len(dset)}")

        for class_id in range(len(dset.class2idx)):
            self.sub_meta[class_id] = []

        for idx in range(len(dset)):
            img_path, label = dset[idx]
            label = int(label)
            if label not in self.sub_meta:
                self.sub_meta[label] = []
            self.sub_meta[label].append(img_path)

        if VERBOSE:
            print(f"[ISIC] 构建完成: classes_found={len(self.sub_meta)}, total_paths_sampled={sum(len(v) for v in self.sub_meta.values())}")

        min_samples = batch_size
        self.cl_list = [cl for cl, items in self.sub_meta.items() if len(items) >= min_samples]

        if not self.cl_list:
            raise RuntimeError(f"No ISIC classes have >= {min_samples} images; cannot build episodes.")
        if n_way is not None and len(self.cl_list) < n_way:
            raise RuntimeError(
                f"ISIC dataset only has {len(self.cl_list)} classes with >= {min_samples} images, "
                f"but {n_way}-way episodes are requested."
            )

        self.sub_meta = {cl: self.sub_meta[cl] for cl in self.cl_list}
        self.batch_size = int(batch_size)
        self.transform = transform

    def __getitem__(self, i):
        cl = int(self.cl_list[i])
        paths = self.sub_meta[cl]
        n = len(paths)
        if n < self.batch_size:
            raise RuntimeError(f"Class {cl} has only {n} images (< batch_size={self.batch_size}).")

        idx = torch.randperm(n)[: self.batch_size].tolist()

        imgs = []
        for j in idx:
            img_path = paths[j]
            try:
                with Image.open(img_path) as im:
                    img = self.transform(im.convert('RGB'))
            except Exception:
                img = self.transform(Image.new('RGB', (224, 224)))
            imgs.append(img)

        x = torch.stack(imgs, dim=0)
        y = torch.full((self.batch_size,), cl, dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.cl_list)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


class TransformLoader:
    def __init__(
        self,
        image_size,
        normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)
    ):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            return ImageJitter(self.jitter_param)

        method = getattr(transforms, transform_type, None)

        if transform_type == 'RandomSizedCrop':
            return transforms.RandomResizedCrop(self.image_size)
        if transform_type == 'CenterCrop':
            return transforms.CenterCrop(self.image_size)
        if transform_type == 'Scale':
            return transforms.Resize([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        if transform_type == 'Normalize':
            if method is None:
                return transforms.Normalize(**self.normalize_param)
            return method(**self.normalize_param)

        if method is None:
            raise AttributeError(f"Unknown transform: {transform_type}")
        return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
            transform_funcs = [self.parse_transform(x) for x in transform_list]
            return transforms.Compose(transform_funcs)

        # =================== 评测裁剪：这里就是你要改的地方 ===================
        mode = os.environ.get("ISIC_EVAL_MODE", "448crop").lower()

        if mode == "448crop":
            resize_short = int(round(self.image_size * 2.0))     # 224 -> 448（你原来的写法）:contentReference[oaicite:2]{index=2}
            tf = [transforms.Resize(resize_short), transforms.CenterCrop(self.image_size)]
        elif mode == "336crop":
            resize_short = int(round(self.image_size * 1.5))     # 224 -> 336
            tf = [transforms.Resize(resize_short), transforms.CenterCrop(self.image_size)]
        elif mode == "256crop":
            resize_short = int(round(self.image_size * 1.15))    # 224 -> ~256（ImageNet 标配）
            tf = [transforms.Resize(resize_short), transforms.CenterCrop(self.image_size)]
        elif mode == "224squash":
            # 直接拉成 224x224，不做 crop（有时 ISIC 更稳）
            tf = [transforms.Resize([self.image_size, self.image_size])]
        else:
            raise ValueError(f"Unknown ISIC_EVAL_MODE={mode}, expected 448crop/336crop/256crop/224squash")

        tf += [transforms.ToTensor(), self.parse_transform('Normalize')]
        return transforms.Compose(tf)


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
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=WORKERS, pin_memory=True)
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
        if len(dataset.cl_list) < self.n_way:
            raise RuntimeError(
                f"ISIC dataset only has {len(dataset.cl_list)} classes with >= {self.batch_size} images, "
                f"but {self.n_way}-way episodes are requested."
            )

        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        data_loader_params = dict(
            batch_sampler=sampler,
            num_workers=WORKERS,
            pin_memory=True,
            persistent_workers=True if WORKERS > 0 else False,
        )
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader
