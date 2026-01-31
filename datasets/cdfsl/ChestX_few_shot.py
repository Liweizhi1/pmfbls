#+#+#+#+#+#+#+#+assistant to=functions.apply_patch  红鼎json  大发快三是国家
"""ChestX few-shot dataset (CDFSL-style).
This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
"""
import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from .additional_transforms import ImageJitter
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
ChestX_path = "./data/ChestX"
class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path=ChestX_path+"/Data_Entry_2017.csv", \
        image_path = ChestX_path+"/images/"):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = image_path
        self.csv_path = csv_path
        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}

        labels_set = []

        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name  = []
        self.labels = []


        for name, label in zip(self.image_name_all,self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)

        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)

    def __getitem__(self, index):
        # Return file path to avoid decoding all images during dataset init.
        single_image_name = self.image_name[index]
        img_path = os.path.join(self.img_path, single_image_name)
        single_image_label = int(self.labels[index])
        return (img_path, single_image_label)

    def __len__(self):
        return self.data_len


def identity(x):
    return x
class SimpleDataset:
    def __init__(self, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []


        d = CustomDatasetFromImages()

        # Store image paths (not decoded images) for fast startup.
        for img_path, label in d:
            self.meta['image_names'].append(img_path)
            self.meta['image_labels'].append(label)

    def __getitem__(self, i):
        img_path = self.meta['image_names'][i]
        try:
            with Image.open(img_path) as im:
                img = im.convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, batch_size, transform):

        self.sub_meta = {}
        self.cl_list = range(7)


        for cl in self.cl_list:
            self.sub_meta[cl] = []

        d = CustomDatasetFromImages()

        for img_path, label in d:
            self.sub_meta[int(label)].append(img_path)

        for key, item in self.sub_meta.items():
            print (len(self.sub_meta[key]))

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)

        for cl in self.cl_list:
            print (cl)
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        img_path = self.sub_meta[i]
        try:
            with Image.open(img_path) as im:
                img = im.convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

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
        if transform_type=='Scale':
            return transforms.Resize([int(self.image_size*1.15), int(self.image_size*1.15)])
        method = getattr(transforms, transform_type)

        if transform_type=='RandomSizedCrop':
            return method(self.image_size)

        elif transform_type=='CenterCrop':
            return method(self.image_size)
        
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        """ChestX few-shot dataset (CDFSL-style).

        This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
        """
        import os

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform)

        # NOTE: On Windows, DataLoader multiprocessing requires picklable datasets.
        # This dataset structure (nested episodic loaders) is not multiprocessing-safe,
        # so we force single-process loading.
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 0, pin_memory = True)
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

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )
        # Force single-process loading for Windows compatibility.
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 0, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':

    base_datamgr            = SetDataManager(224, n_query = 16, n_support = 5)
    base_loader             = base_datamgr.get_data_loader(aug = True)

