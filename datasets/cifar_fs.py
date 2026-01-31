import numpy as np
import torchvision.transforms as transforms

def dataset_setting(nSupport, img_size=32):
    """
    Return dataset setting

    :param int nSupport: number of support examples
    """
    # ImageNet mean/std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    trainTransform = transforms.Compose([
                                         #transforms.RandomCrop(32, padding=4),
                                         transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
                                         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                         transforms.RandomHorizontalFlip(),
                                         #lambda x: np.asarray(x),
                                         transforms.ToTensor(),
                                         normalize
                                        ])

    valTransform = transforms.Compose([#lambda x: np.asarray(x),
                                       transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor(),
                                       normalize])
    inputW, inputH, nbCls = img_size, img_size, 64

    trainDir = './data/cifar-fs/train/'
    valDir = './data/cifar-fs/val/'
    testDir = './data/cifar-fs/test/'
    
    episodeJson = './data/cifar-fs/val1000Episode_5_way_1_shot.json' if nSupport == 1 \
        else './data/cifar-fs/val1000Episode_5_way_5_shot.json'

    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, episodeJson, nbCls
