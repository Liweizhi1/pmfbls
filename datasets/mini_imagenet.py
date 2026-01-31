import os
import numpy as np
import torchvision.transforms as transforms


def dataset_setting(nSupport, img_size=80, data_path=None):
    """
    Return dataset setting

    :param int nSupport: number of support examples
    """
    mean = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
    std = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
    normalize = transforms.Normalize(mean=mean, std=std)
    trainTransform = transforms.Compose([#transforms.RandomCrop(img_size, padding=8),
                                         transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
                                         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                         transforms.RandomHorizontalFlip(),
                                         #lambda x: np.asarray(x),
                                         transforms.ToTensor(),
                                         normalize
                                        ])

    valTransform = transforms.Compose([#transforms.CenterCrop(80),
                                       transforms.Resize((img_size, img_size)),
                                       #lambda x: np.asarray(x),
                                       transforms.ToTensor(),
                                       normalize])

    inputW, inputH, nbCls = img_size, img_size, 64

    # Determine base directory. Prefer explicit `data_path` from args when provided.
    if data_path:
        base = data_path
    else:
        # Support two common folder namings: 'Mini-ImageNet' and 'mini-imagenet'
        base1 = './data/Mini-ImageNet'
        base2 = './data/mini-imagenet'
        if os.path.isdir(os.path.join(base1, 'train')):
            base = base1
        elif os.path.isdir(os.path.join(base2, 'train')):
            base = base2
        else:
            # fallback to default base1 so original behavior unchanged if neither exists
            base = base1

    trainDir = os.path.join(base, 'train') + '/'
    valDir = os.path.join(base, 'val') + '/'
    testDir = os.path.join(base, 'test') + '/'
    episodeJson = os.path.join(base, 'val1000Episode_5_way_1_shot.json') if nSupport == 1 \
            else os.path.join(base, 'val1000Episode_5_way_5_shot.json')

    # If expected trainDir doesn't exist, try to auto-detect a plausible train directory
    if not os.path.isdir(trainDir):
        # look for a directory under base that contains many class subfolders
        candidate = None
        for root, dirs, files in os.walk(base):
            # skip hidden/system folders
            bname = os.path.basename(root)
            if bname.startswith('.'):
                continue
            # many subdirectories -> likely class folders
            if len(dirs) >= 10:
                candidate = root
                break
            # or many image files -> could be flattened train folder
            img_count = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            if img_count >= 500:
                candidate = root
                break

        if candidate:
            # normalize to trailing slash
            trainDir = os.path.join(candidate, '')
            # attempt to set val/test as sibling 'val'/'test' if they exist
            cand_parent = os.path.dirname(candidate)
            if os.path.isdir(os.path.join(cand_parent, 'val')):
                valDir = os.path.join(cand_parent, 'val') + '/'
            if os.path.isdir(os.path.join(cand_parent, 'test')):
                testDir = os.path.join(cand_parent, 'test') + '/'
            print(f"[mini_imagenet] Auto-detected trainDir={trainDir}")
        else:
            print(f"[mini_imagenet] Warning: expected trainDir {trainDir} not found and auto-detection failed.")

    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, episodeJson, nbCls
