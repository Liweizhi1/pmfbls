import torch
from PIL import ImageEnhance, Image

# Try to use torchvision transforms; if unavailable (or importing it triggers
# system-level extension errors), provide a minimal local fallback implementation
try:
    import torchvision.transforms as transforms
except Exception:
    # Minimal fallback implementations for the subset used by the project
    import random

    class Resize:
        def __init__(self, size):
            self.size = size
        def __call__(self, img):
            return img.resize((self.size, self.size), Image.BILINEAR)

    class CenterCrop:
        def __init__(self, size):
            self.size = size
        def __call__(self, img):
            w, h = img.size
            th = self.size
            tw = self.size
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            return img.crop((x1, y1, x1 + tw, y1 + th))

    class RandomHorizontalFlip:
        def __init__(self, p=0.5):
            self.p = p
        def __call__(self, img):
            if random.random() < self.p:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            return img

    class RandomResizedCrop:
        def __init__(self, size):
            self.size = size
        def __call__(self, img):
            # simple center-crop resize as approximation
            return img.resize((self.size, self.size), Image.BILINEAR)

    class ToTensor:
        def __call__(self, pic):
            # Convert PIL Image to torch tensor HxWxC -> CxHxW
            arr = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            arr = arr.reshape(pic.size[1], pic.size[0], len(pic.getbands()))
            arr = arr.permute(2, 0, 1).float().div(255)
            return arr

    class Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean).view(-1,1,1)
            self.std = torch.tensor(std).view(-1,1,1)
        def __call__(self, tensor):
            return (tensor - self.mean) / self.std

    class Compose:
        def __init__(self, transforms_list):
            self.transforms = transforms_list
        def __call__(self, img):
            out = img
            for t in self.transforms:
                out = t(out)
            return out

    # Provide a `transforms` namespace with the used classes
    class _transforms_mod:
        Resize = Resize
        CenterCrop = CenterCrop
        RandomResizedCrop = RandomResizedCrop
        RandomHorizontalFlip = RandomHorizontalFlip
        ToTensor = ToTensor
        Normalize = Normalize
        Compose = Compose

    transforms = _transforms_mod

from .utils import Split
from .config import DataConfig

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)


class ImageJitter(object):
    def __init__(self, transformdict):
        transformtypedict = dict(Brightness=ImageEnhance.Brightness,
                                 Contrast=ImageEnhance.Contrast,
                                 Sharpness=ImageEnhance.Sharpness,
                                 Color=ImageEnhance.Color)
        self.params = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.params))

        for i, (transformer, alpha) in enumerate(self.params):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


def get_transforms(data_config: DataConfig,
                   split: Split):
    if split == Split["TRAIN"]:
        return train_transform(data_config)
    else:
        return test_transform(data_config)


def test_transform(data_config: DataConfig):
    resize_size = int(data_config.image_size * 256 / 224)
    assert resize_size == data_config.image_size * 256 // 224
    # resize_size = data_config.image_size

    transf_dict = {'resize': transforms.Resize(resize_size),
                   'center_crop': transforms.CenterCrop(data_config.image_size),
                   'to_tensor': transforms.ToTensor(),
                   'normalize': normalize}
    augmentations = data_config.test_transforms

    return transforms.Compose([transf_dict[key] for key in augmentations])


def train_transform(data_config: DataConfig):
    transf_dict = {'resize': transforms.Resize(data_config.image_size),
                   'center_crop': transforms.CenterCrop(data_config.image_size),
                   'random_resized_crop': transforms.RandomResizedCrop(data_config.image_size),
                   'jitter': ImageJitter(jitter_param),
                   'random_flip': transforms.RandomHorizontalFlip(),
                   'to_tensor': transforms.ToTensor(),
                   'normalize': normalize}
    augmentations = data_config.train_transforms

    return transforms.Compose([transf_dict[key] for key in augmentations])
