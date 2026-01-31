import os
import random
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader

from .samplers import RASampler
from .episodic_dataset import EpisodeDataset, EpisodeJSONDataset
from .meta_val_dataset import MetaValDataset
from .meta_h5_dataset import FullMetaDatasetH5
from .meta_dataset.utils import Split


def get_sets(args):
    if args.dataset == 'cifar_fs':
        from .cifar_fs import dataset_setting
    elif args.dataset == 'cifar_fs_elite': # + elite data augmentation
        from .cifar_fs_elite import dataset_setting
    elif args.dataset == 'mini_imagenet':
        from .mini_imagenet import dataset_setting
    elif args.dataset == 'meta_dataset':
        if args.eval:
            trainSet = valSet = None
            testSet = FullMetaDatasetH5(args, Split.TEST)
        else:
            trainSet = FullMetaDatasetH5(args, Split.TRAIN)
            valSet = {}
            for source in args.val_sources:
                valSet[source] = MetaValDataset(os.path.join(args.data_path, source,
                                                             f'val_ep{args.nValEpisode}_img{args.image_size}.h5'),
                                                num_episodes=args.nValEpisode)
            testSet = None
        return trainSet, valSet, testSet
    else:
        raise ValueError(f'{dataset} is not supported.')

    # If not meta_dataset
    trainTransform, valTransform, inputW, inputH, \
    trainDir, valDir, testDir, episodeJson, nbCls = \
            dataset_setting(args.nSupport, args.img_size)

    trainSet = EpisodeDataset(imgDir = trainDir,
                              nCls = args.nClsEpisode,
                              nSupport = args.nSupport,
                              nQuery = args.nQuery,
                              transform = trainTransform,
                              inputW = inputW,
                              inputH = inputH,
                              nEpisode = args.nEpisode)

    valSet = EpisodeJSONDataset(episodeJson,
                                valDir,
                                inputW,
                                inputH,
                                valTransform)

    testSet = EpisodeDataset(imgDir = testDir,
                             nCls = args.nClsEpisode,
                             nSupport = args.nSupport,
                             nQuery = args.nQuery,
                             transform = valTransform,
                             inputW = inputW,
                             inputH = inputH,
                             nEpisode = args.nEpisode)

    return trainSet, valSet, testSet


def get_loaders(args, num_tasks, global_rank):
    # datasets
    if args.eval:
        _, _, dataset_vals = get_sets(args)
    else:
        dataset_train, dataset_vals, _ = get_sets(args)

    # Worker init function
    if 'meta_dataset' in args.dataset: # meta_dataset & meta_dataset_h5
        #worker_init_fn = partial(worker_init_fn_, seed=args.seed)
        #worker_init_fn = lambda _: np.random.seed()
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
    else:
        worker_init_fn = None

    # Val loader
    # NOTE: meta-dataset has separate val-set per domain
    if not isinstance(dataset_vals, dict):
        dataset_vals = {'single': dataset_vals}

    data_loader_val = {}

    for j, (source, dataset_val) in enumerate(dataset_vals.items()):
        if args.distributed:
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                          'This will slightly alter validation results as extra duplicate entries are added to achieve '
                          'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        generator = torch.Generator()
        generator.manual_seed(args.seed + 10000 + j)

        # NOTE:
        #  - 原代码在验证/评估阶段固定 batch_size=1、num_workers=3，
        #    每次只跑 1 个 episode，GPU 利用率较低；
        #  - 这里改为使用 args.batch_size 和 args.num_workers，
        #    允许一次性处理多个 episode，从而显著提升 Step 3 吞吐。
        data_loader = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=generator
        )
        data_loader_val[source] = data_loader

    if 'single' in dataset_vals:
        data_loader_val = data_loader_val['single']

    if args.eval:
        return None, data_loader_val

    # Train loader
    if args.distributed:
        if args.repeated_aug: # (by default OFF)
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    return data_loader_train, data_loader_val


def get_bscd_loader(dataset="EuroSAT", test_n_way=5, n_shot=5, image_size=224, iter_num=600, seed: int = 0, fixed_episode_file: str = ''):
    n_query = 15
    few_shot_params = dict(n_way=test_n_way , n_support=n_shot)

    if dataset == "EuroSAT":
        from .cdfsl.EuroSAT_few_shot import SetDataManager
    elif dataset == "ISIC":
        from .cdfsl.ISIC_few_shot import SetDataManager
    elif dataset == "CropDisease":
        from .cdfsl.CropDisease_few_shot import SetDataManager
    elif dataset == "ChestX":
        from .cdfsl.ChestX_few_shot import SetDataManager
    else:
        raise ValueError(f'Datast {dataset} is not supported.')

    # If a fixed episode file is provided, load episodes from it.
    # Support two formats:
    #  - 'x': preloaded numpy arrays shaped (n_episodes, n_total, C, H, W)
    #  - 'paths': path lists shaped (n_episodes, n_total) -> load images on-the-fly
    if fixed_episode_file and os.path.exists(fixed_episode_file):
        arr = np.load(fixed_episode_file, allow_pickle=True)

        if 'x' in arr:
            x_all = arr['x']

            def _fixed_loader():
                for xi in x_all:
                    # add batch dim (1, n_total, C, H, W)
                    yield (torch.from_numpy(xi).unsqueeze(0), None)

            novel_loader = _fixed_loader()

            n_eps = x_all.shape[0]

        elif 'paths' in arr:
            from PIL import Image

            paths_all = arr['paths']

            def load_img_to_array(p, image_size=224):
                with Image.open(p) as im:
                    im = im.convert('RGB').resize((image_size, image_size))
                    a = np.array(im).astype(np.float32) / 255.0
                    a = a.transpose(2, 0, 1)
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
                    a = (a - mean) / std
                    return a

            def _paths_loader():
                for epi_paths in paths_all:
                    imgs = []
                    for p in epi_paths:
                        imgs.append(load_img_to_array(p, image_size=image_size))
                    xi = np.stack(imgs, axis=0).astype(np.float32)
                    yield (torch.from_numpy(xi).unsqueeze(0), None)

            novel_loader = _paths_loader()
            n_eps = len(paths_all)

        else:
            # fallback to default generation
            datamgr = SetDataManager(image_size, n_eposide=iter_num, n_query=n_query, **few_shot_params)
            novel_loader = datamgr.get_data_loader(aug =False)
            n_eps = None

        # make novel_loader have a length when using fixed arrays/paths
        class _LenWrapper:
            def __init__(self, iterable, n):
                self.iterable = iterable
                self._n = n
            def __iter__(self):
                return self.iterable
            def __len__(self):
                return self._n

        if n_eps is not None:
            novel_loader = _LenWrapper(novel_loader, n_eps)
    else:
        datamgr = SetDataManager(image_size, n_eposide=iter_num, n_query=n_query, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug =False)

    def _loader_wrap():
        for x, y in novel_loader:
            # Support two data formats from novel_loader:
            # 1) fixed tensor: x shaped (b, n_total, C, H, W) ordered by class
            # 2) episodic SetDataManager batch: x is a sequence/tuple of per-class
            #    batches, each element is (imgs, labels) with shape (per_class, C, H, W).
            per_class = n_shot + n_query
            if isinstance(x, (list, tuple)):
                # assemble per-class tensors into shape (b=1, test_n_way, per_class, C, H, W)
                imgs_list = []
                for item in x:
                    # item can be (imgs, labels) or imgs tensor
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        imgs = item[0]
                    else:
                        imgs = item
                    # convert numpy arrays to torch tensors when needed
                    if not torch.is_tensor(imgs):
                        imgs = torch.as_tensor(imgs)
                    imgs_list.append(imgs)
                # ensure all per-class tensors have the same shape
                shapes = [tuple(t.shape) for t in imgs_list]
                if len(set(shapes)) != 1:
                    raise RuntimeError(f'Inconsistent per-class shapes in loader: {shapes}')
                xi = torch.stack(imgs_list, dim=0)  # (test_n_way, per_class, C, H, W)
                xi = xi.unsqueeze(0)  # (1, test_n_way, per_class, C, H, W)
                b = xi.size(0)
                x_resh = xi
                SupportTensor = x_resh[:, :, :n_shot].contiguous().view(b, test_n_way * n_shot, *xi.size()[3:])
                QryTensor = x_resh[:, :, n_shot:].contiguous().view(b, test_n_way * n_query, *xi.size()[3:])
            else:
                # x can have several tensor layouts depending on DataLoader collate:
                # - (b, n_total, C, H, W) where n_total = n_way * per_class
                # - (n_way, per_class, C, H, W) produced by collating per-class batches
                # - (b, n_way, per_class, C, H, W)
                if not torch.is_tensor(x):
                    raise RuntimeError('Unsupported loader output type')
                dims = x.dim()
                if dims == 5:
                    # shape possibilities: (b, n_total, C,H,W) OR (n_way, per_class, C,H,W)
                    a0, a1 = x.size(0), x.size(1)
                    if a1 == per_class and a0 == test_n_way:
                        # (n_way, per_class, C,H,W) -> single episode
                        xi = x.unsqueeze(0)  # (1, n_way, per_class, C,H,W)
                        b = 1
                        x_resh = xi
                        SupportTensor = x_resh[:, :, :n_shot].contiguous().view(b, test_n_way * n_shot, *xi.size()[3:])
                        QryTensor = x_resh[:, :, n_shot:].contiguous().view(b, test_n_way * n_query, *xi.size()[3:])
                    elif a1 == per_class and a0 % test_n_way == 0:
                        # (b*n_way, per_class, ...) where a0 = b * n_way
                        b = a0 // test_n_way
                        xi = x.view(b, test_n_way, per_class, *x.size()[2:])
                        x_resh = xi
                        SupportTensor = x_resh[:, :, :n_shot].contiguous().view(b, test_n_way * n_shot, *x.size()[2:])
                        QryTensor = x_resh[:, :, n_shot:].contiguous().view(b, test_n_way * n_query, *x.size()[2:])
                    else:
                        # assume (b, n_total, C,H,W)
                        b = x.size(0)
                        n_total = x.size(1)
                        if n_total == test_n_way * per_class:
                            x_resh = x.view(b, test_n_way, per_class, *x.size()[2:])
                            SupportTensor = x_resh[:, :, :n_shot].contiguous().view(b, test_n_way * n_shot, *x.size()[2:])
                            QryTensor = x_resh[:, :, n_shot:].contiguous().view(b, test_n_way * n_query, *x.size()[2:])
                        else:
                            # fallback: try to split first test_n_way*n_shot for support and next for query
                            SupportTensor = x[:, : (test_n_way * n_shot)].contiguous().view(b, test_n_way * n_shot, *x.size()[2:])
                            QryTensor = x[:, (test_n_way * n_shot): (test_n_way * (n_shot + n_query))].contiguous().view(b, test_n_way * n_query, *x.size()[2:])
                elif dims == 6:
                    # (b, n_way, per_class, C,H,W)
                    b = x.size(0)
                    x_resh = x
                    SupportTensor = x_resh[:, :, :n_shot].contiguous().view(b, test_n_way * n_shot, *x.size()[3:])
                    QryTensor = x_resh[:, :, n_shot:].contiguous().view(b, test_n_way * n_query, *x.size()[3:])
                else:
                    raise RuntimeError(f'Unsupported tensor shape from novel_loader: dims={dims}, shape={tuple(x.size())}')

            SupportLabel = torch.from_numpy(np.repeat(range(test_n_way), n_shot)).view(1, test_n_way * n_shot)
            QryLabel = torch.from_numpy(np.repeat(range(test_n_way), n_query)).view(1, test_n_way * n_query)

            yield SupportTensor, SupportLabel, QryTensor, QryLabel

    class _Loader(object):
        def __init__(self):
            self.iterable = _loader_wrap()
            # NOTE: the following are required by engine.py:_evaluate()
            self.dataset = self
            self.generator = torch.Generator().manual_seed(int(seed))

        def __len__(self):
            return len(novel_loader)
        def __iter__(self):
            return self.iterable

    return _Loader()
