import os
import sys
import time
import torch

# Keep output quiet by default
os.environ.setdefault('ISIC_VERBOSE', '0')
os.environ.setdefault('ISIC_NUM_WORKERS', '0')

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import get_bscd_loader


def main():
    t0 = time.time()
    loader = get_bscd_loader('ISIC', test_n_way=5, n_shot=5, image_size=224, iter_num=1, seed=123)
    print('loader_len', len(loader), 'init_s', round(time.time() - t0, 2))

    for xs, ys, xq, yq in loader:
        print('xs', tuple(xs.shape), 'xq', tuple(xq.shape))
        print('xs mean/std', float(xs.mean()), float(xs.std()))
        print('xq mean/std', float(xq.mean()), float(xq.std()))
        print('ys uniq', torch.unique(ys).tolist(), 'yq uniq', torch.unique(yq).tolist())
        break


if __name__ == '__main__':
    main()
