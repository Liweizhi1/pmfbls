import time
import argparse
import os
import statistics
import sys
from pathlib import Path
import pandas as pd
import tempfile

import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so imports work no matter cwd
# Script is located at bls/pmf_cvpr22/scripts/diag_isic_speed.py; go up three parents to reach workspace root
proj_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(proj_root))

# Import dataset components from project
from bls.pmf_cvpr22.datasets.cdfsl.ISIC_few_shot import (
    SimpleDataset,
    SetDataset,
    CustomDatasetFromImages,
    TransformLoader,
    ISIC_task3_train_path,
)


class SimpleFromMeta(torch.utils.data.Dataset):
    """Top-level dataset class so it can be pickled by multiprocessing on Windows."""
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        p = self.image_paths[idx]
        with Image.open(p) as im:
            img = self.transform(im.convert('RGB'))
        return img, int(self.labels[idx])


def time_dataloader(dataset, batch_size=16, num_workers=4, pin_memory=True, persistent_workers=False, n_batches=50):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=pin_memory, persistent_workers=persistent_workers)
    it = iter(dl)
    times = []
    # warm-up 2
    for _ in range(2):
        try:
            next(it)
        except StopIteration:
            break
    for i in range(n_batches):
        t0 = time.time()
        try:
            batch = next(it)
        except StopIteration:
            break
        t1 = time.time()
        times.append(t1 - t0)
        if i % 10 == 0:
            print(f"  batch {i} load time: {times[-1]:.4f}s")
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default=None, help='Override image directory')
    parser.add_argument('--csv', type=str, default=None, help='Override csv path')
    parser.add_argument('--label-col', type=str, default=None, help='Column name for labels in CSV (overrides default ISIC_LABEL_COL)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n-batches', type=int, default=30)
    parser.add_argument('--workers-list', type=str, default='0,2,4,8', help='comma separated worker counts to test')
    parser.add_argument('--test-type', type=str, choices=['simple','set'], default='simple', help='Which dataset wrapper to test')
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()

    img_dir = args.image_dir if args.image_dir is not None else ISIC_task3_train_path + os.sep
    csv = args.csv if args.csv is not None else os.path.join(os.path.dirname(__file__), '..', '..', 'ISIC', 'metadata.csv')

    # Determine label column: priority --label-col, env, then auto-detect with heuristics
    # If user provided a concrete value (not 'auto'), use it
    if args.label_col is not None and args.label_col.lower() != 'auto':
        chosen_label = args.label_col
    else:
        chosen_label = os.environ.get('ISIC_LABEL_COL', None)
        # Read csv header/full to inspect columns
        try:
            df_full = pd.read_csv(csv)
            cols = list(df_full.columns)
        except Exception:
            cols = []

        print('CSV columns:', cols)

        # Prefer known label column names
        known = ['diagnosis_3', 'diagnosis', 'dx', 'label', 'target', 'diagnosis_1', 'diagnosis_2']
        chosen_label = None if chosen_label is None else chosen_label
        for k in known:
            if k in cols:
                chosen_label = k
                break

        # If still not found, pick a column with reasonable cardinality (likely labels)
        if chosen_label is None and cols:
            n_rows = len(df_full)
            best_col = None
            best_uniques = None
            for c in cols:
                cl = c.lower()
                # skip obvious id/file columns
                if cl in ('isic_id', 'image_id', 'imageid', 'image_name', 'image', 'filename', 'file', 'id', 'path'):
                    continue
                try:
                    nunique = df_full[c].nunique(dropna=True)
                except Exception:
                    continue
                # prefer small-to-medium unique counts (2..min(500,n_rows//2))
                if nunique >= 2 and nunique <= max(500, max(2, n_rows // 2)):
                    if best_uniques is None or nunique < best_uniques:
                        best_uniques = nunique
                        best_col = c
            chosen_label = best_col

    if chosen_label is None:
        raise RuntimeError(f"Unable to determine label column from CSV; pass --label-col. Available columns: {cols}")
    os.environ['ISIC_LABEL_COL'] = chosen_label
    print(f'Using label column: {chosen_label}')

    print(f'Using image_dir={img_dir}')
    print(f'Using csv={csv}')

    transform = TransformLoader(args.image_size).get_composed_transform(aug=False)

    if args.test_type == 'simple':
        print('Constructing SimpleDataset (loads metadata and builds list, no image decoding yet)')
        # Use provided csv/image-dir to build an equivalent simple dataset
        # If CSV lacks 'isic_id' column, try to auto-detect common id column names
        try:
            df_full = pd.read_csv(csv)
        except Exception as e:
            raise RuntimeError(f'Failed to read csv {csv}: {e}')

        csv_to_use = csv
        if 'isic_id' not in df_full.columns:
            id_candidates = ['isic_id', 'image_id', 'imageid', 'image_name', 'image', 'filename', 'file', 'id']
            found = None
            for c in id_candidates:
                if c in df_full.columns:
                    found = c
                    break
            if found is not None:
                df_full = df_full.rename(columns={found: 'isic_id'})
                tmpf = tempfile.NamedTemporaryFile(prefix='diag_isic_', suffix='.csv', delete=False)
                df_full.to_csv(tmpf.name, index=False)
                csv_to_use = tmpf.name
                print(f"Renamed CSV column '{found}' -> 'isic_id' and wrote temp csv: {csv_to_use}")
            else:
                raise RuntimeError('CSV does not contain an image-id column (tried common names).')

        dset_meta = CustomDatasetFromImages(csv_path=csv_to_use, image_path=img_dir)
        image_names = [os.path.join(img_dir, f"{nid}.jpg") for nid in dset_meta.image_name]

        # Use top-level SimpleFromMeta (picklable on Windows)
        ds = SimpleFromMeta(image_names, dset_meta.labels, transform)
        dataset_desc = f'SimpleFromMeta len={len(ds)}'
    else:
        print('Constructing SetDataset (will sample per-class episodes)')
        ds = SetDataset(batch_size=args.batch_size, transform=transform)
        dataset_desc = f'SetDataset classes={len(ds.cl_list)}'

    print(dataset_desc)

    workers_list = [int(x) for x in args.workers_list.split(',') if x.strip()]

    results = []
    for w in workers_list:
        print(f'\n--- Testing num_workers={w} ---')
        pin = True
        persistent = True if w > 0 else False
        times = time_dataloader(ds, batch_size=args.batch_size, num_workers=w, pin_memory=pin,
                                persistent_workers=persistent, n_batches=args.n_batches)
        if len(times) == 0:
            print('No batches produced (dataset too small?)')
            continue
        mean = statistics.mean(times)
        median = statistics.median(times)
        p95 = sorted(times)[min(len(times)-1, int(len(times)*0.95))]
        print(f'num_workers={w} batches={len(times)} mean={mean:.4f}s median={median:.4f}s p95={p95:.4f}s')
        results.append((w, mean, median, p95))

    print('\nSummary:')
    for w, mean, median, p95 in results:
        print(f'workers={w}: mean={mean:.4f}s median={median:.4f}s p95={p95:.4f}s')

    print('\nRecommendation: If mean load time per batch > 0.2s and GPU utilization low, you have IO bottleneck.')


if __name__ == '__main__':
    main()
