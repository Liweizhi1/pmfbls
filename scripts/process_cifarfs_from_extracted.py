#!/usr/bin/env python3
import os
import shutil
import sys

# 默认路径（相对于 scripts/）
DEFAULT_BASE = os.path.join(os.path.dirname(__file__), '..', 'cifar-fs-data', 'cifar100')
DEFAULT_BASE = os.path.abspath(DEFAULT_BASE)

if len(sys.argv) > 1:
    base_dir = os.path.abspath(sys.argv[1])
else:
    base_dir = DEFAULT_BASE

splits_dir = os.path.join(base_dir, 'splits', 'bertinetto')
source_imgs_dir = os.path.join(base_dir, 'data')
processed_root = os.path.join(os.path.dirname(base_dir), 'cifar_fs_processed')

print('Base dir:', base_dir)
print('Splits dir:', splits_dir)
print('Source images dir:', source_imgs_dir)
print('Processed root:', processed_root)

if not os.path.isdir(splits_dir):
    raise SystemExit('Splits directory not found: ' + splits_dir)
if not os.path.isdir(source_imgs_dir):
    raise SystemExit('Source images directory not found: ' + source_imgs_dir)

splits = {
    'train': 'train.txt',
    'val': 'val.txt',
    'test': 'test.txt'
}

for split_name, split_file in splits.items():
    split_path = os.path.join(splits_dir, split_file)
    if not os.path.exists(split_path):
        print('Warning: split file not found:', split_path)
        continue

    dest_dir = os.path.join(processed_root, split_name)
    os.makedirs(dest_dir, exist_ok=True)

    with open(split_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # If first line is a header like 'train' or 'val', remove it
    if lines and lines[0].lower() == split_name:
        lines = lines[1:]

    print(f'Processing split {split_name}: {len(lines)} classes')

    for class_name in lines:
        src = os.path.join(source_imgs_dir, class_name)
        dst = os.path.join(dest_dir, class_name)
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            print('Warning: source class folder not found:', src)

print('\nDone. Processed dataset available at:')
print(processed_root)
