#!/usr/bin/env python3
import argparse, csv, os, shutil

def process(csv_path, images_dir, out_base, split, dry_run=False):
    if not os.path.exists(csv_path):
        print(f"[skip] {csv_path} not found")
        return
    moved = 0
    missing = 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get('filename') or row.get('file') or row.get('image')
            label = row.get('label') or row.get('class') or row.get('lbl')
            if not fname or not label:
                print(f"[warn] malformed row: {row}")
                continue
            src = os.path.join(images_dir, fname)
            dest_dir = os.path.join(out_base, split, label)
            dest = os.path.join(dest_dir, fname)
            if not os.path.exists(src):
                print(f"[missing] {src}")
                missing += 1
                continue
            if dry_run:
                print(f"[dry] {src} -> {dest}")
            else:
                os.makedirs(dest_dir, exist_ok=True)
                if not os.path.exists(dest):
                    shutil.move(src, dest)
                    print(f"[mv] {src} -> {dest}")
                    moved += 1
                else:
                    print(f"[exist] {dest}, skipping")
    print(f"[done] split={split} moved={moved} missing={missing}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-base', default='data/mini-imagenet', help='base data dir containing CSVs and images/')
    p.add_argument('--images-dir', default=None, help='images folder (defaults to <data-base>/images)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    base = args.data_base
    images = args.images_dir or os.path.join(base, 'images')
    for split in ('train','val','test'):
        csv_path = os.path.join(base, f'{split}.csv')
        process(csv_path, images, base, split, dry_run=args.dry_run)

if __name__ == '__main__':
    main()