"""
Preprocess ISIC2018 images: resize + center crop and save to an output folder.

Usage (PowerShell):
 $env:PYTHONIOENCODING = 'utf-8'
 python preprocess_isic.py --csv ../data/ISIC2018/ISIC2018_Task3_Training_GroundTruth.csv --input-dir ../data/ISIC2018/images --out-dir ../data/ISIC2018/preprocessed_224 --size 224 --workers 4

The script preserves original filenames and writes JPEGs to `out_dir`.
"""

import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError('Pillow is required. Install with: pip install pillow')

try:
    import pandas as pd
except Exception:
    pd = None


def process_one(input_root: str, output_root: str, img_name: str, size: int, resize_short: Optional[int]):
    # ensure .jpg suffix
    if not img_name.lower().endswith('.jpg'):
        img_name_out = f"{img_name}.jpg"
        img_name_in = f"{img_name}.jpg"
    else:
        img_name_out = img_name
        img_name_in = img_name

    in_path = os.path.join(input_root, img_name_in)
    out_path = os.path.join(output_root, img_name_out)

    try:
        with Image.open(in_path) as im:
            im = im.convert('RGB')

            # If resize_short provided, resize preserving aspect ratio by short-edge
            if resize_short is not None:
                w, h = im.size
                short = min(w, h)
                if short != resize_short:
                    scale = resize_short / float(short)
                    new_w = int(round(w * scale))
                    new_h = int(round(h * scale))
                    im = im.resize((new_w, new_h), resample=Image.LANCZOS)

            # center crop to (size, size)
            w, h = im.size
            left = max(0, (w - size) // 2)
            top = max(0, (h - size) // 2)
            im = im.crop((left, top, left + size, top + size))

            # ensure output dir exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            im.save(out_path, format='JPEG', quality=90)
            return True, img_name_out, ''
    except Exception as e:
        return False, img_name_out, str(e)


def load_image_list_from_csv(csv_path: str):
    if pd is None:
        raise RuntimeError('pandas is required to read CSV files. Install with: pip install pandas')
    df = pd.read_csv(csv_path)
    if 'image' in df.columns:
        return df['image'].astype(str).tolist()
    if 'image_id' in df.columns:
        return df['image_id'].astype(str).tolist()
    # fallback: first column
    return df.iloc[:, 0].astype(str).tolist()


def gather_all_images(input_dir: str):
    p = Path(input_dir)
    imgs = [f.name for f in p.rglob('*.jpg')]
    imgs += [f.name[:-4] for f in p.rglob('*.jpg')]  # also name without suffix
    return list(set(imgs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None, help='ISIC CSV file with image ids')
    parser.add_argument('--input-dir', type=str, required=True, help='原始图片目录')
    parser.add_argument('--out-dir', type=str, required=True, help='预处理后输出目录')
    parser.add_argument('--size', type=int, default=224, help='最终 crop 大小')
    parser.add_argument('--resize-short', type=int, default=None, help='短边 resize 大小 (默认 None -> 直接 resize 到最终大小)')
    parser.add_argument('--workers', type=int, default=4, help='并行进程数')
    parser.add_argument('--force', action='store_true', help='强制重新生成已存在文件')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.out_dir)
    size = int(args.size)

    if args.csv:
        image_list = load_image_list_from_csv(args.csv)
    else:
        image_list = gather_all_images(input_dir)

    # normalize names (strip suffix)
    normalized = []
    for name in image_list:
        if name.lower().endswith('.jpg'):
            normalized.append(name[:-4])
        else:
            normalized.append(name)

    # prepare tasks
    tasks = []
    for nm in normalized:
        out_path = os.path.join(out_dir, f"{nm}.jpg")
        if os.path.exists(out_path) and not args.force:
            continue
        input_path = os.path.join(input_dir, f"{nm}.jpg")
        if not os.path.exists(input_path):
            # skip missing
            continue
        tasks.append(nm)

    if not tasks:
        print('No tasks to process (all exist or no input files).')
        return

    print(f'Processing {len(tasks)} images -> {out_dir} using {args.workers} workers')

    process_fn = partial(process_one, input_dir, out_dir, size=size, resize_short=args.resize_short)

    failures = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(process_fn, nm): nm for nm in tasks}
        for fut in as_completed(futures):
            ok, name, err = fut.result()
            if not ok:
                failures.append((name, err))

    if failures:
        print(f'Finished with {len(failures)} failures:')
        for n, e in failures[:10]:
            print('-', n, e)
    else:
        print('All done.')


if __name__ == '__main__':
    main()
