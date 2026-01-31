import os
import shutil
import zipfile
import urllib.request
from tqdm import tqdm

# 配置
DATA_ROOT = "./cifar-fs-data"  # 数据集保存路径
DOWNLOAD_URL = "https://zenodo.org/record/7978538/files/cifar100.zip"
ZIP_FILE = os.path.join(DATA_ROOT, "cifar100.zip")
RAW_DIR = os.path.join(DATA_ROOT, "raw_cifar100")
PROCESSED_DIR = os.path.join(DATA_ROOT, "cifar_fs_processed")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract():
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    # 1. 下载
    if not os.path.exists(ZIP_FILE):
        print(f"正在从 Zenodo 下载源文件 (约 160MB)...")
        print(f"URL: {DOWNLOAD_URL}")
        
        import requests
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        try:
            response = requests.get(DOWNLOAD_URL, stream=True, headers=headers)
            
            if response.status_code == 200:
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024 # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(ZIP_FILE, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
            else:
                print(f"Download failed with status code: {response.status_code}")
                # Print response content if small
                if len(response.content) < 1000:
                    print(response.content)
                return # Exit if download fails
        except Exception as e:
            print(f"An error occurred during download: {e}")
            return

    else:
        print("压缩包已存在，跳过下载。")

    # 2. 解压
    print("正在解压...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zfile:
        zfile.extractall(RAW_DIR)
    
    # 3. 处理划分 (Splits)
    print("正在按照 Bertinetto et al. (2019) 标准整理 CIFAR-FS...")
    # 原始解压路径通常为 raw_cifar100/cifar100/...
    # 具体路径取决于压缩包结构，这里根据常见结构适配
    base_extract_path = os.path.join(RAW_DIR, 'cifar100') 
    split_path = os.path.join(base_extract_path, 'splits', 'bertinetto')
    source_imgs_dir = os.path.join(base_extract_path, 'data')

    splits = {
        'train': 'train.txt',
        'val': 'val.txt',
        'test': 'test.txt'
    }

    for split_name, split_file in splits.items():
        print(f"处理 {split_name} 集...")
        dest_dir = os.path.join(PROCESSED_DIR, split_name)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        split_file_path = os.path.join(split_path, split_file)
        
        with open(split_file_path, 'r') as f:
            classes = f.readlines()
            
        for class_name in classes:
            class_name = class_name.strip()
            # 源路径：data/class_name
            src = os.path.join(source_imgs_dir, class_name)
            # 目标路径：processed/train/class_name
            dst = os.path.join(dest_dir, class_name)
            
            if os.path.exists(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                print(f"警告: 找不到类别目录 {src}")

    print(f"\n✅ 成功！数据集已整理至: {PROCESSED_DIR}")
    print(f"包含文件夹: train, val, test")

if __name__ == "__main__":
    download_and_extract()