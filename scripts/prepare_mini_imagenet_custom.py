import os
import csv
import shutil
from pathlib import Path

def prepare_mini_imagenet(data_root):
    data_root = Path(data_root)
    images_dir = data_root / 'images'
    
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return

    splits = ['val', 'test']
    file_to_split = {}
    
    # Read val and test CSVs
    for split in splits:
        csv_path = data_root / f'{split}.csv'
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_to_split[row['filename']] = (split, row['label'])
        else:
            print(f"Warning: {csv_path} not found.")

    # Process images
    files = list(images_dir.glob('*.jpg'))
    print(f"Found {len(files)} images in {images_dir}")
    
    for file_path in files:
        filename = file_path.name
        
        if filename in file_to_split:
            split, label = file_to_split[filename]
        else:
            # Assume train if not in val/test
            split = 'train'
            # Extract label from filename (e.g., n0153282900000005.jpg -> n01532829)
            # Assuming label is the first 9 characters (n + 8 digits)
            label = filename[:9]
            
        # Create destination directory
        dest_dir = data_root / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Move file
        shutil.move(str(file_path), str(dest_dir / filename))
        
    print("Dataset organization complete.")

if __name__ == '__main__':
    prepare_mini_imagenet('data/Mini-ImageNet')
