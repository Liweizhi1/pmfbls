import os
import json
import random
from pathlib import Path

def generate_json(data_root, n_episodes=1000, n_way=5, n_shot=1, n_query=15, output_name=None):
    data_root = Path(data_root)
    val_dir = data_root / 'val'
    
    if not val_dir.exists():
        print(f"Validation directory not found: {val_dir}")
        return

    classes = [d.name for d in val_dir.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} validation classes.")
    
    class_images = {}
    for cls in classes:
        imgs = list((val_dir / cls).glob('*.jpg'))
        class_images[cls] = [f"{cls}/{img.name}" for img in imgs]
        
    episodes = []
    for _ in range(n_episodes):
        sampled_classes = random.sample(classes, n_way)
        support = []
        query = []
        
        for cls in sampled_classes:
            imgs = class_images[cls]
            # Ensure enough images
            if len(imgs) < n_shot + n_query:
                # If not enough, sample with replacement or just take all
                sampled_imgs = random.choices(imgs, k=n_shot + n_query)
            else:
                sampled_imgs = random.sample(imgs, n_shot + n_query)
            
            support.append(sampled_imgs[:n_shot])
            query.append(sampled_imgs[n_shot:])
            
        episodes.append({
            "Support": support,
            "Query": query
        })
        
    if output_name is None:
        output_name = f"val{n_episodes}Episode_{n_way}_way_{n_shot}_shot.json"
        
    output_path = data_root / output_name
    with open(output_path, 'w') as f:
        json.dump(episodes, f)
    print(f"Generated {output_path}")

if __name__ == '__main__':
    # Generate 1-shot
    generate_json('data/Mini-ImageNet', n_shot=1)
    # Generate 5-shot
    generate_json('data/Mini-ImageNet', n_shot=5)
