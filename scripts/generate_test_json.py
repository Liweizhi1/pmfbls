import os
import json
import random
from pathlib import Path

def generate_test_json(data_root, n_episodes=1000, n_way=5, n_shot=1, n_query=15, output_name=None):
    data_root = Path(data_root)
    # CHANGE: Read from 'test' directory
    test_dir = data_root / 'test'
    
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return

    classes = [d.name for d in test_dir.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} test classes.")
    
    class_images = {}
    for cls in classes:
        # Support both jpg and png
        imgs = list((test_dir / cls).glob('*.png')) + list((test_dir / cls).glob('*.jpg'))
        class_images[cls] = [f"{cls}/{img.name}" for img in imgs]
        
    episodes = []
    for _ in range(n_episodes):
        sampled_classes = random.sample(classes, n_way)
        support = []
        query = []
        
        for cls in sampled_classes:
            imgs = class_images[cls]
            if len(imgs) < n_shot + n_query:
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
        output_name = f"test{n_episodes}Episode_{n_way}_way_{n_shot}_shot.json"
        
    output_path = data_root / output_name
    with open(output_path, 'w') as f:
        json.dump(episodes, f)
    print(f"Generated {output_path}")

if __name__ == '__main__':
    # Generate 1-shot
    generate_test_json('data/cifar-fs', n_episodes=1000, n_way=5, n_shot=1, n_query=15, output_name='test1000Episode_5_way_1_shot.json')
    # Generate 5-shot
    generate_test_json('data/cifar-fs', n_episodes=1000, n_way=5, n_shot=5, n_query=15, output_name='test1000Episode_5_way_5_shot.json')
