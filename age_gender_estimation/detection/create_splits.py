#!/usr/bin/env python3
"""
Create split files for CrowdHuman dataset based on annotation directories.
"""

from pathlib import Path
import json

def create_splits():
    """Create train.txt, val.txt, and test.txt split files."""
    
    # Directories
    img_dir = Path('/workspace/data/crowdhuman/Images')
    img_test_dir = Path('/workspace/data/crowdhuman/images_test')
    ann_train_dir = Path('/workspace/data/crowdhuman/annotations_json/annotation_train')
    ann_val_dir = Path('/workspace/data/crowdhuman/annotations_json/annotation_val')
    splits_dir = Path('/workspace/portfolio/age_gender_estimation/detection/splits')
    
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    all_images = {}
    for img_path in img_dir.glob('*.jpg'):
        all_images[img_path.stem] = img_path.resolve()
    for img_path in img_dir.glob('*.png'):
        all_images[img_path.stem] = img_path.resolve()
    
    # Get annotation IDs
    train_ids = set()
    val_ids = set()
    
    for ann_file in ann_train_dir.glob('*.json'):
        try:
            with open(ann_file) as f:
                data = json.load(f)
                img_id = data.get('ID', '')
                train_ids.add(img_id)
                train_ids.add(img_id.replace(',', '_'))
        except Exception as e:
            print(f"Error reading {ann_file}: {e}")
    
    for ann_file in ann_val_dir.glob('*.json'):
        try:
            with open(ann_file) as f:
                data = json.load(f)
                img_id = data.get('ID', '')
                val_ids.add(img_id)
                val_ids.add(img_id.replace(',', '_'))
        except Exception as e:
            print(f"Error reading {ann_file}: {e}")
    
    # Split images
    train_images = []
    val_images = []
    test_images = []
    
    for img_stem, img_path in all_images.items():
        if img_stem in train_ids or img_stem.replace('_', ',') in train_ids:
            train_images.append(str(img_path))
        elif img_stem in val_ids or img_stem.replace('_', ',') in val_ids:
            val_images.append(str(img_path))
        else:
            # Check if it's in test directory
            test_path = img_test_dir / img_path.name
            if test_path.exists():
                test_images.append(str(test_path.resolve()))
    
    # Add test directory images
    for img_path in img_test_dir.glob('*.jpg'):
        if str(img_path.resolve()) not in test_images:
            test_images.append(str(img_path.resolve()))
    for img_path in img_test_dir.glob('*.png'):
        if str(img_path.resolve()) not in test_images:
            test_images.append(str(img_path.resolve()))
    
    # Write split files
    with open(splits_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(sorted(train_images)))
    
    with open(splits_dir / 'val.txt', 'w') as f:
        f.write('\n'.join(sorted(val_images)))
    
    with open(splits_dir / 'test.txt', 'w') as f:
        f.write('\n'.join(sorted(test_images)))
    
    print(f"Created split files:")
    print(f"  Train: {len(train_images)} images -> {splits_dir / 'train.txt'}")
    print(f"  Val: {len(val_images)} images -> {splits_dir / 'val.txt'}")
    print(f"  Test: {len(test_images)} images -> {splits_dir / 'test.txt'}")

if __name__ == '__main__':
    create_splits()

