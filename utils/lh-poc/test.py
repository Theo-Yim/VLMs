import pandas as pd
import os
from dataloader import LHDataLoader

data_root = "/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data/K-LH-302 2025-08-22 155843_export"
image_root = "/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data-image/image/20250722"
result_dir = "/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data-result"
plot_dir = "/mnt/nas2/users/sbchoi/kh-practices/lh-poc/plot"
os.makedirs(result_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

loader = LHDataLoader(data_root, image_root)

# CSV 파일 읽기
df = pd.read_csv('plot/similarity_scores.csv')

# unknown인 predicted_defect 찾기
unknown_df = df[df['predicted_defect'] == 'unknown']
unknown_label_ids = set(unknown_df['label_id'].tolist())

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images = []
image_paths = []
for item in loader:
    data_key = item['meta_data']['data_key']
    label_id = item['label_id']
    result_path = os.path.join(result_dir, f"{label_id}.txt")
    if label_id not in unknown_label_ids:
        continue

    image_path = os.path.join(image_root, data_key)
    if os.path.exists(image_path):
        images.append(image_path)
        image_paths.append(image_path)
    else:
        print(f"Image does not exist: {image_path}")
        
    if len(images) == 9:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for idx, img_path in enumerate(images):
            row = idx // 3
            col = idx % 3
            img = mpimg.imread(img_path)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Image {idx+1}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'unknown_defects_{len(image_paths)//9}.png'))
        plt.close()
        images = []

# Handle remaining images if any
if images:
    rows = (len(images) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
    for idx, img_path in enumerate(images):
        row = idx // 3
        col = idx % 3
        img = mpimg.imread(img_path)
        if rows > 1:
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Image {idx+1}')
        else:
            axes[col].imshow(img)
            axes[col].axis('off')
            axes[col].set_title(f'Image {idx+1}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'unknown_defects_{(len(image_paths)//9)+1}.png'))
    plt.close()

print("하자유형이 unknown인 label_id 목록:")
for label_id in unknown_label_ids:
    print(label_id)


