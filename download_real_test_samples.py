"""
Downloads authentic real dermoscopic captures from the held-out HAM10000 test set
for the Quick Load sample library in the Streamlit Dashboard.
"""
import os
import urllib.request
import json
import pandas as pd
from dataset import LABEL_TO_IDX, create_stratified_splits

OUTPUT_DIR = "./real_test_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load real metadata and perform strict lesion-id split
df = pd.read_csv('HAM10000_metadata.csv')
if 'label' not in df.columns:
    df['label'] = df['dx'].map(LABEL_TO_IDX)

train_df, val_df, test_df = create_stratified_splits(df, val_ratio=0.15, test_ratio=0.15, random_state=42)
print(f"[Samples] Partitioned HAM10000: {len(test_df)} test images across {test_df['lesion_id'].nunique()} unseen lesions.")

# 2. Get file mapping from HuggingFace repo
hf_api_url = 'https://huggingface.co/api/datasets/aditya022/HAM10000_Dataset'
req = urllib.request.Request(hf_api_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    siblings = [s['rfilename'] for s in data.get('siblings', [])]
    img_map = {s.split('/')[-1].replace('.jpg',''): s for s in siblings if s.endswith('.jpg')}

# 3. Sample 2 real cases per diagnostic class from the unseen test set (14 total)
sampled_records = []
for dx_class in ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']:
    class_test_subset = test_df[test_df['dx'] == dx_class]
    # Sample 2 distinct lesions
    selected = class_test_subset.drop_duplicates(subset=['lesion_id']).head(2)
    for _, row in selected.iterrows():
        img_id = row['image_id']
        if img_id in img_map:
            repo_path = img_map[img_id]
            dl_url = f"https://huggingface.co/datasets/aditya022/HAM10000_Dataset/resolve/main/{repo_path}"
            dest_path = os.path.join(OUTPUT_DIR, f"{img_id}.jpg")
            print(f"Downloading real {dx_class.upper()} sample: {img_id}...")
            urllib.request.urlretrieve(dl_url, dest_path)
            sampled_records.append({
                'image_id': img_id,
                'lesion_id': row['lesion_id'],
                'dx': row['dx'],
                'age': row.get('age', 50),
                'sex': row.get('sex', 'unknown'),
                'localization': row.get('localization', 'skin'),
                'path': dest_path
            })

samples_df = pd.DataFrame(sampled_records)
samples_df.to_csv(os.path.join(OUTPUT_DIR, "samples_metadata.csv"), index=False)
print(f"[SUCCESS] Downloaded {len(samples_df)} real HAM10000 test lesions into {OUTPUT_DIR}")
