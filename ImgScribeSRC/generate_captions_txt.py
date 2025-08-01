# generate_captions_txt.py

import os

# File paths
base_dir = "Flickr8k_Dataset"
token_file = os.path.join(base_dir, "Flickr8k_text", "Flickr8k.token.txt")
train_file = os.path.join(base_dir, "Flickr8k_text", "Flickr_8k.trainImages.txt")
output_file = os.path.join(base_dir, "captions.txt")

# Load training image list
with open(train_file, "r") as f:
    train_images = set(line.strip() for line in f)

# Extract #0 caption for each image
captions = {}

with open(token_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        img_tag, caption = parts
        filename = img_tag.split("#")[0]
        if filename in train_images and img_tag.endswith("#0"):
            captions[filename] = caption.strip()

# Write to captions.txt
with open(output_file, "w", encoding="utf-8") as f:
    for fname, cap in captions.items():
        f.write(f"{fname}|{cap}\n")

print(f"✅ Saved {len(captions)} captions to {output_file}")
