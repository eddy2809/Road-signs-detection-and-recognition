import os
import random

# Directory dataset merged
imagesets_main_dir = 'PascalVoc/merged_dataset/ImageSets/Main'

# Path di input
train_file_path = os.path.join(imagesets_main_dir, 'train.txt')
valid_file_path = os.path.join(imagesets_main_dir, 'valid.txt')
test_file_path = os.path.join(imagesets_main_dir, 'test.txt')

with open(train_file_path, 'r') as f:
    image_ids = [line.strip() for line in f.readlines()]

# Effettua una permutazione casuale del dataset
random.shuffle(image_ids)

# Calcola gli indici per i 3 split
total = len(image_ids)
train_end = int(total * 0.7)
valid_end = train_end + int(total * 0.2)

# Split in train, valid e test
train_ids = image_ids[:train_end]
valid_ids = image_ids[train_end:valid_end]
test_ids = image_ids[valid_end:]

# Copia dei file
with open(train_file_path, 'w') as f:
    for image_id in train_ids:
        f.write(f"{image_id}\n")

with open(valid_file_path, 'w') as f:
    for image_id in valid_ids:
        f.write(f"{image_id}\n")

with open(test_file_path, 'w') as f:
    for image_id in test_ids:
        f.write(f"{image_id}\n")

print("Split completato:")
print(f"  Train: {len(train_ids)}")
print(f"  Valid: {len(valid_ids)}")
print(f"  Test : {len(test_ids)}")
