import os
import shutil

# Cartelle di origine
splits = ['train', 'valid', 'test']
source_root = 'dataset_rbfl'             # es. cartella scaricata da Roboflow
target_root = 'dataset'          # nuova struttura Pascal VOC classica

# Crea cartelle target
os.makedirs(f"{target_root}/Annotations", exist_ok=True)
os.makedirs(f"{target_root}/JPEGImages", exist_ok=True)
os.makedirs(f"{target_root}/ImageSets/Main", exist_ok=True)

# Elabora ogni split (train/val/test)
for split in splits:
    split_dir = os.path.join(source_root, split)
    list_path = os.path.join(target_root, 'ImageSets/Main', f"{'val' if split == 'valid' else split}.txt")

    with open(list_path, "w") as f:
        for file in os.listdir(split_dir):
            if file.endswith(".jpg"):
                name = os.path.splitext(file)[0]
                # Copia immagini e annotazioni
                shutil.copy(os.path.join(split_dir, f"{name}.jpg"), os.path.join(target_root, "JPEGImages", f"{name}.jpg"))
                shutil.copy(os.path.join(split_dir, f"{name}.xml"), os.path.join(target_root, "Annotations", f"{name}.xml"))
                # Aggiungi alla lista
                f.write(f"{name}\n")
