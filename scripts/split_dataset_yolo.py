'''

Questo script divide un dataset in formato YOLO (con immagini .jpg e annotazioni .txt)
in tre sottoinsiemi: train (70%), valid (20%), e test (10%), mescolando i dati in modo casuale.
Per ciascuno split, copia le immagini e i relativi file di annotazione nelle rispettive sottocartelle 
dentro out_dataset/train, out_dataset/valid, out_dataset/test.

'''


import os
import random
import shutil
from pathlib import Path

# Per avere risultati ripetibili
random.seed(42)

#path dataset merged
base_dir = Path(".")
original_images = base_dir / "merged" / "train" / "images"
original_labels = base_dir / "merged" / "train" / "labels"

#divisione effettuata
train_perc = 0.7
val_perc= 0.2
#test_perc = 0.1

# legge tutti i nomi di file jpg
image_list = list(original_images.glob("*.jpg"))

#Mescola i file,
random.shuffle(image_list)

# calcola il numero di file per ogni split
n_total = len(image_list)
n_train = int(train_perc * n_total)
n_val = int(val_perc * n_total)

train_files = image_list[:n_train]
val_files = image_list[n_train:n_train + n_val]
test_files = image_list[n_train + n_val:]

splits = {
    "train": train_files,
    "valid": val_files,
    "test": test_files
}

#funzione utility per copiare immagini e relative labe
def copy_files(file_list, split_name):
    base_dir = Path(".")
    
    img_out_dir = base_dir / "out_dataset" / split_name / "images"
    lbl_out_dir = base_dir / "out_dataset" / split_name / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in file_list:
        #copia dell'immagine
        shutil.copy2(img_path, img_out_dir / img_path.name)
        
        #copia della label
        label_path = original_labels / (img_path.stem + ".txt")
        shutil.copy2(label_path, lbl_out_dir / label_path.name)

for split_name, files in splits.items():
    copy_files(files, split_name)

print("Split effettuato")