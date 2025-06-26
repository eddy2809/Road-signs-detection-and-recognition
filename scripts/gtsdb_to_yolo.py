
"""
Questo script filtra e converte le annotazioni del dataset GTSDB (German Traffic Sign Detection Benchmark)
nel formato YOLO, salvando solo le classi specificate in class_map. Il risultato è dunque 
un sottoinsieme del dataset GTSDB già pronto per essere usato con YOLO.
"""

import os
from PIL import Image
import shutil

# Mapping classi da ID originale a ID YOLO
class_map = {
   17: 0,  # Divieto di accesso -> no entry
   0: 1,   # Limite di velocità 20
   1: 2,   # Limite di velocità 30
   2: 3,   # Limite di velocità 50
   3: 4,   # Limite di velocità 60
   4: 5,   # Limite di velocità 70
   5: 6,   # Limite di velocità 80
   7: 7,   # Limite di velocità 100
   8: 8,   # Limite di velocità 120
   14: 9,  # Stop
   13: 10, # Dare precedenza -> give way
   34: 11, # Obbligo a sinistra -> go left
   33: 12, # Obbligo a destra -> go right
   19: 13, # Curva a sinistra
   20: 14, # Curva a destra
   27: 15, # Attraversamento pedonale
   9: 16   # Divieto di sorpasso
}

# Percorsi originali
img_dir = 'FULLIJCNN2013'
gt_file = os.path.join(img_dir, 'gt.txt')

# Percorsi di output
output_base = 'GTSDB_YOLO_filtered'
output_img = os.path.join(output_base, 'images')
output_lbl = os.path.join(output_base, 'labels')
os.makedirs(output_img, exist_ok=True)
os.makedirs(output_lbl, exist_ok=True)

# Leggi le annotazioni
with open(gt_file, 'r') as f:
    lines = f.readlines()

# Conversione
for line in lines:
    parts = line.strip().split(';')
    filename = parts[0]
    left, top, right, bottom = map(int, parts[1:5])
    class_id = int(parts[5])

    if class_id not in class_map:
        continue  # ignora le classi escluse

    yolo_class_id = class_map[class_id]

    img_path = os.path.join(img_dir, filename)
    if not os.path.exists(img_path):
        continue  # immagine mancante

    # Copia l'immagine nella nuova cartella
    shutil.copy(img_path, os.path.join(output_img, filename))

    # Ottieni dimensioni immagine
    img = Image.open(img_path)
    img_w, img_h = img.size

    # Bounding box in formato YOLO
    x_center = ((left + right) / 2) / img_w
    y_center = ((top + bottom) / 2) / img_h
    bbox_width = (right - left) / img_w
    bbox_height = (bottom - top) / img_h

    # Scrivi file YOLO
    label_path = os.path.join(output_lbl, os.path.splitext(filename)[0] + '.txt')
    with open(label_path, 'a') as out_file:
        out_file.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
