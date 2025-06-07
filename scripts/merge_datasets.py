# -*- coding: utf-8 -*-
"""
Created on Wed May 28 13:58:00 2025

@author: loren
"""

import os
import shutil

# Path ai tre dataset e alla destinazione finale
base_path = "dataset_di_base/train"
add1_path = "precedenza_obblighi_unire/train"
add2_path = "limiti_da_unire/train"
final_path = "dataset_finale/train"

# Crea le cartelle finali
os.makedirs(os.path.join(final_path, "images"), exist_ok=True)
os.makedirs(os.path.join(final_path, "labels"), exist_ok=True)

# Mapping classi base
class_list = ['green_light', 'make_way', 'no_entry', 'pedestrian_crossing', 'red_light',
              'speed_limit_100', 'speed_limit_110', 'speed_limit_120', 'speed_limit_20',
              'speed_limit_30', 'speed_limit_40', 'speed_limit_50', 'speed_limit_60',
              'speed_limit_70', 'speed_limit_80', 'speed_limit_90', 'stop', 'turn_left', 'turn_right']
class_map = {name: i for i, name in enumerate(class_list)}

def copy_and_remap(source_path, final_path):
    # Scorre ogni file .txt nella cartella labels del dataset 
    for label_file in os.listdir(os.path.join(source_path, "labels")):
        # Costruisce i percorsi ai file label e immagine
        label_path = os.path.join(source_path, "labels", label_file)
        image_path = os.path.join(source_path, "images", label_file.replace(".txt", ".jpg"))
        if not os.path.exists(image_path):  
            image_path = image_path.replace(".jpg", ".png")

        # Copia l'immagine nella destinazione finale
        shutil.copy(image_path, os.path.join(final_path, "images", os.path.basename(image_path)))

        # Legge tutte le righe del file .txt
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Estrae il class_id (es. 0) per ogni riga del file di label.
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            class_name = None

            # Traduce l'ID locale nel nome della classe
            if "precedenza_obblighi" in source_path:
                class_name = ['make_way', 'turn_left', 'turn_right'][class_id]
            elif "limiti_da_unire" in source_path:
                class_name = ['speed_limit_110', 'speed_limit_90'][class_id]
                
            # Ricalcola il nuovo ID di classe e costruisce la nuova riga
            new_class_id = class_map[class_name]
            parts[0] = str(new_class_id)
            new_lines.append(" ".join(parts))

        # Scrive il file delle etichette aggiornato
        with open(os.path.join(final_path, "labels", label_file), "w") as f:
            f.write("\n".join(new_lines))

# Copia tutti i file images/ e labels/ dal dataset base nel dataset finale senza modifiche (le classi sono già coerenti)
for subdir in ["images", "labels"]:
    src = os.path.join(base_path, subdir)
    dst = os.path.join(final_path, subdir)
    for file in os.listdir(src):
        shutil.copy(os.path.join(src, file), os.path.join(dst, file))

#  Aggiunta dei due dataset extra con remapping
copy_and_remap(add1_path, final_path)
copy_and_remap(add2_path, final_path)
