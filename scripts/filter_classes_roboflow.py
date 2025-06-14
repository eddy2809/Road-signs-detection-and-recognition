#Script per filtrare le classi da un dataset roboflow, selezionando solo quelle scelte per il progetto

import os
import shutil


#path input e output
ROBOFLOW_IMG_DIR = "dataset/train/images"
ROBOFLOW_LBL_DIR = "dataset/train/labels"
ROBOFLOW_OUT_LBL_DIR = "dataset/train/labels_filtered"
ROBOFLOW_OUT_IMG_DIR = "dataset/train/images_filtered"


#mapping delle classi
selected_classes_mapping = {
    "Give Way": 8,
    "Turn right ahead": 27,
    # altre classe: indice array
}

#classi del dataset di roboflow
roboflow_classes = [
    "-Road narrows on right", "50 mph speed limit", "Attention Please-", "Beware of children",
    "CYCLE ROUTE AHEAD WARNING", "Dangerous Left Curve Ahead", "Dangerous Right Curve Ahead",
    "End of all speed and passing limits", "Give Way", "Go Straight or Turn Right",
    "Go straight or turn left", "Keep-Left", "Keep-Right", "Left Zig Zag Traffic",
    "No Entry", "No_Over_Taking", "Overtaking by trucks is prohibited", "Pedestrian Crossing",
    "Round-About", "Slippery Road Ahead", "Speed Limit 20 KMPH", "Speed Limit 30 KMPH",
    "Stop_Sign", "Straight Ahead Only", "Traffic_signal", "Truck traffic is prohibited",
    "Turn left ahead", "Turn right ahead", "Uneven Road"
]


#selezione classi di interesse
id_mapping = {
    roboflow_classes.index(k): v for k, v in selected_classes_mapping.items()
}

#creazione directory output
os.makedirs(ROBOFLOW_OUT_LBL_DIR, exist_ok=True)
os.makedirs(ROBOFLOW_OUT_IMG_DIR, exist_ok=True)

kept_images = set()


#filtraggio classi
for filename in os.listdir(ROBOFLOW_LBL_DIR):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(ROBOFLOW_LBL_DIR, filename)
    output_path = os.path.join(ROBOFLOW_OUT_LBL_DIR, filename)

    with open(input_path, "r") as f:
        lines = f.readlines()

    #prende l'etichetta della classe (fare refactoring con pandas)
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        if class_id in id_mapping:
            parts[0] = str(id_mapping[class_id])
            new_lines.append(" ".join(parts) + "\n")

    if new_lines:
        with open(output_path, "w") as f:
            f.writelines(new_lines)
        img_name = os.path.splitext(filename)[0]
        kept_images.add(img_name)


#copia immagini ed etichette filtrate
for filename in os.listdir(ROBOFLOW_IMG_DIR):
    name, ext = os.path.splitext(filename)
    if name in kept_images and ext.lower() in [".jpg", ".jpeg", ".png"]:
        src = os.path.join(ROBOFLOW_IMG_DIR, filename)
        dst = os.path.join(ROBOFLOW_OUT_IMG_DIR, filename)
        shutil.copy(src, dst)

print(f"{len(kept_images)} immagini selezionate.")
