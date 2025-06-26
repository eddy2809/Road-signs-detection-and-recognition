import os
import shutil
import sys

print(sys.path)
sys.path.append("PascalVoc/")

# Directory dataset scaricati da roboflow
datasets = ['PascalVoc/final_datasetPVOC', 'PascalVoc/limitiPVOC', 'PascalVoc/obblighiPVOC']

# Directory di output
merged_dataset = 'merged_dataset'

# Directory del formato pascal VOC
subdirs = ['Annotations', 'JPEGImages', 'ImageSets/Main']

# Crea le cartelle di output
for subdir in subdirs:
    os.makedirs(os.path.join(merged_dataset, subdir), exist_ok=True)

merged_train_file = os.path.join(merged_dataset, 'ImageSets/Main/train.txt')

# Effettua il merge 
with open(merged_train_file, 'w') as merged_train:
    for dataset in datasets:
        annotations_dir = os.path.join(dataset, 'Annotations')
        images_dir = os.path.join(dataset, 'JPEGImages')
        train_file = os.path.join(dataset, 'ImageSets/Main/train.txt')

        with open(train_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]

        for image_id in image_ids:
            src_xml = os.path.join(annotations_dir, f'{image_id}.xml')
            dst_xml = os.path.join(merged_dataset, 'Annotations', f'{image_id}.xml')
            shutil.copy2(src_xml, dst_xml)

            src_img = os.path.join(images_dir, f'{image_id}.jpg')
            dst_img = os.path.join(merged_dataset, 'JPEGImages', f'{image_id}.jpg')
            shutil.copy2(src_img, dst_img)

            merged_train.write(f'{image_id}\n')

print("Dataset uniti con successo")
