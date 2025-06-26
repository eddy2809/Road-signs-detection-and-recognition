
import os
import torch
import xml.etree.ElementTree as ET
import torchvision.transforms as T
import sys
sys.path.append(os.path.abspath(".."))
from torch.utils.data import Dataset
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, root, image_set="train"):
        """
        Costruttore della classe VOCDataset.

        :param root: root directory del dataset
        :type root: str
        :param image_set: nome del set di immagini, default="train"
        :type image_set: str
        """
        self.root = root
        self.transforms = T.Compose([
                                T.ToTensor()                         
                            ])
        image_set_file = os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")
        with open(image_set_file) as f:
            self.image_ids = [x.strip() for x in f.readlines()]
        self.img_dir = os.path.join(root, "JPEGImages")
        self.ann_dir = os.path.join(root, "Annotations")
        self.classes = ["__background__"] + self._find_classes()
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith((".jpg", ".png"))])

    def _find_classes(self):
        """ Trova le classi presenti nel dataset, leggendo le annotation delle immagini
            :return: lista di stringhe contenenti i nomi delle classi
        """
        cls = set()
        for fname in os.listdir(self.ann_dir):
            tree = ET.parse(os.path.join(self.ann_dir, fname))
            for obj in tree.findall("object"):
                cls.add(obj.find("name").text)
        return sorted(list(cls))

    def __getitem__(self, idx):
        """
        Restituisce l'immagine e la sua annotazione a partire dall'indice idx.
        
        :param idx: indice dell'immagine richiesta
        :type idx: int
        :return: immagine e annotazione dell'immagine
        :rtype: tuple (torch.Tensor, dict)
        """
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        ann_path = os.path.join(self.ann_dir, f"{image_id}.xml")

        img = Image.open(img_path).convert("RGB")
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes, labels = [], []
        for obj in root.findall("object"):
            label = obj.find("name").text
            labels.append(self.classes.index(label))

            bndbox = obj.find("bndbox")
            bbox = [
                float(bndbox.find("xmin").text),
                float(bndbox.find("ymin").text),
                float(bndbox.find("xmax").text),
                float(bndbox.find("ymax").text),
            ]
            boxes.append(bbox)

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        """
        Restituisce la dimensione del dataset.

        """
        return len(self.image_ids)
    