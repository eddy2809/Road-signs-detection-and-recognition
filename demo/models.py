# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 20:00:01 2025

@author: loren
"""

# models.py
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import requests

def download_from_hf(url, output_path):
    if not os.path.exists(output_path):
        print("🔽 Scaricamento modello da Hugging Face...")
        r = requests.get(url)
        with open(output_path, "wb") as f:
            f.write(r.content)

def get_model(num_classes, imgsz):
    """
    Crea un modello Faster R-CNN con backbone ResNet-50-FPN, pre-addestrato sul set di dati COCO,
    personalizzato per un numero specificato di classi e dimensioni dell'immagine.

    Argomenti:
        num_classes (int): il numero di classi per il rilevamento, compreso lo sfondo.
        imgsz (int): la dimensione dell'immagine per il ridimensionamento delle immagini di input.

    Restituisce:
        torchvision.models.detection.FasterRCNN: un modello Faster R-CNN pronto per l'addestramento o l'inferenza.

    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, min_size=imgsz, max_size=imgsz
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_custom_fasterrcnn_model(model_path, imgsz):
    """
    Carica un modello Faster R-CNN ResNet50 addestrato per un task personalizzato,
    usando i pesi salvati con `state_dict`.
    
    Args:
        model_path (str): Percorso del file .pth contenente lo state_dict salvato.
        num_classes (int): Numero totale di classi (incluso lo sfondo).
        
    Returns:
        torch.nn.Module: Modello caricato e spostato su GPU (se disponibile).
    """
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    hf_url = "https://huggingface.co/lorelarocca2001/FasterRCNN_best/resolve/main/fasterrcnn_voc_best.pth"
    download_from_hf(hf_url, model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = torch.load(model_path, map_location=device)
    model_weights = checkpoints["model_state_dict"]
    classes = checkpoints["class_names"]
    model = get_model(num_classes=20, imgsz=imgsz)  
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()
    return model, classes

