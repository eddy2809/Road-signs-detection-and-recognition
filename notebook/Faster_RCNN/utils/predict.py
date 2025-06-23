import torch
import gc
import sys
import os
sys.path.append(os.path.abspath(".."))
from utils.dataset import VOCDataset    
from utils.engine import show_predictions
from utils.engine import load_custom_fasterrcnn_model

def dataset_predict(experiment_name,model,dataset_path):

    """
    Valuta un modello Faster R-CNN su un set di dati di test utilizzando i pesi del miglior modello dell'esperimento passato in input.

    Argomenti:
        experiment_name (str): il nome dell'esperimento, utilizzato per individuare i pesi del modello.
        model (torch.nn.Module): il modello addestrato.
        dataset_path (str): il percorso del set di dati in formato VOC.

    Restituisce:
        Nessuno
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset_test = VOCDataset(dataset_path, image_set="test")
    _,classes = load_custom_fasterrcnn_model(model_path=f"{experiment_name}/weights/fasterrcnn_voc_best.pth",imgsz=416)
    show_predictions(experiment_name=experiment_name,model=model,dataset=dataset_test,device=device,score_threshold=0.25,class_names=classes)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()