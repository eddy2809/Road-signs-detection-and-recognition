import torch
import gc
import sys
import os
sys.path.append(os.path.abspath(".."))
from torch.utils.data import DataLoader
from utils.dataset import VOCDataset
from utils.collate import collate_fn
from utils.engine import evaluate_metrics
from models.faster_rcnn import get_model
from utils.engine import load_custom_fasterrcnn_model



def eval(experiment_name,model,dataset_path,batch_size=1,imgsz=416):
    
    #EVALUATION SUL TEST SET CON IL BEST MODEL SELEZIONATO
    """
    
    Valuta un modello Faster R-CNN su un set di dati di test utilizzando i pesi del miglior modello dell'esperimento passato in input.

    Argomenti:
        experiment_name (str): il nome dell'esperimento, utilizzato per individuare i pesi del modello.
        model (torch.nn.Module): il modello Faster R-CNN da valutare.
        dataset_path (str): il percorso del set di dati in formato VOC.
        batch_size (int, opzionale): il numero di campioni per batch da caricare. Il valore predefinito è 1.
        imgsz (int, opzionale): la dimensione delle immagini utilizzate per la valutazione. Il valore predefinito è 416.

    Restituisce:
        Nessuno

    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset_train = VOCDataset(dataset_path, image_set="train")
    dataset_test = VOCDataset(dataset_path, image_set="test")
    data_loader_test = DataLoader(dataset_test, batch_size, shuffle=True, collate_fn=collate_fn)
    model =  load_custom_fasterrcnn_model(model_path=f"{experiment_name}/weights/fasterrcnn_voc_best.pth", num_classes=len(dataset_train.classes),imgsz=imgsz)
    print("Score sul test set:\t")

    evaluate_metrics(experiment_name=experiment_name,model=model, data_loader=data_loader_test, device=device, epoch=None,set="test")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()