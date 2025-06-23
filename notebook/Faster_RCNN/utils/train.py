import torch
import os
import gc
import sys
sys.path.append(os.path.abspath(".."))
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import VOCDataset
from utils.collate import collate_fn
from tqdm import tqdm
from utils.engine import evaluate_metrics
from models.faster_rcnn import get_model



def dataset_train(experiment_name,dataset_path,epochs=5,imgsz=416,momentum=0.9,lr=0.005):
    
    """
    Addestra un modello Faster R-CNN con un backbone ResNet50 su un set di dati personalizzato.
    
    Argomenti:
        experiment_name (str): il nome dell'esperimento, utilizzato per salvare i log e i pesi del modello.
        dataset_path (str): percorso del set di dati, che deve essere in formato VOC.
        epochs (int, opzionale): numero di epoche di addestramento. Il valore predefinito é 5.
        imgsz (int, opzionale): dimensione delle immagini utilizzate per l'addestramento. Il valore predefinito è 416.

    Restituisce:
        torch.nn.Module: il modello addestrato.
    """
    batch_size=1 #fisso per frcnn altrimenti da problemi
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_weights_dir = f"{experiment_name}/weights"
    
    writer = SummaryWriter(f"logs/faster_rcnn50/{experiment_name}")
    
    dataset_train = VOCDataset(dataset_path, image_set="train")
    dataset_valid = VOCDataset(dataset_path, image_set="val")
    data_loader_train = DataLoader(dataset_train, batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_valid = DataLoader(dataset_valid, batch_size, shuffle=True, collate_fn=collate_fn)



    model = get_model(num_classes=len(dataset_train.classes),imgsz=imgsz)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_score_valid = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_loss_valid = 0.0
        print(f"Epoca {epoch+1}/{epochs}: \n\tLoss sul train set:")
        for images, targets in tqdm(data_loader_train, desc=f"\t\t"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
            total_loss += losses.item()
            torch.cuda.empty_cache()
            

        
        lr_scheduler.step()
        
        print(f"\t\t-Score: {total_loss:.4f}")
        print(f"\tLoss sul Valid set:")
        with torch.set_grad_enabled(False):
            for images, targets in tqdm(data_loader_valid, desc=f"\t\t"):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                
                total_loss_valid += losses.item()
                
                torch.cuda.empty_cache()
            
        print(f"\t\t-Score: {total_loss_valid:.4f}")    
        writer.add_scalar("loss/train", total_loss, epoch+1)
        writer.add_scalar("loss/valid", total_loss_valid, epoch+1)
        
        # Crea un dizionario per contenere tutti i dati da salvare insieme al modello
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'class_names': dataset_train.classes,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'total_loss': total_loss,
            'total_loss_valid': total_loss_valid,
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }

        #EVALUATION PER EPOCA SUL VALID SET
        model.eval() 
        print("\tEvaluation sul valid set:")
        current_score_valid = evaluate_metrics(experiment_name=experiment_name,model=model, data_loader=data_loader_valid, device=device, epoch=epoch+1,set="valid")

        os.makedirs(save_weights_dir, exist_ok=True)
        
        if current_score_valid > best_score_valid:
            best_score_valid = current_score_valid
            torch.save(checkpoint, f"{save_weights_dir}/fasterrcnn_voc_best.pth")
            

        torch.save(checkpoint, f"{save_weights_dir}/fasterrcnn_voc_last.pth")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Training complete. Model saved.\n")
    return model