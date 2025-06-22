import sys
import os
import pandas as pd
import torch
import random

sys.path.append(os.path.abspath(".."))

from models.faster_rcnn import get_model
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


def evaluate_metrics(experiment_name,model, data_loader, device,epoch,set):
    """
    Valuta il modello su un determinato set di dati e salva i risultati in un file CSV.

    Argomenti:
        experiment_name (str): nome dell'esperimento
        model (nn.Module): il modello da valutare
        data_loader (DataLoader): il caricatore di dati contenente il set di dati da valutare
        device (torch.device): il dispositivo da utilizzare per la valutazione
        epoch (int, opzionale): l'epoca corrente. Se fornita, verrà salvata nel file CSV
        set (str): il nome del set di dati da valutare (ad esempio “train”, ‘val’, “test”)

    Restituisce:
        mAP50 (float): la precisione media a IoU=0,50

    """
    csv_path = f"{experiment_name}/{set}_metrics.csv"
    mAP50 = 0.0
    model.eval()
    metric = MeanAveragePrecision()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="\t\tEvaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            metric.update(outputs, targets)
            torch.cuda.empty_cache()

    results = metric.compute()

    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            if(k=="map_50"):
                print(f"\t\t{k}: {v.item() if v.numel() == 1 else v}")
                mAP50 = v.item()
    metric.reset()
    # Save to CSV
    summary = {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else str(v)
               for k, v in results.items()}

    if (epoch is not None): summary = dict([("epoch", epoch)] + list(summary.items()))

    df = pd.DataFrame([summary])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)
    return mAP50



def show_predictions(experiment_name,model, dataset, device, score_threshold=0.25,class_names=[]):
    """
    Salva le predizioni del modello in immagini con le box disegnate sopra.
    
    Args:
        experiment_name (str): Nome dell'esperimento.
        model (torch.nn.Module): Modello addestrato.
        dataset (VOCDataset): Dataset contenente le immagini da elaborare.
        device (torch.device): Dispositivo su cui elaborare i dati.
        score_threshold (float, optional): Soglia di punteggio minimo per considerare una predizione valida. Defaults to 0.25.
        class_names (list, optional): Nomi delle classi. Defaults to [].
    
    Restituisce:
        None
    """
    
    model.eval()
    output_dir=f"{experiment_name}/predictions"
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(dataset)):
        img, target = dataset[idx]
        img_tensor = img.to(device).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)[0]

        # Filtro predizioni con punteggio basso
        keep = output['scores'] >= score_threshold
        pred_boxes = output['boxes'][keep].cpu()
        pred_labels = output['labels'][keep].cpu()
        scores = output['scores'][keep].cpu()

        # Disegna box predetti
        pred_img = draw_bounding_boxes((img * 255).byte(), pred_boxes, labels=[f"{class_names[l.item()]}: {s.item():.2f}" for l,s in zip(pred_labels,scores)],
                                       colors=["#{:06x}".format(random.randint(0, 0xFFFFFF)) for l in pred_labels], width=2,font_size=20,font="arial")

        # Nome originale immagine
        original_filename = dataset.image_files[idx] # Es: "img001.jpg"
        output_filename = os.path.splitext(original_filename)[0] + "_pred.jpg"

        # Salva immagine
        img_pil = to_pil_image(pred_img)
        img_pil.save(os.path.join(output_dir, output_filename))
        
    print(f"saved prediction to: {output_dir}")

def load_custom_fasterrcnn_model(model_path, num_classes,imgsz):
    """
    Carica un modello Faster R-CNN ResNet50 addestrato per un task personalizzato,
    usando i pesi salvati con `state_dict`.
    
    Args:
        model_path (str): Percorso del file .pth contenente lo state_dict salvato.
        num_classes (int): Numero totale di classi (incluso lo sfondo).
        
    Returns:
        torch.nn.Module: Modello caricato e spostato su GPU (se disponibile).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica il modello base
    model = get_model(num_classes=num_classes,imgsz=imgsz)

    # Carica i pesi salvati
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))

    # Sposta il modello sul dispositivo corretto
    model.to(device)
    model.eval()

    return model
