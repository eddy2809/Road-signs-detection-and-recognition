import sys
import os
import torch
import random
sys.path.append(os.path.abspath(".."))
from utils.engine import load_custom_fasterrcnn_model
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


def compute_img(model_path, img,device,imgsz=416):
    """
    Effettua predizioni su un'immagine utilizzando un modello Faster R-CNN
    pre-addestrato e caricato da disco.

    Args:
        model_path (str): Percorso del file .pth contenente lo state_dict del modello
        img (str): Percorso dell'immagine da elaborare
        device (torch.device): Dispositivo su cui effettuare le operazioni
        imgsz (int): Dimensione dell'immagine di input (default: 416)

    Returns:
        output (dict): Dizionario contenente le predizioni del modello
        classes (list): Lista delle classi presenti nel dataset
        input_tensor (torch.Tensor): Tensor dell'immagine elaborata
    """
    model,classes = load_custom_fasterrcnn_model(model_path=model_path,imgsz=imgsz)
    
    image = Image.open(img).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((imgsz, imgsz))])
    input_tensor = transform(image)
    input_tensor = input_tensor.to(device)

    input_list_of_tensors = [input_tensor]
    model.eval()  
    with torch.no_grad():
        output = model(input_list_of_tensors)

    return output,classes,input_tensor

def predict_img(model_path, img_path,imgsz=416,score_threshold=0.25):
    """
    Genera un'immagine con le predizioni del modello su un'immagine.

    Args:
        model_path (str): Percorso del file .pth contenente lo state_dict del modello
        img_path (str): Percorso dell'immagine da elaborare
        device (torch.device): Dispositivo su cui effettuare le operazioni
        imgsz (int): Dimensione dell'immagine di input (default: 416)
        score_threshold (float): Soglia minima di punteggio per considerare una predizione valida (default: 0.25)

    Returns:
        None
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    output_dir = "demo/output"
    os.makedirs(output_dir, exist_ok=True)
    output,classes,input_tensor = compute_img(model_path=model_path,device=device, img=img_path,imgsz=imgsz)
    results_for_image = output[0]
    
    # Filtro predizioni con soglia
    keep = results_for_image['scores'] >= score_threshold
    pred_boxes = results_for_image['boxes'][keep].cpu()
    pred_labels = results_for_image['labels'][keep].cpu()
    scores = results_for_image['scores'][keep].cpu()

    # Disegna box predetti
    pred_img = draw_bounding_boxes((input_tensor * 255).byte(), pred_boxes, labels=[f"{classes[l.item()]}: {s.item():.2f}" for l,s in zip(pred_labels,scores)],
                                    colors=["#{:06x}".format(random.randint(0, 0xFFFFFF)) for l in pred_labels], width=2,font_size=20,font="arial")

    #nome file immagine
    file_name_with_ext = os.path.basename(img_path)
    name, ext = os.path.splitext(file_name_with_ext)
    output_name= name + "_pred" + ext

    # Salva immagine
    img_pil = to_pil_image(pred_img)
    img_pil.save(os.path.join(output_dir, output_name))

    print(f"\nImmagine predetta salvata correttamente in: {os.path.join(output_dir, output_name)}")

    