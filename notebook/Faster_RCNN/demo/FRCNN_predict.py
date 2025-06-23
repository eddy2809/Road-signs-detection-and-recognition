import sys
import os
import torch
import random
import cv2
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
        imgsz (int): Dimensione dell'immagine di input (default: 416)
        score_threshold (float): Soglia minima di punteggio per considerare una predizione valida (default: 0.25)

    Returns:
        None
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    output_dir = "demo/output"
    os.makedirs(output_dir, exist_ok=True)
    output,classes,input_tensor = compute_img(model_path=model_path,device=device, img=img_path,imgsz=imgsz)
    results__for_image_for_image = output[0]
    
    # Filtro predizioni con soglia
    keep = results__for_image_for_image['scores'] >= score_threshold
    pred_boxes = results__for_image_for_image['boxes'][keep].cpu()
    pred_labels = results__for_image_for_image['labels'][keep].cpu()
    scores = results__for_image_for_image['scores'][keep].cpu()

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

def predict_mp4_video(model_path,video_path,imgsz=416,score_threshold=0.25):
    """
    Genera un video con le predizioni del modello su un video in formato mp4 frame per frame.

    Args:
        model_path (str): Percorso del file .pth contenente lo state_dict del modello
        imgsz (int): Dimensione dell'immagine di input (default: 416)
        score_threshold (float): Soglia minima di punteggio per considerare una predizione valida (default: 0.25)

    Returns:
        None
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #percorso output
    output_path = "demo/output/"
    file_name_with_ext = os.path.basename(video_path)
    name, ext = os.path.splitext(file_name_with_ext)
    output_video_name= name + "_pred" + ext
    output_path_video = output_path + output_video_name  

    #caricamento modello
    try:
        model,classes = load_custom_fasterrcnn_model(model_path=model_path,imgsz=imgsz)
        print(f"Modello caricato con {len(classes)} classi: {classes}")
    except FileNotFoundError:
        print(f"Errore: File modello non trovato a {model_path}")
        print("Assicurati di aver specificato il percorso corretto al tuo modello addestrato.")


    class_colors = {}
    # Itera attraverso le classi per assegnare un colore randomico statico
    # Salta la classe '__background__'
    for i, class_name in enumerate(classes):
        if class_name == '__background__':
            continue # Non assegnare un colore per lo sfondo o non disegnarlo
        class_colors[i] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    print(f"Colori statici generati per le classi: {class_colors}")


    # Trasformazioni per i frame
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((imgsz, imgsz))
    ])

    #caricamento video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il video a {video_path}")
        exit()

    # Ottieni le proprietà del video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video caricato: {video_path} ({frame_width}x{frame_height} @ {fps} FPS)")

    
    # Codec video writer (FourCC) - 'mp4v' per .mp4, 'XVID' per .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path_video, fourcc, fps, (imgsz, imgsz))

    if not out.isOpened():
        print(f"Errore: Impossibile aprire il writer video a {output_path_video}")
        exit()


    #per ogni frame
    frame_count = 0
    with torch.no_grad(): # Nessun calcolo del gradiente in inferenza per risparmiare memoria
        while True:
            ret, frame = cap.read() # Leggi un frame

            if not ret:
                break # Fine del video

            frame_count += 1
            print(f"Processing frame {frame_count}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}...")
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(image_pil)
            input_tensor = input_tensor.to(device) 
            output = model([input_tensor])
            results_for_image = output[0]

            # Filtra i risultati in base alla soglia di confidenza
            keep = results_for_image['scores'] >= score_threshold
            pred_boxes = results_for_image['boxes'][keep].cpu() 
            pred_labels = results_for_image['labels'][keep].cpu()
            scores = results_for_image['scores'][keep].cpu()


            # Crea la lista di colori per i bounding box attuali
            current_bbox_colors = [class_colors[l.item()] for l in pred_labels]

            # Disegna box predetti
            pred_img = draw_bounding_boxes((input_tensor * 255).byte(), pred_boxes, labels=[f"{classes[l.item()]}: {s.item():.2f}" for l,s in zip(pred_labels,scores)],
                                    colors=current_bbox_colors, width=2,font_size=20,font="arial")

            # Converti il tensore PyTorch (C, H, W) in NumPy array (H, W, C) e poi da RGB a BGR per OpenCV
            drawn_frame_np = pred_img.cpu().permute(1, 2, 0).contiguous().numpy()
            drawn_frame_bgr = cv2.cvtColor(drawn_frame_np, cv2.COLOR_RGB2BGR)

            # Scrivi il frame elaborato nel video di output
            out.write(drawn_frame_bgr)

            # (Opzionale) Mostra il frame in tempo reale
            cv2.imshow('Faster R-CNN Inference', drawn_frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Premi 'q' per uscire
                break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Elaborazione video completata. Output salvato in: {output_path_video}")
        
            
        

    