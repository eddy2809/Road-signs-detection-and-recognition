# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 20:00:55 2025

@author: loren
"""

import os, cv2, time, datetime, random
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import draw_bounding_boxes
from models import load_custom_fasterrcnn_model

def compute_img(model_path, img, device, imgsz):
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
    model, classes = load_custom_fasterrcnn_model(model_path, imgsz)
    image = Image.open(img).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((imgsz))
    ])
    input_tensor = transform(image).to(device)
    with torch.no_grad():
        output = model([input_tensor])
    return output, classes, input_tensor


def predict_video_stream_frcnn(
    model_path, video_path, output_video_path, output_frames_subdir,
    imgsz, score_threshold,
    progress_bar=None, time_remaining_placeholder=None,
    stframe=None, log_output=None, stop_flag=False
):   
    """
   Esegue l'inferenza video frame per frame utilizzando un modello Faster R-CNN.
   Per ogni frame:
     - effettua la predizione degli oggetti,
     - disegna i bounding box e le etichette con le classi,
     - salva il frame annotato su disco e nel video di output,
     - aggiorna eventualmente una barra di avanzamento e il log su Streamlit.
   Supporta l'interruzione anticipata tramite il flag `stop_flag`.
   """
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_custom_fasterrcnn_model(model_path, imgsz)

    class_colors = {
        i: "#{:06x}".format(random.randint(0, 0xFFFFFF))
        for i, c in enumerate(classes) if c != '__background__'
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((imgsz, imgsz))
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Errore apertura video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (imgsz, imgsz))
    os.makedirs(output_frames_subdir, exist_ok=True)

    current_frame = 0
    start_time = time.time()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret or stop_flag:
                break
            
            current_frame += 1

            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(image_pil).to(device)
            output = model([input_tensor])[0]

            keep = output['scores'] >= score_threshold
            boxes = output['boxes'][keep].cpu()
            labels = output['labels'][keep].cpu()
            scores = output['scores'][keep].cpu()

            colors = [class_colors.get(l.item(), "#00FF00") for l in labels]
            pred_img = draw_bounding_boxes(
                (input_tensor * 255).byte().cpu(), boxes,
                labels=[f"{classes[l.item()]}: {s.item():.2f}" for l, s in zip(labels, scores)],
                colors=colors, width=2, font_size=20
            )

            frame_np = pred_img.permute(1, 2, 0).cpu().numpy()
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

            frame_path = os.path.join(output_frames_subdir, f"frame_{current_frame:04d}.jpg")
            cv2.imwrite(frame_path, frame_bgr)

            if stframe:
                stframe.image(frame_np, caption=f"Frame {current_frame + 1}", channels="RGB", use_container_width=True)

            if progress_bar:
                progress = min((current_frame + 1) / frame_count, 1.0)
                progress_bar.progress(progress)
                elapsed = time.time() - start_time
                eta = (elapsed / (current_frame + 1)) * (frame_count - (current_frame + 1))
                time_remaining_placeholder.caption(
                    f"⏱️ Tempo stimato rimanente: {str(datetime.timedelta(seconds=int(eta)))}"
                )

            if log_output:
                log_output.code(f"Frame elaborati: {current_frame + 1} / {frame_count}", language="log")

            #current_frame += 1

    cap.release()
    out.release()

