# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 19:59:02 2025

@author: loren
"""


import streamlit as st
from PIL import Image
import numpy as np
import os, cv2, time, datetime, torch
from ultralytics import YOLO
from models import load_custom_fasterrcnn_model
from inference import compute_img, predict_video_stream_frcnn
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import random

# Caricamento del modello YOLO
model = YOLO("./models/yolo12_best.pt")

# Caricamento del modello Faster R-CNN personalizzato
fasterrcnn_model, class_names = load_custom_fasterrcnn_model(
    model_path="./models/fasterrcnn_voc_best.pth", imgsz=416
)

# Selezione modalità operativa: immagine o video
mode = st.sidebar.selectbox("Seleziona modalità", ["Immagine", "Video"])

# Selezione del modello da usare per l'inferenza
model_type = st.sidebar.radio(
    "Seleziona modello",
    options=["YOLO", "Faster R-CNN"],
    index=0,
    help="Scegli il modello da usare per l'inferenza."
)

# Titolo principale della WebApp
st.title("Traffics Signs Detection WebApp")

# Selezione dimensione immagine per l'inferenza
imgsz = st.selectbox(
    "Seleziona dimensione imgsz per l'inferenza",
    [416, 512, 640, 768, 1024, 1280, 1600, 2048],
    index=2
)

# Slider per impostare la soglia di confidenza
conf = st.slider(
    "Soglia di confidenza (conf)",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# Stato iniziale della sessione (per gestire l'inferenza video)
if "processing" not in st.session_state:
    st.session_state.processing = False
if "stop" not in st.session_state:
    st.session_state.stop = False

# === IMMAGINE ===
if mode == "Immagine":
    uploaded_file = st.file_uploader("Carica un'immagine", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        original_height, original_width, _ = image_np.shape
        st.info(f"Dimensione immagine: {original_width}x{original_height} - imgsz={imgsz} - conf={conf}")

        start_time = time.time()

        if model_type == "YOLO":
            # Inferenza con YOLO
            results = model.predict(temp_image_path, imgsz=imgsz, conf=conf)
            res_plotted = results[0].plot()
            inference_time = time.time() - start_time
        else:
            # Inferenza con Faster R-CNN
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            output, classes, input_tensor = compute_img(
                model_path="./models/fasterrcnn_voc_best.pth",
                img=temp_image_path,
                device=device,
                imgsz=imgsz
            )
            pred = output[0]
            keep = pred['scores'] >= conf
            boxes = pred['boxes'][keep].cpu()
            labels = pred['labels'][keep].cpu()
            scores = pred['scores'][keep].cpu()
            
            drawn = draw_bounding_boxes(
                (input_tensor * 255).byte().cpu(),
                 boxes,
                 labels=[f"{classes[l.item()]}: {s.item():.2f}" for l, s in zip(labels, scores)],
                 colors=["#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in labels],
                 width=2,
                 font_size=20
)
            
        res_plotted = to_pil_image(drawn)
        inference_time = time.time() - start_time

        # Visualizza immagine originale e risultato
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_np, caption="Immagine Originale", use_container_width=True)
        with col2:
            if model_type == "YOLO":
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st.image(res_plotted_rgb, caption="Risultato YOLO", use_container_width=True)
            else:
                st.image(res_plotted, caption="Risultato Faster R-CNN", use_container_width=True)

        st.success(f"Inferenza completata in {inference_time:.2f} secondi.")

# === VIDEO ===
elif mode == "Video":
    uploaded_video = st.file_uploader("Carica un video", type=["mp4"])

    if uploaded_video is not None:
        temp_input_path = "temp_video.mp4"
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_video.read())

        st.info("Video caricato. Pronto per l'inferenza.")

        # Stato elaborazione video
        if "stop_inferenza" not in st.session_state:
            st.session_state.stop_inferenza = False
        if "elaborazione_in_corso" not in st.session_state:
            st.session_state.elaborazione_in_corso = False
        if "elaborazione_interrotta" not in st.session_state:
            st.session_state.elaborazione_interrotta = False

        col1, col2 = st.columns(2)

        # Pulsanti di controllo elaborazione
        if col1.button("Avvia elaborazione"):
            st.session_state.stop_inferenza = False
            st.session_state.elaborazione_in_corso = True
            st.rerun()

        if col2.button("Interrompi elaborazione", disabled=not st.session_state.elaborazione_in_corso):
            st.session_state.stop_inferenza = True
            st.session_state.elaborazione_in_corso = False
            st.session_state.elaborazione_interrotta = True
            st.rerun()

        elif st.session_state.elaborazione_interrotta:
            st.warning("⚠️ L'elaborazione è stata interrotta. I frame elaborati e il video parziale sono stati salvati rispettivamente nelle cartelle frames e videos.")
            st.session_state.elaborazione_interrotta = False

        # Esecuzione inferenza
        if st.session_state.elaborazione_in_corso:
            progress_bar = st.progress(0)
            time_remaining_placeholder = st.empty()
            log_output = st.empty()
            stframe = st.empty()
            status_message = st.empty()
            status_message.info(f"Inferenza {model_type} in corso...")

            # Estrai info video
            cap = cv2.VideoCapture(temp_input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Prepara nomi file e cartelle per salvataggio
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = os.path.splitext(os.path.basename(temp_input_path))[0]
            output_video_path = os.path.join("videos", f"{base_filename}_{timestamp}.mp4")
            output_frames_subdir = os.path.join("frames", f"{base_filename}_{timestamp}")
            os.makedirs("videos", exist_ok=True)
            os.makedirs(output_frames_subdir, exist_ok=True)

            # === YOLO: inferenza frame per frame ===
            if model_type == "YOLO":
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                # Calcolo del numero totale di frame
                total_frames = int(cv2.VideoCapture(temp_input_path).get(cv2.CAP_PROP_FRAME_COUNT))

                results = model.predict(
                    source=temp_input_path,
                    imgsz=imgsz,
                    conf=conf,
                    stream=True
                )

                current_frame = 0
                start_time = time.time()
                for result in results:
                    if st.session_state.stop_inferenza:
                        log_output.warning("🛑 Elaborazione interrotta.")
                        break
                    
                    current_frame+=1

                    plotted = result.plot()
                    frame_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, caption=f"Frame {current_frame + 1}", channels="RGB", use_container_width=True)

                    frame_path = os.path.join(output_frames_subdir, f"frame_{current_frame:04d}.jpg")
                    cv2.imwrite(frame_path, plotted)
                    video_writer.write(plotted)

                    #current_frame += 1
                    progress = min(current_frame / total_frames, 1.0)
                    progress_bar.progress(progress)
                    elapsed = time.time() - start_time
                    eta = (elapsed / (current_frame + 1)) * (total_frames - (current_frame + 1))
                    eta_formatted = str(datetime.timedelta(seconds=int(eta)))
                    time_remaining_placeholder.caption(f"⏱️ Tempo stimato rimanente: {eta_formatted}")
                    log_output.code(f"Frame elaborati: {current_frame + 1} / {total_frames}", language="log")

                video_writer.release()

            # === Faster R-CNN: inferenza con funzione custom ===
            else:
                predict_video_stream_frcnn(
                    model_path="./models/fasterrcnn_voc_best.pth",
                    video_path=temp_input_path,
                    output_video_path=output_video_path,
                    output_frames_subdir=output_frames_subdir,
                    imgsz=imgsz,
                    score_threshold=conf,
                    progress_bar=progress_bar,
                    time_remaining_placeholder=time_remaining_placeholder,
                    stframe=stframe,
                    log_output=log_output,
                    stop_flag=st.session_state.stop_inferenza
                )

            # Conclusione elaborazione
            st.session_state.elaborazione_in_corso = False
            progress_bar.progress(1.0)
            time_remaining_placeholder.caption("✅ Elaborazione completata.")
            status_message.success("✅ Inferenza completata con successo.")

            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)

            st.markdown("---")
            st.subheader("Scarica il video elaborato")
            if os.path.exists(output_video_path):
                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="Scarica video con predizioni",
                        data=f,
                        file_name=os.path.basename(output_video_path),
                        mime="video/mp4"
                    )
            else:
                st.warning("⚠️ Il file video non è stato generato correttamente.")

