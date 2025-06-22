# Importiamo le librerie necessarie
import streamlit as st                    
from PIL import Image                    
import numpy as np                       
from ultralytics import YOLO             
import time, datetime, os                
import cv2                               

# Carica il modello YOLO pre-addestrato
model = YOLO("./yolo12_no_harmful_aug_640/weights/best.pt")

# Sidebar: selezione modalità (immagine o video)
mode = st.sidebar.selectbox("Seleziona modalità", ["Immagine", "Video"])

# Titolo della WebApp
st.title("YOLO Object Detection WebApp")

# Selezione dimensione immagini per il resize prima dell'inferenza
imgsz = st.selectbox(
    "Seleziona dimensione imgsz per l'inferenza",
    ["Nessun ridimensionamento", 416, 512, 640, 768, 1024, 1280, 1600, 2048],
    index=3,
    help=(
        "imgsz è la dimensione a cui l'immagine o il frame video verrà ridimensionato prima dell'inferenza. "
        "YOLO richiede input quadrati. Valori più alti migliorano il rilevamento ma aumentano il tempo di elaborazione."
    )
)

# Slider per impostare la soglia di confidenza
conf = st.slider(
    "Soglia di confidenza (conf)",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help=(
        "conf è la soglia di confidenza minima per visualizzare una predizione. "
        "Valori alti filtrano i risultati meno certi."
    )
)

# Inizializza lo stato persistente dell'app
if "processing" not in st.session_state:
    st.session_state.processing = False
if "stop" not in st.session_state:
    st.session_state.stop = False

# === MODALITÀ IMMAGINE ===
if mode == "Immagine":
    # Caricamento file immagine
    uploaded_file = st.file_uploader("Carica un'immagine", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Salva immagine temporaneamente
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
           f.write(uploaded_file.getvalue())

        # Apre l'immagine e la converte in RGB
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)  # Converte in array NumPy

        # Ottiene dimensioni immagine originali
        original_height, original_width, _ = image_np.shape
        st.info(f"Dimensione immagine: {original_width}x{original_height} - Inferenza con imgsz={imgsz}x{imgsz} - conf={conf}") 

        # Esegue l'inferenza e misura il tempo
        start_time = time.time()
        results = model.predict(temp_image_path, imgsz=imgsz, conf=conf)
        inference_time = time.time() - start_time

        # Disegna i bounding box sulle predizioni
        res_plotted = results[0].plot()

        # Mostra immagini prima e dopo il rilevamento
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_np, caption="Immagine Originale", use_column_width=True)
        with col2:
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_plotted_rgb, caption="Risultato YOLO", use_column_width=True)

        # Visualizza il tempo di inferenza
        st.success(f"Inferenza completata in {inference_time:.2f} secondi.")

# === MODALITÀ VIDEO ===
elif mode == "Video":
    # Caricamento file video
    uploaded_video = st.file_uploader("Carica un video", type=["mp4"])

    if uploaded_video is not None:
        # Salva video temporaneamente
        temp_input_path = "temp_video.mp4"
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_video.read())

        st.info("Video caricato. Pronto per l'inferenza.")

        # Inizializza le variabili di stato se non esistono
        if "stop_inferenza" not in st.session_state:
            st.session_state.stop_inferenza = False
        if "elaborazione_in_corso" not in st.session_state:
            st.session_state.elaborazione_in_corso = False
        if "elaborazione_interrotta" not in st.session_state:
            st.session_state.elaborazione_interrotta = False

        # Colonne per pulsanti "Avvia" e "Interrompi"
        col1, col2 = st.columns(2)

        # Pulsante per avviare l'elaborazione
        if col1.button("Avvia elaborazione"):
            st.session_state.stop_inferenza = False
            st.session_state.elaborazione_in_corso = True
            st.experimental_rerun()

        # Pulsante per interrompere l'elaborazione (attivo solo se in corso)
        if col2.button("Interrompi elaborazione", disabled=not st.session_state.elaborazione_in_corso):
            st.session_state.stop_inferenza = True
            st.session_state.elaborazione_in_corso = False
            st.session_state.elaborazione_interrotta = True
            st.experimental_rerun()

        # Messaggio se l'elaborazione è stata interrotta
        elif st.session_state.elaborazione_interrotta:
            st.warning("⚠️ L'elaborazione è stata interrotta. I frame elaborati e il video parziale sono stati salvati rispettivamente nelle cartelle frames e videos.")
            st.session_state.elaborazione_interrotta = False

        # === Inferenza frame per frame ===
        if st.session_state.elaborazione_in_corso:
            # UI dinamica
            progress_bar = st.progress(0)
            time_remaining_placeholder = st.empty()
            log_output = st.empty()
            stframe = st.empty()
            download_placeholder = st.empty()
            status_message = st.empty()
            status_message.info("Inferenza YOLO in corso sul video...")

            # Caricamento info video
            cap = cv2.VideoCapture(temp_input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Crea nomi per cartelle di output (video e frame)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = os.path.splitext(os.path.basename(temp_input_path))[0]
            output_video_path = os.path.join("videos", f"{base_filename}_{timestamp}.mp4")
            output_frames_subdir = os.path.join("frames", f"{base_filename}_{timestamp}")
            os.makedirs("videos", exist_ok=True)
            os.makedirs(output_frames_subdir, exist_ok=True)

            # Inizializza salvataggio video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            # Inizio inferenza
            start_time = time.time()
            results = model.predict(
                source=temp_input_path,
                imgsz=imgsz,
                conf=conf,
                stream=True  # Modalità streaming per processare frame per frame
            )

            # Loop su ogni frame
            current_frame = 0
            for result in results:
                # Se è stato cliccato "interrompi"
                if st.session_state.stop_inferenza:
                    log_output.warning("🛑 Elaborazione interrotta.")
                    break

                # Disegna bounding box e mostra in tempo reale
                plotted = result.plot()
                frame_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, caption=f"Frame {current_frame + 1}", channels="RGB", use_column_width=True)

                # Salva il frame come immagine
                frame_path = os.path.join(output_frames_subdir, f"frame_{current_frame:04d}.jpg")
                cv2.imwrite(frame_path, plotted)

                # Scrive frame su file video
                video_writer.write(plotted)

                # Aggiorna barra di avanzamento e tempo stimato
                current_frame += 1
                progress = min(current_frame / total_frames, 1.0)
                progress_bar.progress(progress)
                elapsed = time.time() - start_time
                eta = (elapsed / current_frame) * (total_frames - current_frame) if current_frame > 0 else 0
                eta_formatted = str(datetime.timedelta(seconds=int(eta)))
                time_remaining_placeholder.caption(f"⏱️ Tempo stimato rimanente: {eta_formatted}")
                log_output.code(f"Frame {current_frame} elaborato", language="log")

            # Termina e chiude il video
            video_writer.release()
            st.session_state.elaborazione_in_corso = False

            # Messaggi finali
            progress_bar.progress(1.0)
            time_remaining_placeholder.caption("✅ Elaborazione completata.")
            status_message.success("✅ Inferenza completata con successo.")

            # Rimuove il file video temporaneo
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)

            # Download video
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

















           






































