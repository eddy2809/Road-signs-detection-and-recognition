import streamlit as st                   
from PIL import Image                   
import numpy as np                      
from ultralytics import YOLO            
import time                             
import cv2                              
import os                               

# Carica il modello YOLO dal percorso specificato
model = YOLO("./yolo12_no_harmful_aug_640/weights/best.pt")

# Sidebar per scegliere tra elaborazione immagine o video
mode = st.sidebar.selectbox("Seleziona modalità", ["Immagine", "Video"])

# Titolo della WebApp
st.title("YOLO Object Detection WebApp")

# Scelta della dimensione imgsz usata per il resize prima dell’inferenza
imgsz = st.selectbox(
    "Seleziona dimensione imgsz per l'inferenza",
    [416, 512, 640, 768, 1024, 1280, 1600, 2048],
    index=2,
    help=(
        "imgsz è la dimensione a cui l'immagine o il frame video verrà ridimensionato prima dell'inferenza. "
        "YOLO richiede input quadrati (es. imgsz=640 implica un ridimensionamento a 640×640 pixel). "
        "Valori più alti possono migliorare la precisione nel rilevamento di oggetti piccoli, "
        "ma aumentano il tempo di elaborazione e il consumo di memoria. "
        "Valori più bassi offrono inferenze più veloci ma possono ridurre la qualità del rilevamento."
    )
)

# Slider per selezionare la soglia di confidenza per le predizioni
conf = st.slider(
    "Soglia di confidenza (conf)",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help=(
        "conf è la soglia di confidenza minima per mantenere una predizione. "
        "Ogni rilevamento è associato a un punteggio (confidenza) tra 0 e 1 che indica quanto il modello è sicuro. "
        "Aumentare conf riduce i falsi positivi ma può nascondere oggetti rilevati con bassa certezza. "
        "Valori più bassi mostrano più risultati ma includono anche quelli più incerti."
    )
)

# Inizializzazione dello stato dell'applicazione
if "processing" not in st.session_state:
    st.session_state.processing = False
if "stop" not in st.session_state:
    st.session_state.stop = False

# Modalità immagine
if mode == "Immagine":
    uploaded_file = st.file_uploader("Carica un'immagine", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")  # Apre l'immagine e la converte in RGB
        image_np = np.array(image)                        # Converte l'immagine in array NumPy

        # Ottieni dimensioni originali
        original_height, original_width, _ = image_np.shape
        st.info(f"Dimensione immagine: {original_width}x{original_height} - Inferenza con imgsz={imgsz}x{imgsz} - conf={conf}")

        # Esegui inferenza e misura il tempo
        start_time = time.time()
        results = model.predict(image_np, imgsz=imgsz, conf=conf)
        inference_time = time.time() - start_time

        # Disegna le predizioni
        res_plotted = results[0].plot()

        # Mostra immagine originale e immagine con bounding box affiancate
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_np, caption="Immagine Originale", use_column_width=True)
        with col2:
            st.image(res_plotted, caption="Risultato YOLO", use_column_width=True)

        # Mostra tempo di inferenza
        st.success(f"Inferenza completata in {inference_time:.2f} secondi.")

# Modalità video
elif mode == "Video":
    uploaded_video = st.file_uploader("Carica un video", type=["mp4"])

    if uploaded_video is not None:
        temp_input_path = "temp_video.mp4"      # Percorso temporaneo per il video caricato
        temp_output_path = "output_video.mp4"   # Percorso per il video di output finale
        frames_dir = "frames"                   # Cartella per salvare i frame elaborati
        os.makedirs(frames_dir, exist_ok=True)  # Crea la cartella se non esiste

        # Salva il video caricato su disco
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_video.read())

        # Apre il video con OpenCV
        cap = cv2.VideoCapture(temp_input_path)

        # Ottiene dimensioni e FPS del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Mostra info video all'utente
        st.info(
            f"Risoluzione video: {width}x{height} - FPS: {fps:.2f} "
            f"- Inferenza con imgsz={imgsz}x{imgsz} - conf={conf}"
        )

        stframe = st.empty()  # Placeholder per mostrare frame nella UI
        
        
        # Pulsante per avviare l'elaborazione
        if st.button("Avvia elaborazione"):
            # Pulizia automatica dei frame salvati precedenti
            for file in os.listdir(frames_dir):
                file_path = os.path.join(frames_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            st.session_state.processing = True
            st.session_state.stop = False
            st.session_state.elaborazione_completata = False  # Reset stato finale


        # Pulsante per interrompere manualmente
        if st.button("Interrompi elaborazione"):
            st.session_state.stop = True

        # Se l'elaborazione è attiva
        if st.session_state.get("processing", False):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec MP4
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

            st.info("Elaborazione video frame-by-frame in corso...")
            frame_idx = 0  # Contatore dei frame

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or st.session_state.stop:
                    break  # Fine video o interruzione

                # Converte il frame in RGB per YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Inferenza YOLO sul frame
                start_time = time.time()
                results = model.predict(frame_rgb, imgsz=imgsz, conf=conf)
                inference_time = time.time() - start_time

                # Disegna le predizioni
                res_plotted = results[0].plot()
                res_bgr = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)

                # Salva il frame nel video e come immagine
                out.write(res_bgr)
                frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(frame_filename, res_bgr)
                frame_idx += 1

                # Mostra il frame aggiornato
                stframe.image(
                    res_plotted,
                    caption=f"Inferenza: {inference_time:.2f}s",
                    channels="RGB",
                    use_column_width=True
                )

            # Rilascia risorse
            cap.release()
            out.release()
            st.session_state.processing = False
            st.session_state.elaborazione_completata = True  # Abilita download video

            if st.session_state.stop:
                st.warning("Elaborazione interrotta manualmente. I frame sono stati salvati nella cartella frames.")
            else:
                st.success("Elaborazione completata con successo. I frame sono stati salvati nella cartella frames.")

        # Mostra il bottone per scaricare il video solo dopo stop o completamento
        if st.session_state.get("elaborazione_completata", False):
            st.markdown("---")
            st.subheader("Scarica il video generato dai frame salvati")

            # Funzione per creare video a partire dai frame salvati
            def genera_video_da_frame(output_path, frame_folder, width, height, fps):
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".jpg")])
                for file in frame_files:
                    frame_path = os.path.join(frame_folder, file)
                    frame = cv2.imread(frame_path)
                    out.write(frame)
                out.release()

            # Genera e scarica il video dai frame
            genera_video_da_frame(temp_output_path, frames_dir, width, height, fps)

            with open(temp_output_path, "rb") as f:
                st.download_button(
                    "Scarica video elaborato",
                    f,
                    file_name="output_video.mp4",
                    mime="video/mp4"
                )










