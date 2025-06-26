# Road signs detection and recognition

## Descrizione
Il seguente repository è relativo al progetto dell'insegnamento [Machine Learning](https://www.dmi.unict.it/farinella/ML/), tenuto presso il Dipartimento di Matematica e Informatica dell'Università di Catania.  Il team degli sviluppatori del progetto è composto da Lorenzo La Rocca, Raffaele Terracino, Edoardo Tantari.
Il problema che si pone per lo sviluppo del progetto  è quello di costruire e confrontare diversi modelli di **object detection** che siano in grado di rilevare e classificare segnali stradali. Nel contesto del progetto è stato scelto un sottoinsieme di 19 classi tra tutti i possibili segnali stradali europei.  

I segnali stradali sono i seguenti:  
* segnale di direzione obbligatoria a sinistra;
* segnale di direzione obbligatoria a destra;
* segnale di dare precedenza;
* segnale di divieto di accesso;
* segnale di stop;
* semaforo di colore verde;
* semaforo di colore rosso;
* segnale di limite di velocità da 20 a 120 km/h;  

Tra le architetture one-stage allo stato dell'arte, è stato scelto di utilizzare [YOLOV12](https://docs.ultralytics.com/models/yolo12/), mentre tra quelle two-stage è stato scelto [Faster_RCNN_ResNet50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html).

## Requisiti
Per poter procedere all'esecuzione dei diversi notebook presenti nel repository e della demo, è necessario prima installarne le relative dipendenze come segue:  
```
#Creare un ambiente virtuale ed eseguire il comado successivo
pip install -r requirements.txt
```
## Esecuzione
I notebook jupyter relativi alle fasi di **training**, **evaluation** e **testing** dei diversi modelli sono contenuti nella sezione `notebook`.  


# Demo

Bisogna assicurarsi di usare la più recente versione di **Streamlit** (attualmente `1.46.0`).

## Esecuzione

Per eseguire la webapp:

1. Spostarsi nella cartella`demo`.
2. Eseguire il seguente comando da terminale:

```bash
streamlit run main.py
```

## Note

Il primo avvio della webapp sarà più lento rispetto ai successivi per via del download da [HuggingFace](https://huggingface.co/) del modello FasterRCNN di 315 MB.

## Struttura del progetto

- `demo/` – Webapp Streamlit per la dimostrazione del progetto su immagini o video
- `docs/` – Relazione tecnica sul progetto
- `models/` – Checkpoint e metriche dei modelli YOLO e Faster R-CNN addestrati
- `scripts/` – Script ausiliari per la costruzione del dataset
- `src/` – Codice sorgente principale per il training e l'inferenza dei modelli.
- `.gitignore` – File per escludere elementi non tracciati dal versionamento Git
- `README.md` – Descrizione del progetto, istruzioni e note tecniche
- `requirements.txt` – Elenco delle dipendenze Python necessarie all'esecuzione
