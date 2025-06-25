## Requisiti

Bisogna assicurarsi di usare la più recente versione di **Streamlit** (attualmente `1.46.0`).

## Esecuzione

Per eseguire la webapp:

1. Spostarsi nella cartella contenente il file `main.py`.
2. Eseguire i seguenti comandi da terminale:

```bash
pip install -r requirements.txt
streamlit run main.py

## Note

Il primo avvio della webapp sarà più lento rispetto ai successivi per via del download da HuggingFace del modello FasterRCNN di 315 MB.
