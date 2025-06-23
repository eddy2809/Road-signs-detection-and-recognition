## Requisiti
Bisogna assicurarsi di usare la più recente versione di Streamlit (1.46.0)

## Esecuzione

Per eseguire la webapp:

1. Spostarsi nella cartella contenente il file `webapp.py`.
2. Eseguire i seguenti comandi da terminale:

pip install -r requirements.txt
streamlit run webapp.py --server.enableXsrfProtection=false

## Note aggiuntive
È importante includere il flag --server.enableXsrfProtection=false per disabilitare la protezione XSRF, altrimenti durante il caricamento dei file si verificherebbe un errore AxiosError: Request failed with status code 403.