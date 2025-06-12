Per eseguire la webapp, è necessario spostarsi nella cartella contenente il file webapp.py ed eseguire i seguenti comandi da terminale:
pip install -r requirements.txt
streamlit run webapp.py --server.enableXsrfProtection=false
È importante includere il flag --server.enableXsrfProtection=false per disabilitare la protezione XSRF,
altrimenti durante il caricamento dei file si verificherebbe un errore AxiosError: Request failed with status code 403.