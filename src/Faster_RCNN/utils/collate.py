def collate_fn(batch):
    """
    Una funzione di collazione da utilizzare quando si crea un torch.utils.data.DataLoader 
    da un set di dati. Questa funzione accetta un batch di punti dati e restituisce 
    una tupla della forma (input, target) dove input è una tupla dei 
    punti dati di input e target è una tupla dei target corrispondenti. 

    Argomenti:
        batch (elenco): un elenco di tuple, dove ogni tupla è della forma (input, target)

    Restituisce:
        tupla: una tupla della forma (input, target)

    """
    return tuple(zip(*batch))