import os
import logging
from tqdm import tqdm


class rawDataReader:
    def __init__(self, data_path="../data/rawdata"):
        self.DATA_PATH = data_path

    def read_all_files(self):
        """Restituisce una lista di stringhe contenente il contenuto di tutti i file nella cartella DATA_PATH."""
        rawdata = []

        # Ottieni lista dei file nella cartella
        files = [f for f in os.listdir(self.DATA_PATH) if os.path.isfile(os.path.join(self.DATA_PATH, f))]

        if len(files) < 1:
            raise Exception("Nessun file di testo trovato per l'estrazione dati.")

        # Concatena contenuto di ogni file nella cartella
        for file_name in tqdm(files):
            f = open(os.path.join(self.DATA_PATH, file_name), 'r', encoding='utf-8')
            rawdata.append(f.read())
            f.close()

        # LOGGING:: Stampa i file utilizzati per l'estrazione dati
        logging.info("File grezzi usati: \n" + "\n".join(f"\t{file}" for file in files))

        return rawdata
