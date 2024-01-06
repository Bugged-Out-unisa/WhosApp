import os
from tqdm import tqdm

class rawDataReader():
    def __init__(self):
        # Path della cartella delle chat
        # dove verranno analizzati in automatico tutti i file al suo interno
        self.DATA_PATH = "../rawdata"
        self.__read_all_files()
    
    def __init__(self, dataPath):
        self.DATA_PATH = dataPath
        self.__read_all_files()

    def __read_all_files(self):
        '''Restituisce una lista di stringhe contenente il contenuto di tutti i file nella cartella DATA_PATH.'''
        rawdata = []
        
        # Ottieni lista dei file nella cartella
        files = [f for f in os.listdir(self.DATA_PATH) if os.path.isfile(os.path.join(self.DATA_PATH, f))]

        # Concatena contenuto di ogni file nella cartella
        for file_name in tqdm(files):
            f = open(os.path.join(self.DATA_PATH, file_name), 'r', encoding='utf-8')
            rawdata.append(f.read())
            f.close()

        return rawdata