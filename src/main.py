import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from utility.extractChat import ExtractChat
from utility.dataFrameProcess import DataFrameProcessor
from utility.model_list import models
from utility.modelTraining import ModelTraining
from utility.featureConstruction import featureConstruction
#from feel_it import EmotionClassifier, SentimentClassifier


# Path della cartella delle chat
# dove verranno analizzati in automatico tutti i file al suo interno
DATA_PATH = "../rawdata"


def read_all_files():
    '''Restituisce una lista di stringhe contenente il contenuto di tutti i file nella cartella DATA_PATH.'''
    rawdata = []
    
    # Ottieni lista dei file nella cartella
    files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]

    # Concatena contenuto di ogni file nella cartella
    for file_name in tqdm(files):
        f = open(os.path.join(DATA_PATH, file_name), 'r', encoding='utf-8')
        rawdata.append(f.read())
        f.close()

    return rawdata

if __name__ == "__main__":
    try:
        modelName = sys.argv[1]
        model = models[modelName]
    except:
        print("Modello specificato non trovato...")
        print("-- Utilizzato Random Forest --")
        model = models["random_forest"]

    print("\n[LOADING] Leggendo le chat dai file grezzi...")
    rawdata = read_all_files()

    print("\n[LOADING] Estraendo informazioni dai dati grezzi...")
    dates, users, messages = ExtractChat(rawdata).extract()

    print("\n[LOADING] Creando il dataframe e applicando data cleaning e undersampling...")
    df = DataFrameProcessor(dates, users, messages).get_dataframe()

    print("\n[LOADING] Applicando feature construction...")
    featureConstruction(df)

    if isinstance(model, list):
        for i,m in enumerate(model, 1):
            print("\n[LOADING] Addestrando il modello... #{}".format(i))
            ModelTraining(m, df)

    else:
        print("\n[LOADING] Addestrando il modello...")
        ModelTraining(model, df)