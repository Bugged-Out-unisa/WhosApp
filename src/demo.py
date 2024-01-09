import pandas as pd
import numpy as np
import skops.io as skio
from modelTraining import ModelTraining
from utility.featureConstruction import featureConstruction

class modelExecution:
    """
        Classe che si occupa di eseguire un modello allenato e, 
        per ogni utente, di predire con quanta probabilità abbia scritto il messaggio in input
        sia per quel singolo messaggio che per l'intera conversazione.
    """

    MODEL_PATH = "../models/"

    def __init__(self, model = None):
        fileModel = self.MODEL_PATH + model

        self.__trainedModel = skio.load(fileModel)
    
    def dataframe_for_messages(self, message):
        df = pd.DataFrame({"message": message})
            
        return featureConstruction(dataFrame= df, datasetPath="./", saveDataFrame=False).get_dataframe()

    def __predict__(self):
        """
            Prevedi iterativamente l'autore del messaggio inserito in input.
            Assegna delle probabilità anche in base allo storico di tutti i messaggi inseriti.
        """

        # Per ogni utente da prevedere, crea una lista di previsioni
        num_users = self.__trainedModel.n_classes_
        predictions = [[] for _ in range(num_users)]

        while True:
            
            # Ottieni il messaggio da input
            message = input("\nScrivi un messaggio:\n")

            # Crea il dataframe e costruisci le feature su quel messaggio
            df = self.dataframe_for_messages([message])

            # Ottieni probabilità per ogni utente
            # [0] perché resituisce una lista di previsioni (come se si aspettasse più messaggi)
            users_prob = self.__trainedModel.predict_proba(df)[0]

            # Per ogni utente, ottieni probabilità per il messaggio inserito e salva in lista
            for i in range(num_users):
                predictions[i].append(users_prob[i])

            # Stampa report
            # Solo per l'ultima previsione [-1]
            print("\n---SINGOLO---")
            print("\n".join([f"USER {i}: {predictions[i][-1]:.2f}" for i in range(num_users)]))
            
            # Media delle previsioni
            print("\n----MEDIA----")
            print("\n".join([f"USER {i}: {np.average(predictions[i]):.2f}" for i in range(num_users)]))


if __name__ == "__main__":
    print("[INFO] Caricamento del modello addestrato...")
    modelExecution("model.skops").__predict__()