import pandas as pd
import numpy as np
import requests
import json
from threading import Lock
from flask import Flask, jsonify, request
from utility.dataset.featureConstruction import featureConstruction
from utility.cmdlineManagement.trainedModelSelection import TrainedModelSelection

app = Flask(__name__)
lock = Lock()

class modelExecution:
    """
        Classe che si occupa di eseguire un modello allenato e, 
        per ogni utente, di predire con quanta probabilità abbia scritto il messaggio in input
        sia per quel singolo messaggio che per l'intera conversazione.
    """

    MODEL_PATH = "../models/"

    def __init__(self, model = None, scaler = None):
        if model == None:
            selection = TrainedModelSelection()
            
            self.__trainedModel = selection.model
            self.__scaler = selection.scaler
        else:
            self.__trainedModel = model
            self.__scaler = scaler
        
        self.predictions = [[] for _ in range(self.__trainedModel.n_classes_)]

    def dataframe_for_messages(self, message):
        # Applica feature construction al messaggio
        df = featureConstruction(dataFrame=pd.DataFrame({"message": message}),\
                                 datasetPath="./", saveDataFrame=False)\
                                .get_dataframe()
        
        # Applica scaling
        return pd.DataFrame(self.__scaler.transform(df), columns=df.columns)

    def __rest_predict__(self, data):
        if request.method == "POST":

            num_users = self.__trainedModel.n_classes_

            output = {
                "mappedUsers": {},
                "single": [],
                "average": []
            }

            # Ottieni il messaggio da input
            #message = input("\nScrivi un messaggio:\n")

            # Crea il dataframe e costruisci le feature su quel messaggio
            df = self.dataframe_for_messages([data])

            # Ottieni probabilità per ogni utente
            # [0] perché resituisce una lista di previsioni (come se si aspettasse più messaggi)
            users_prob = self.__trainedModel.predict_proba(df)[0]

            # Per ogni utente, ottieni probabilità per il messaggio inserito e salva in lista
            for i in range(num_users):
                self.predictions[i].append(users_prob[i])

            # Stampa report
            # Solo per l'ultima previsione [-1]
            # message = "<b>SINGOLO</b><br>"
            # message += "<br>".join([f"USER {i}: {self.predictions[i][-1]:.2f}" for i in range(num_users)])

            output["single"] = [self.predictions[i][-1] for i in range(num_users)]

            # Media delle previsioni
            # message += "<br><b>MEDIA</b><br>"
            # message += "<br>".join([f"USER {i}: {np.average(self.predictions[i]):.2f}" for i in range(num_users)])

            output["average"] = [np.average(self.predictions[i]) for i in range(num_users)]

            print(output)

            return output

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

@app.route("/WhosApp", methods=["POST"])
def serverModelExecution():
    with lock:       
        data = request.get_json()
        response = execution.__rest_predict__(data["text"])

        # response = {"text": message}
    
    return jsonify(response), 200

if __name__ == "__main__":
    global execution 
    execution = modelExecution()
    print("Modello pronto all'uso\n[In ascolto sulla porta 5000]")
    app.run(port = 5000)