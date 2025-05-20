import numpy as np
import pandas as pd
import re
import requests
import json
from threading import Lock
from flask import Flask, jsonify, request
from utility.dataset.featureConstruction import featureConstruction
from utility.dataset.embeddingsCreation import EmbeddingsCreation
from utility.cmdlineManagement.trainedModelSelection import TrainedModelSelection
from new_training import build_single_message_meta

app = Flask(__name__)
lock = Lock()

class modelExecution:
    """
        Classe che si occupa di eseguire un modello allenato e, 
        per ogni utente, di predire con quanta probabilità abbia scritto il messaggio in input
        sia per quel singolo messaggio che per l'intera conversazione.
    """

    MODEL_PATH = "../models/"
    CONFIG_PATH = "../configs/"

    
    def __init__(self, pipeline = None):
        self.model_name = None

        if pipeline is None:
            print("\n --- FEATURE ---")
            self.model_name, pipeline = TrainedModelSelection().select_model()
            self.model_name = re.sub(r'model_|\.joblib', '', self.model_name)

        json_file = json.load(open(f"{self.CONFIG_PATH}frontend_users.json", "r"))

        self.mapped_users = list(json_file[self.model_name].values())

        # Carica il modello e lo scaler dalla pipeline
        self._trained_feature_model = pipeline.named_steps['classifier']
        self.__scaler = pipeline.named_steps['scaler']

        # Carica modello CNN
        print("\n --- EMBEDDINGS ---")
        self.__trained_embeddings_model = TrainedModelSelection().select_model()

        # Carica meta-learner
        print("\n --- META-LEARNER ---")
        _, self.__trained_meta_model = TrainedModelSelection().select_model()

        self.predictions = [[] for _ in range(self._trained_feature_model.n_classes_)]

    def dataframe_for_messages(self, message):
        # Applica feature construction al messaggio
        df = featureConstruction(dataFrame=pd.DataFrame({"message": message}),\
                                 datasetPath="./", saveDataFrame=False)\
                                .get_dataframe()
        
        df.drop("message_id", axis=1, errors='ignore')
        
        # Applica scaling
        return pd.DataFrame(self.__scaler.transform(df), columns=df.columns)
    
    def embeddings_for_message(self, message):
        df = EmbeddingsCreation(dataFrame=pd.DataFrame({"message": message}),\
                                datasetPath="./", saveDataFrame=False\
                                ).get_dataframe()
        
        df.drop("message_id", axis=1, errors='ignore')

        return df

    def __rest_predict__(self, data):
        if request.method == "POST":

            num_users = self._trained_feature_model.n_classes_

            output = {
                "mappedUsers": self.mapped_users,
                "single": [],
                "average": []
            }

            # Ottieni il messaggio da input
            #message = input("\nScrivi un messaggio:\n")

            # Crea il dataframe e costruisci le feature su quel messaggio
            feature_df = self.dataframe_for_messages([data])
            embeddings_df = self.embeddings_for_message([data])

            # Ottieni probabilità per ogni utente
            # [0] perché resituisce una lista di previsioni (come se si aspettasse più messaggi)
            feature_prob = self._trained_feature_model.predict_proba(feature_df)[0]
            embeddings_prob = self.__trained_embeddings_model.predict_proba(embeddings_df)[0]

            meta_df = build_single_message_meta(feature_prob, embeddings_prob)
            
            users_prob = self.__trained_meta_model.predict_proba(meta_df)

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

    
execution = modelExecution()

@app.route("/WhosApp", methods=["POST"])
def serverModelExecution():
    with lock:       
        data = request.get_json()
        response = execution.__rest_predict__(data["text"])
    
    return jsonify(response), 200

if __name__ == "__main__":
    print("Modello pronto all'uso\n[In ascolto sulla porta 5000]")
    app.run(port = 5000)
