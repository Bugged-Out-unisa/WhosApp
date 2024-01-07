import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from utility.model_list import models
from datasetCreation import datasetCreation
import skops.io as skio

# HOW TO USE:
# py modelTraining.py <outputName> <modelName> <datasetName> -r <*retrain>

#CHECKS IF SPECIFIED DATASET EXIST
    #(dataCreation.py return already existing DF)

#ELSE IT CREATES A NEW DATASET WITH SPECIFIED NAME from datasetCreation.py

#ONCE A DATASET IS GIVEN, IT TRAINS MODEL THEN PERSISTS IT


class ModelTraining:
    def __init__(self, outputName :str = None, model = None, dataFrame :pd.DataFrame = None, retrain :bool = None):
        self.MODEL_PATH = "../models/"
        self.__outputName = outputName
        self.__model = model
        self.__dataFrame = dataFrame
        self.__isToRetrain = retrain

        # Crea cartella dataset se non esiste
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)

        self.__main__()

    def __main__(self):
        #Se non passato per funzione controlla args da cmd line
        parser = argparse.ArgumentParser()

        # mandatory arguments
        parser.add_argument("outputName", help="Nome file del modello salvato")
        parser.add_argument("modelName", help="Nome del modello da trainare")
        parser.add_argument("datasetName", help="Nome del dataset da usare nel training")

        # optional arguments
        parser.add_argument("-r", "--retrain", help="Opzione di retraining", required=False)

        args = None

        if any(value is None for value in [self.__outputName, self.__model, self.__dataFrame]):
            datasetName = modelName = ""
            
            try:
                args = parser.parse_args()

                self.__outputName = args.outputName
                modelName = args.modelName
                datasetName = args.datasetName
            except:
                raise Exception("--Errore esecuzione da linea di comando--\nIl comando dovrebbe essere eseguito così:\npy modelTraining.py <outputName> <modelName> <datasetName> -r <*retrain>")
               
            try:
                self.__model = models[modelName]
            except:
                raise Exception("Modello specificato non trovato")

            try:
                self.__dataFrame = datasetCreation(datasetName, False).getDataframe()
            except:
                raise Exception("##MODELLO## ERRORE INDIVIDUAZIONE DATASET")

            if args.retrain:
                self.__isToRetrain = True if args.retrain == "retrain" else False

        yes_choices = ["yes","y"]
        no_choices = ["no", "n"]
        
        self.__outputName = self.__outputName if self.__outputName.endswith(".skops") else  self.__outputName + ".skops"

        # controllo in caso si voglia sovrascrivere comunque
        while(os.path.exists(self.MODEL_PATH + self.__outputName)  and not self.__isToRetrain):
            user_input = input("Il modello '{}' già esiste, sovrascriverlo? [Y/N]\n".format(self.__outputName))
            user_input = user_input.lower()

            if user_input in yes_choices:
                break
            elif user_input in no_choices:
                print("Operazione di Training annullata")
                return 1
            else:
                print(' ')
                continue

        self.__model_training()



    def __model_training(self):
        '''Applica random forest sul dataframe.'''
        # Definisci le features (X) e il target (Y) cioè la variabile da prevedere
        X = self.__dataFrame.drop(['user', 'message'], axis=1)
        y = self.__dataFrame["user"]

        # TRASFORMA IL MESSAGGIO IN UNA MATRICE DI FREQUENZA DELLE PAROLE (bag of words)
        # così il modello capisce le parole più utilizzate da un utente
        # ---------------------------------
        # Vettorizza le parole presenti nel messaggio
        vec = CountVectorizer()
        X_message = vec.fit_transform(self.__dataFrame['message'])

        # Unisci la matrice al dataframe
        df_words_count = pd.DataFrame(X_message.toarray(), columns=vec.get_feature_names_out())
        X = pd.concat([X, df_words_count], axis=1)
        # ---------------------------------

        # FEATURE SCALING
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X))

        # TRAINING CON CROSS VALIDATION
        cv  = 5 # numero di fold (di solito 5 o 10)
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        scores = cross_validate(self.__model, X, y, cv=cv, scoring=scoring)

        # Stampa un report sulle metriche di valutazione del modello
        print(f"[INFO] Media delle metriche di valutazione dopo {cv}-fold cross validation:")
        indexes = list(scores.keys())

        for index in indexes:
            print(f"{index}: %0.2f (+/- %0.2f)" % (scores[index].mean(), scores[index].std() * 2))

        # TRAINING CON SPLIT CLASSICO
        test_size = 0.2 # percentuale del dataset di test dopo lo split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.__model.fit(X_train, y_train)
        predictions = self.__model.predict(X_test)

        # Genera un report per il modello addestrato
        print(f"\n[INFO] Report con {int((1-test_size)*100)}% training set e {int(test_size*100)}% test set:")
        
        # Calcola l'accuratezza del modello
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {round(accuracy, 2)}\n')

        # Stampa il classification report
        report = classification_report(y_test, predictions)
        print(report)

        # Stampa le feature più predittive
        n = 20 # numero di feature
        print("\n[INFO] Top {} feature più predittive:".format(n))

        feature_names = X.columns.tolist() # Estrai i nomi di tutte le feature

        try:
            importances = self.__model.feature_importances_
            important_features = np.argsort(importances)[::-1]
            top_n_features = important_features[:n]

            for i in top_n_features:
                print(f"{feature_names[i]}: %0.5f" % importances[i])
        except:
            print("Il modello non verifica importanza delle features")
        
        skio.dump(self.__model, self.MODEL_PATH + self.__outputName)

if __name__ == "__main__":
    ModelTraining()