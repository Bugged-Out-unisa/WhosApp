import sys
import os
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
# py modelTraining.py <outputName> <modelName> <datasetName> <*retrain>

#CHECKS IF SPECIFIED DATASET EXIST
    #(dataCreation.py return already existing DF)

#ELSE IT CREATES A NEW DATASET WITH SPECIFIED NAME from datasetCreation.py

#ONCE A DATASET IS GIVEN, IT TRAINS MODEL THEN PERSISTS IT


class ModelTraining:
    def __init__(self, outputName :str = None, model = None, dataFrame :pd.DataFrame = None, retrain : bool = None):
        self.MODEL_PATH = "../models/"
        self.__outputName = outputName
        self.__model = model
        self.__modelName = ""
        self.__dataFrame = dataFrame
        self.__isToRetrain = retrain
        self.__main__()

    def __main__(self):
        if self.__outputName is None:
            try:
                self.__outputName = sys.argv[1]
            except:
                raise Exception("Non è stato inserito un nome per il modello") 

        if self.__model is None:
            try:
                self.__modelName = sys.argv[2]
            except:
                raise Exception("Nome modello non specificato...")     
               
            try:
                self.__model = models[self.__modelName]
            except:
                raise Exception("Modello specificato non trovato")


        if self.__dataFrame is None:
            try:
                datasetName = sys.argv[3]
                
                self.__dataFrame = datasetCreation(datasetName, False).getDataframe()
            except:
                raise Exception("##MODELLO## ERRORE INDIVIDUAZIONE DATASET")
        
        if self.__isToRetrain is None:
            try:
                self.__isToRetrain = True if sys.argv[4] == "retrain" else False

            except:
                self.__isToRetrain = False        

        yes_choices = ["yes","y"]
        no_choices = ["no", "n"]
        
        self.__outputName = self.__outputName if self.__outputName.endswith(".skops") else  self.__outputName + ".skops"

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
        X = self.__dataFrame.drop(['user', 'date', 'message'], axis=1) # tutto tranne le colonne listate
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