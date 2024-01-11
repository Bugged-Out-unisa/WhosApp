import os
import time
import logging
import calendar
import numpy as np
import pandas as pd
import skops.io as skio
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler
from joblib import dump


class ModelTraining:
    __YES_CHOICES = ["yes", "y", ""]
    __NO_CHOICES = ["no", "n"]
    MODEL_PATH = "../models/"

    def __init__(self, outputName: str = None, model=None, dataFrame: pd.DataFrame = None, retrain: bool = None):
        if outputName:
            self.__outputName = self.__check_extension_file(outputName, ".skops", "model_")
        else:
            self.__outputName = "model_" + str(calendar.timegm(time.gmtime())) + ".skops"

        if model is not None:
            self.__model = model
        else:
            raise ValueError("Inserire il modello da addestrare")

        if dataFrame is not None:
            self.__dataFrame = dataFrame
        else:
            raise ValueError("Inserire il dataset da usare per l'addestramento")

        self.__isToRetrain = retrain if retrain is not None else False
        self.__check_model_path()

    def run(self):
        """Avvia il training del modello."""
        self.__check_duplicates()

        print("[INFO] Training del modello in corso...\n")
        self.__model_training()

    def __check_model_path(self):
        """Controlla se il path del modello esiste, altrimenti lo crea."""
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)

    @staticmethod
    def __check_extension_file(filename: str, ext: str, pre: str):
        """Controlla se l'estensione del file è quella specificata"""
        if not filename.endswith(ext):
            filename += ext
        if not filename.startswith(pre):
            filename = pre + filename
        return filename

    def __check_duplicates(self):
        # Controllo in caso si voglia sovrascrivere comunque
        while os.path.exists(self.MODEL_PATH + self.__outputName) and not self.__isToRetrain:

            user_input = input("Il modello '{}' già esiste, sovrascriverlo? [Y/N]\n".format(self.__outputName))
            user_input = user_input.lower()

            if user_input in self.__YES_CHOICES:
                break
            elif user_input in self.__NO_CHOICES:
                print("Operazione di Training annullata")
                return 1
            else:
                print('Inserire \"Y\" o \"N\"')
                continue

    def __model_training(self):
        """Applica random forest sul dataframe."""

        # LOGGING:: Stampa il nome del modello trainato
        logging.info(f"Modello trainato: {self.__outputName}")

        # Definisci le features (X) e il target (Y) cioè la variabile da prevedere
        X = self.__dataFrame.drop(['user'], axis=1)
        y = self.__dataFrame["user"]

        # Applica feature scaling
        X = self.__feature_scaling(X)

        # TRAINING CON CROSS VALIDATION
        cv = 5  # numero di fold (di solito 5 o 10)
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        scores = cross_validate(self.__model, X, y, cv=cv, scoring=scoring)

        # Stampa un report sulle metriche di valutazione del modello
        print(f"[INFO] Media delle metriche di valutazione dopo {cv}-fold cross validation:")
        # LOGGING:: Stampa un report sulle metriche di valutazione del modello
        logging.info(f"Media delle metriche di valutazione dopo {cv}-fold cross validation:")

        indexes = list(scores.keys())

        for index in indexes:
            # LOGGING:: Stampa le metriche di valutazione del modello
            logging.info(f"\t{index}: %0.2f (+/- %0.2f)" % (scores[index].mean(), scores[index].std() * 2))
            print(f"\t{index}: %0.2f (+/- %0.2f)" % (scores[index].mean(), scores[index].std() * 2))

        # TRAINING CON SPLIT CLASSICO
        test_size = 0.2  # percentuale del dataset di test dopo lo split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.__model.fit(X_train, y_train)
        predictions = self.__model.predict(X_test)

        # Genera un report per il modello addestrato
        print(f"\n[INFO] Report con {int((1 - test_size) * 100)}% training set e {int(test_size * 100)}% test set:")

        # Calcola l'accuratezza del modello
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {round(accuracy, 2)}\n')
        # LOGGING:: Stampa l'accuratezza del modello
        logging.info(f"Accuracy: {round(accuracy, 2)}\n")

        # Stampa il classification report
        report = classification_report(y_test, predictions)
        print(report)
        # LOGGING:: Stampa il classification report
        logging.info(report)

        # Stampa le feature più predittive
        n = 20  # numero di feature
        print("\n[INFO] Top {} feature più predittive:".format(n))

        feature_names = X.columns.tolist()  # Estrai i nomi di tutte le feature

        try:
            importances = self.__model.feature_importances_
            important_features = np.argsort(importances)[::-1]
            top_n_features = important_features[:n]

            for i in top_n_features:
                print(f"{feature_names[i]}: %0.5f" % importances[i])
        except Exception:
            print("Il modello non verifica importanza delle features")

        skio.dump(self.__model, self.MODEL_PATH + self.__outputName)

    def __feature_scaling(self, df: pd.DataFrame):
        """     
            Scala le feature in un range da 0 a 1 e 
            salva lo scaler in un file avente lo stesso nome del modello.
        """
        # Applica minmax scaling
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        # Salva lo scaler in un file avente lo stesso nome del modello
        # (altrimenti in modelExecution non si potrebbe scalare il messaggio in input)
        scaler_file_path = ModelTraining.get_scaler_path(self.MODEL_PATH + self.__outputName)
        dump(scaler, scaler_file_path)

        return df

    @staticmethod
    def get_scaler_path(model_path: str):
        """Ottieni il path dello scaler associato al modello."""
        return model_path.replace(".skops", "_scaler.joblib")