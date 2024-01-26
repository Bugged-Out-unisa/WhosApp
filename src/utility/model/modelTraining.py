import os
import json
import time
import logging
import calendar
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from utility.dataset.featureConstruction import featureConstruction
from sklearn.model_selection import train_test_split, cross_validate
from utility.exceptions import DatasetNotFoundError, ModelNotFoundError
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


class ModelTraining:
    __YES_CHOICES = ["yes", "y", ""]
    __NO_CHOICES = ["no", "n"]
    MODEL_PATH = "../models/"
    CONFIG_PATH = "../configs/"

    def __init__(
            self, 
            outputName: str = None, 
            model=None, 
            dataFrame: pd.DataFrame = None,
            configFile: str = None,
            retrain: bool = None
    ):
        self.__outputName = self.__check_output_name(outputName)
        self.__model = self.__check_model(model)
        self.__dataFrame = self.__check_dataframe(dataFrame)

        # Imposta configurazione di default
        self.__configFile = self.__check_config_file(configFile)
        self.__init_configs()

        self.__isToRetrain = retrain if retrain is not None else False
        self.__check_model_path()

    def __check_output_name(self, outputName: str) -> str:
        """Controlla se il nome del modello è stato inserito"""
        if outputName:
            return self.__check_extension_file(outputName, ".joblib", "model_")
        else:
            return "model_" + str(calendar.timegm(time.gmtime())) + ".joblib"

    def __check_config_file(self, configFile: str) -> str:
        """Controlla se il file di configurazione è stato inserito"""
        if configFile and os.path.exists(self.CONFIG_PATH + configFile):
            return self.__check_extension_file(configFile, ".json")
        else:
            return "config.json"

    @staticmethod
    def __check_model(model):
        """Controlla se il modello è stato inserito"""
        if model is not None:
            return model
        else:
            raise ModelNotFoundError("Inserire il modello da addestrare")

    @staticmethod
    def __check_dataframe(dataFrame):
        """Controlla se il dataframe è stato inserito"""
        if dataFrame is not None:
            return dataFrame
        else:
            raise DatasetNotFoundError("Inserire il dataset da usare per l'addestramento")

    def run(self):
        """Avvia il training del modello."""
        self.__check_duplicates()

        print("[INFO] Training del modello in corso...")
        return self.__model_training()

    def __init_configs(self):
        """Inizializza i parametri di configurazione."""
        # Leggi file di configurazione
        with open(self.CONFIG_PATH + self.__configFile, 'r') as f:
            print()
            features = json.load(f)

        # Estrai i nomi delle feature con valore falso (cioè le feature da filtrare)
        self.__features_disabled = [k for k, v in features.items() if not v]

        # LOGGING: Stampa le feature usate in fase di training
        logging.info(
            f"Feature usate in fase di training: \n" +
            "\n".join(f"\t {k}" for k, v in features.items() if v)
        )

        if "message_composition" in self.__features_disabled:
            self.__features_disabled.remove("message_composition")
            self.__features_disabled.extend(featureConstruction.POS_LIST)

    def __check_model_path(self):
        """Controlla se la cartella del modello esiste, altrimenti lo crea."""
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)

    @staticmethod
    def __check_extension_file(filename: str, ext: str, pre: str = ""):
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

    def __model_training(self, kfold: int = 0):
        """Applica random forest sul dataframe."""

        # LOGGING:: Stampa il nome del modello trainato
        logging.info(f"Modello trainato: {self.__outputName}")

        # Definisci le features (X) e il target (Y) cioè la variabile da prevedere
        X = self.__dataFrame.drop(['user'], axis=1)
        y = self.__dataFrame["user"]

        # Rimuovi le feature da filtrare (specificate nel file di configurazione)
        if self.__features_disabled:
            # Droppa tutte le colonne generate da bag of words (cioè con nomi numerici)
            if "bag_of_words" in self.__features_disabled:
                self.__features_disabled.extend([col for col in X.columns if col.isdigit()])

            X = X.drop(self.__features_disabled, axis=1, errors='ignore')

        # Applica feature scaling
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # TRAINING CON CROSS VALIDATION
        # (numero di fold di solito 5 o 10)
        if kfold > 0:
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            scores = cross_validate(self.__model, X, y, cv=kfold, scoring=scoring)

            # Stampa un report sulle metriche di valutazione del modello
            print(f"[INFO] Media delle metriche di valutazione dopo {kfold}-fold cross validation:")
            # LOGGING:: Stampa un report sulle metriche di valutazione del modello
            logging.info(f"Media delle metriche di valutazione dopo {kfold}-fold cross validation:")

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
        n = 20  # numero di feature per cmdlin
        m = 100 # numero di feature per logging

        feature_names = X.columns.tolist()  # Estrai i nomi di tutte le feature

        try:
            importances = self.__model.feature_importances_
            important_features = np.argsort(importances)[::-1]
            top_n_features_cmdline = important_features[:n]
            top_n_features_logging = important_features[:m]

            print("\n[INFO] Top {} feature più predittive:".format(n))

            # LOGGING:: Didascalia per le feature più predittive
            logging.info(f"Top {m} feature più predittive:")

            for i in top_n_features_cmdline:
                print(f"{feature_names[i]}: %0.5f" % importances[i])

            for ranking_index, i in enumerate(top_n_features_logging):
                # LOGGING:: Stampa le feature più predittive
                logging.info(f"\t {ranking_index+1}) {feature_names[i]}: %0.5f" % importances[i])

        except Exception:
            print("Il modello non verifica importanza delle features")


        # Crea una pipeline con lo scaler e il classificatore
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', self.__model)
        ])

        # Salva la pipeline (scaler e modello)
        dump(pipeline, self.MODEL_PATH + self.__outputName)

        # CREA CONFUSION MATRIX E ROC CURVE
        cm = confusion_matrix(y_test, predictions, labels=self.__model.classes_)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.__model.classes_)
        disp.plot()
        plt.show()

        # Convert labels to binary format
        y_test_bin = label_binarize(y_test, classes=self.__model.classes_)

        # Compute probabilities for each class
        y_score = self.__model.predict_proba(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(self.__model.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        plt.figure()
        for i in range(len(self.__model.classes_)):
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        return accuracy