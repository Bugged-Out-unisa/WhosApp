import os
import logging
import pandas as pd
from simple_term_menu import TerminalMenu
from utility.exceptions import ExtensionError
import skops.io as skio
from joblib import load
from utility.model.modelTraining import ModelTraining


class TrainedModelSelection:
    MODEL_PATH = "../models/"

    def __init__(self):
        self.__model, self.__scaler = self.__select_model()

    @classmethod
    def __show_models(cls):
        print("Elenco dei modelli disponibili:")
        models = os.listdir(cls.MODEL_PATH)
        return models

    @classmethod
    def __load_model(cls, index, models):
        if 0 <= index < len(models):
            selected_model = models[index]

            if not selected_model.endswith(".skops"):
                raise ExtensionError("Il modello deve essere in formato .skops")

            path = f"{cls.MODEL_PATH}{selected_model}"

            # Carica il model
            model = skio.load(path)
            return model
        else:
            raise ValueError("ID del modello non valido.")

    @classmethod
    def __select_model(cls):
        models = cls.__show_models()
        menu = TerminalMenu(models)
        menu_entry_index = menu.show()

        # Ottieni model
        model_name = models[menu_entry_index]
        model_selected = cls.__load_model(menu_entry_index, models)

        # Ottieni scaler
        scaler_path = ModelTraining.get_scaler_path(model_name)
        scaler_name = os.path.basename(scaler_path)
        scaler_selected = load(scaler_path)

        print(f"Modello selezionato: {model_name}")
        print(f"Scaler selezionato: {scaler_name}")

        # LOGGING:: Stampa il modello (e lo scaler) selezionato
        logging.info(f"Modello usato: {model_name}")
        logging.info(f"Scaler usato: {scaler_name}")

        return model_selected, scaler_selected

    @property
    def model(self):
        return self.__model
    
    @property
    def scaler(self):
        return self.__scaler