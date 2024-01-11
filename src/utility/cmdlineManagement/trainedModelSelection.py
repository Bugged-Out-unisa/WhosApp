import os
import logging
from simple_term_menu import TerminalMenu
from utility.exceptions import ExtensionError
from joblib import load


class TrainedModelSelection:
    MODEL_PATH = "../models/"

    def __init__(self):
        self.__model = self.__select_model()

    @classmethod
    def __show_models(cls):
        print("Elenco dei modelli disponibili:")
        models = os.listdir(cls.MODEL_PATH)
        return models

    @classmethod
    def __load_model(cls, index, models):
        if 0 <= index < len(models):
            selected_model = models[index]

            if not selected_model.endswith(".joblib"):
                raise ExtensionError("Il modello deve essere in formato .joblib")

            path = f"{cls.MODEL_PATH}{selected_model}"

            # Carica il model
            return load(path)
        else:
            raise ValueError("ID del modello non valido.")

    @classmethod
    def __select_model(cls):
        models = cls.__show_models()
        menu = TerminalMenu(models)
        menu_entry_index = menu.show()

        # Ottieni model
        model_name = models[menu_entry_index]

        print(f"Modello selezionato: {model_name}")
        # LOGGING:: Stampa il modello (e lo scaler) selezionato
        logging.info(f"Modello usato: {model_name}")

        return cls.__load_model(menu_entry_index, models)

    @property
    def model(self):
        return self.__model
    
    @property
    def scaler(self):
        return self.__scaler