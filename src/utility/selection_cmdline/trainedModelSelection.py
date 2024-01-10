import os
import logging
import pandas as pd
from simple_term_menu import TerminalMenu
from utility.exceptions import ExtensionError
import skops.io as skio


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

        model_selected = cls.__load_model(menu_entry_index, models)
        print(f"Modello selezionato: {models[menu_entry_index]}")

        # LOGGING:: Stampa il modello selezionato
        logging.info(f"Modello usato: {models[menu_entry_index]}")

        return model_selected

    @property
    def model(self):
        return self.__model

