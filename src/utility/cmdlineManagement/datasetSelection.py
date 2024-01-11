import os
import logging
import pandas as pd
from simple_term_menu import TerminalMenu
from utility.exceptions import ExtensionError


class DatasetSelection:
    DATASET_PATH = "../datasets/"

    def __init__(self):
        self.__dataset = self.__select_dataset()

    @classmethod
    def __show_datasets(cls):
        print("Elenco dei dataset disponibili:")
        datasets = [file for file in os.listdir(cls.DATASET_PATH) if file.endswith(".parquet")]

        # Ordina i dataset in base alla data di creazione
        datasets = sorted(datasets, key=lambda x: os.path.getctime(os.path.join(cls.DATASET_PATH, x)), reverse=True)

        return datasets

    @classmethod
    def __load_dataset(cls, index, datasets):
        if 0 <= index < len(datasets):
            selected_dataset = datasets[index]

            if not selected_dataset.endswith(".parquet"):
                raise ExtensionError("Il dataset deve essere in formato .parquet")

            path = f"{cls.DATASET_PATH}{selected_dataset}"

            # Carica il dataset come DataFrame di pandas
            df = pd.read_parquet(path)
            return df
        else:
            raise ValueError("ID del dataset non valido.")

    @classmethod
    def __select_dataset(cls):
        datasets = cls.__show_datasets()
        menu = TerminalMenu(datasets)
        menu_entry_index = menu.show()

        dataset_selected = cls.__load_dataset(menu_entry_index, datasets)
        print(f"Dataset selezionato: {datasets[menu_entry_index]}")

        # LOGGING:: Stampa il dataset selezionato
        logging.info(f"Dataset usato per il training: {datasets[menu_entry_index]}")

        return dataset_selected

    @property
    def dataset(self):
        return self.__dataset
