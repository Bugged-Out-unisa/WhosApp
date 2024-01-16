import os
import time
import logging
import calendar
from typing import Tuple, Any

import pandas as pd
from utility.dataset.extractChat import ExtractChat
from utility.dataset.rawDataReader import rawDataReader
from utility.dataset.dataFrameProcess import DataFrameProcessor
from utility.dataset.featureConstruction import featureConstruction


class datasetCreation:
    DATA_PATH = "../rawdata"
    DATASET_PATH = "../datasets/"
    CONFIG_PATH = "../configs/"

    def __init__(
            self,
            dataset_name: str = None,
            config_file: str = "config.json",
            alias_file: str = None,
            other_user: str = None,
            remove_other: bool = False,
            refactor: bool = False
    ):
        self.__dataset_name = self.__check_dataset_name(dataset_name)
        self.__config_file = self.__check_config_file(config_file)
        self.__alias_file, self.__other_user, self.__remove_other = self.__check_aliases_file(alias_file, other_user, remove_other)

        # Controlla se refactor è stato inserito
        if not isinstance(refactor, bool):
            raise TypeError("refactor deve essere un booleano")
        self.__isToRefactor = True if refactor else False

        self.__dataFrame = None
        self.__check_dataset_path()

    def __check_config_file(self, config_file: str) -> str:
        """Controlla se il file di configurazione è stato inserito"""
        if config_file and os.path.exists(self.CONFIG_PATH + config_file):
            return self.__check_extension_file(config_file, ".json")
        else:
            return "config.json"

    def __check_aliases_file(
            self,
            aliases_file: str,
            other_user: str,
            remove_other: bool
    ) -> tuple[str, str, bool] | tuple[None, None, bool]:
        """Controlla se il file di configurazione è stato inserito"""
        if aliases_file and os.path.exists(self.CONFIG_PATH + aliases_file):
            return self.__check_extension_file(aliases_file, ".json"), other_user, remove_other
        else:
            return None, None, False

    def __check_dataset_name(self, name: str) -> str:
        """Controlla se il nome del dataset è stato inserito"""
        if name:
            return self.__check_extension_file(name, ".parquet")
        else:
            timestamp = calendar.timegm(time.gmtime())
            return "dataset_" + str(timestamp) + ".parquet"

    @staticmethod
    def __check_extension_file(filename: str, ext: str):
        """Controlla se l'estensione del file è quella specificata"""
        if not filename.endswith(ext):
            filename += ext
        return filename

    @classmethod
    def __check_dataset_path(cls):
        """Crea cartella dataset se non esiste"""
        if not os.path.exists(cls.DATASET_PATH):
            os.makedirs(cls.DATASET_PATH)

    def run(self):
        """Avvia la creazione del dataset."""

        # LOGGING:: Stampa il nome del dataset
        logging.info(f"Dataset name: {self.__dataset_name}")

        # se il file non esiste oppure è richiesta un ricreazione di esso, esegue tutte le operazioni
        if not os.path.exists(self.DATASET_PATH + self.__dataset_name) or self.__isToRefactor:

            print("\n[LOADING] Leggendo le chat dai file grezzi...")
            rawdata = rawDataReader(self.DATA_PATH).read_all_files()

            print("\n[LOADING] Estraendo informazioni dai dati grezzi...")
            if self.__alias_file:
                dates, users, messages = ExtractChat(
                    rawdata,
                    self.CONFIG_PATH + self.__alias_file,
                    self.__other_user
                ).extract()
            else:
                dates, users, messages = ExtractChat(rawdata).extract()

            print("\n[LOADING] Creando il dataframe e applicando data cleaning e undersampling...")
            self.__dataFrame = DataFrameProcessor(dates, users, messages, self.__other_user, self.__remove_other).get_dataframe()

            print("\n[LOADING] Applicando feature construction...")
            self.__dataFrame = featureConstruction(
                self.__dataFrame,
                self.DATASET_PATH + self.__dataset_name,
                self.CONFIG_PATH + self.__config_file
            ).get_dataframe()

            print("[INFO] Dataset creato con successo.")

        # se il file esiste già allora salva il DF esistente
        elif os.path.exists(self.DATASET_PATH + self.__dataset_name):
            print("[INFO] Trovato dataset esistente")
            self.__dataFrame = pd.read_parquet(self.DATASET_PATH + self.__dataset_name)

    # -------- Getter --------
    @property
    def dataFrame(self):
        return self.__dataFrame
